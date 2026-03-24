"""
Usage:
  python ga_threshold_only.py --grid-size 10 --n-ranks 1
  python ga_threshold_only.py --grid-size 20 --n-ranks 1 --cs2-config ./configs/grid20.conf
  python ga_threshold_only.py --grid-size 50 --n-ranks 1 --cs2-config ./configs/grid50.conf
"""

import random, copy, os, shutil, subprocess, re, sys, argparse, json, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

def get_config():
    p = argparse.ArgumentParser(description="Threshold-Only GA — CancerSim2")
    p.add_argument("--grid-size", type=int, default=10)
    p.add_argument("--containment-pct", type=float, default=0.50)
    p.add_argument("--warmup-period", type=int, default=4000)
    p.add_argument("--time-steps", type=int, default=8760)
    p.add_argument("--sample-period", type=int, default=168)
    p.add_argument("--dosage-max", type=float, default=1000.0)
    p.add_argument("--pop-size", type=int, default=20)
    p.add_argument("--generations", type=int, default=25)
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--n-ranks", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-preflight", action="store_true")
    p.add_argument("--cs2-binary", type=str, default=None)
    p.add_argument("--cs2-config", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="./ga_output_threshold")
    args = p.parse_args()

    grid_n   = args.grid_size
    capacity = grid_n ** 3

    # Auto-scale timeout
    if grid_n <= 10:
        timeout = 300
    elif grid_n <= 50:
        timeout = 1800
    else:
        timeout = 7200

    min_threshold = max(10, int(0.02 * capacity))

    cfg = {
        "GRID_N":           grid_n,
        "GRID_CAPACITY":    capacity,
        "CONTAINMENT_PCT":  args.containment_pct,
        "CONTAINMENT_CEIL": int(args.containment_pct * capacity),
        "MIN_THRESHOLD":    min_threshold,

        "TIME_STEPS":       args.time_steps,
        "SAMPLE_PERIOD":    args.sample_period,
        "WARMUP_PERIOD":    args.warmup_period,
        "NUM_SAMPLES":      args.time_steps // args.sample_period,

        "DOSAGE_MIN":       0.0,
        "DOSAGE_MAX":       args.dosage_max,

        "POP_SIZE":         args.pop_size,
        "GENERATIONS":      args.generations,
        "TOURNAMENT_SIZE":  3,
        "ELITISM":          2,

        # Mutation: 50% per gene → expected 1 change per 2-gene individual
        "MUTATION_RATE_PER_GENE": 0.50,

        # Fitness 
        # f = -B - 0.5*D - 100*O - 50*F + 10*H
        "DANGER_CEIL":      int(0.50 * capacity),
        "HOLIDAY_BONUS":    10,
        "BURDEN_WEIGHT":    1.0,
        "TOXICITY_WEIGHT":  0.5,
        "OVERFLOW_PENALTY": 100.0,
        "TERMINAL_WEIGHT":  50,

        # Outcome classification
        "CURE_WINDOW":        5,
        "FATAL_CEIL":         int(0.90 * capacity),
        "FATAL_WINDOW":       3,
        "PROGRESSION_WINDOW": 3,

        # Infra
        "MAX_WORKERS":  args.max_workers,
        "N_RANKS":      args.n_ranks,
        "SIM_TIMEOUT":  timeout,
        "RANDOM_SEED":  args.seed,

        # Paths
        "CS2_BINARY":   args.cs2_binary or os.environ.get("CS2_BINARY", "./bin/CS2"),
        "CONFIG_PATH":  args.cs2_config or os.environ.get("CS2_CONFIG", "./configs/default.conf"),
        "OUTPUT_DIR":   args.output_dir,
        "SAMPLES_DIR":  os.path.join(args.output_dir, "samples"),
        "PHENOS_DIR":   os.path.join(args.output_dir, "phenos"),
        "ROUTINES_DIR": os.path.join(args.output_dir, "routines"),
    }
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# GENOME — fixed 2-gene: [dosage, cell_threshold]
# ─────────────────────────────────────────────────────────────────────────────

class Genome:
    """
    Fixed-length genome with exactly 2 genes:
      gene[0] = dosage        ∈ [0, DOSAGE_MAX]
      gene[1] = cell_threshold ∈ [MIN_THRESHOLD, GRID_CAPACITY]

    The drug rule: apply `dosage` whenever alive cells >= cell_threshold.
    """
    def __init__(self, dosage=0.0, cell_threshold=100):
        self.dosage = dosage
        self.cell_threshold = cell_threshold

    def to_dict(self):
        return {"dosage": self.dosage, "cell_threshold": self.cell_threshold}

    def __repr__(self):
        return f"Genome(dose={self.dosage:.1f}, thresh={self.cell_threshold})"


def random_genome(cfg):
    return Genome(
        dosage=round(random.uniform(cfg["DOSAGE_MIN"], cfg["DOSAGE_MAX"]), 2),
        cell_threshold=random.randint(cfg["MIN_THRESHOLD"], cfg["GRID_CAPACITY"]),
    )


# ─────────────────────────────────────────────────────────────────────────────
# TUMOR TRAJECTORY
# ─────────────────────────────────────────────────────────────────────────────

class TumorTrajectory:
    def __init__(self, timesteps, cell_counts, cumulative_dose,
                 dose_per_step=None, cfg=None):
        self.timesteps       = timesteps
        self.cell_counts     = cell_counts
        self.cumulative_dose = cumulative_dose
        self.dose_per_step   = dose_per_step or []
        self._danger = cfg["DANGER_CEIL"] if cfg else 500

    @property
    def peak(self):
        return max(self.cell_counts) if self.cell_counts else 0

    @property
    def final(self):
        return self.cell_counts[-1] if self.cell_counts else 0

    @property
    def burden(self):
        return sum(self.cell_counts)

    @property
    def managed_steps(self):
        return sum(1 for c in self.cell_counts if 0 < c <= self._danger)

    @property
    def overflow_steps(self):
        return sum(1 for c in self.cell_counts if c > self._danger)

    @property
    def holiday_steps(self):
        if not self.dose_per_step:
            return 0
        return sum(1 for d in self.dose_per_step if d == 0.0)

    @property
    def num_cycles(self):
        if not self.dose_per_step or len(self.dose_per_step) < 2:
            return 0
        return sum(1 for i in range(1, len(self.dose_per_step))
                   if self.dose_per_step[i-1] > 0 and self.dose_per_step[i] == 0)

    def classify_outcome(self, cfg):
        cc = self.cell_counts
        ds = self.dose_per_step
        n = len(cc)
        if n == 0:
            return "FATAL", 0

        cure_w = cfg.get("CURE_WINDOW", 5)
        fatal_ceil = cfg.get("FATAL_CEIL", int(0.90 * cfg["GRID_CAPACITY"]))
        fatal_w = cfg.get("FATAL_WINDOW", 3)
        prog_w = cfg.get("PROGRESSION_WINDOW", 3)

        # Check CURE
        consecutive_zero = 0
        cure_time = None
        for i in range(n):
            if cc[i] == 0:
                consecutive_zero += 1
                if consecutive_zero >= cure_w:
                    cure_time = self.timesteps[i] if self.timesteps else i
                    break
            else:
                consecutive_zero = 0

        # Check FATAL
        consecutive_fatal = 0
        fatal_time = None
        for i in range(n):
            if cc[i] > fatal_ceil:
                consecutive_fatal += 1
                if consecutive_fatal >= fatal_w:
                    fatal_time = self.timesteps[i] if self.timesteps else i
                    break
            else:
                consecutive_fatal = 0

        # Check PROGRESSIVE
        progressive_time = None
        if ds and len(ds) >= n:
            consecutive_prog = 0
            for i in range(1, n):
                if cc[i] > cc[i-1] and (ds[i] > 0 if i < len(ds) else False):
                    consecutive_prog += 1
                    if consecutive_prog >= prog_w:
                        progressive_time = self.timesteps[i] if self.timesteps else i
                        break
                else:
                    consecutive_prog = 0

        if fatal_time is not None:
            if cure_time is None or fatal_time <= cure_time:
                return "FATAL", fatal_time
        if progressive_time is not None:
            if cure_time is None or progressive_time < cure_time:
                return "PROGRESSIVE", progressive_time
        if cure_time is not None:
            return "CURE", cure_time
        return "MANAGED", None


# ─────────────────────────────────────────────────────────────────────────────
# CELL COUNTING
# ─────────────────────────────────────────────────────────────────────────────

def count_alive_cells(filepath):
    count = 0
    with open(filepath) as f:
        next(f)
        for line in f:
            value = line.rstrip().rsplit(",", 1)[-1]
            if not value.startswith("-1|"):
                count += 1
    return count


# ─────────────────────────────────────────────────────────────────────────────
# DRUG ROUTINE — single threshold rule
# ─────────────────────────────────────────────────────────────────────────────

def write_routine(genome, filepath, cfg):
    warmup = cfg["WARMUP_PERIOD"]
    sample_period = cfg["SAMPLE_PERIOD"]
    dosage = genome.dosage
    threshold = genome.cell_threshold

    script = f"""import sys, os

DOSAGE = {dosage}
THRESHOLD = {threshold}
WARMUP = {warmup}
SAMPLE_PERIOD = {sample_period}

_cumulative_dose = 0.0
_dose_log = []
_interval_max_dose = 0.0

def drug_func(timestep, drug_idx, num_cells, num_resistant):
    global _cumulative_dose, _interval_max_dose
    if isinstance(num_resistant, (list, tuple)):
        num_resistant = sum(num_resistant)
    num_cells = int(num_cells or 0)

    # Log interval max at each sample boundary
    if timestep % SAMPLE_PERIOD == 0 and timestep > 0:
        _dose_log.append(_interval_max_dose)
        _interval_max_dose = 0.0

    # No drug during warmup
    if timestep < WARMUP:
        return 0.0

    # Single threshold rule: apply DOSAGE when cells >= THRESHOLD
    dosage = 0.0
    if num_cells >= THRESHOLD:
        dosage = DOSAGE
        # Resistance scaling
        if num_cells > 0 and num_resistant > 0:
            res_frac = num_resistant / num_cells
            dosage *= max(0.3, 1.0 - 0.7 * res_frac)

    _cumulative_dose += dosage
    _interval_max_dose = max(_interval_max_dose, dosage)
    return float(dosage)

def get_drug_type(drug_idx):
    return 0

def finalize():
    _dose_log.append(_interval_max_dose)
    try:
        with open(os.environ.get("CS2_DOSE_LOG", "cumulative_dose.txt"), "w") as f:
            f.write(str(_cumulative_dose) + "\\n")
            f.write(",".join(str(d) for d in _dose_log) + "\\n")
    except Exception:
        pass
"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(script)


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(routine_path, samples_dir, cfg):
    if os.path.exists(samples_dir):
        shutil.rmtree(samples_dir)
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(cfg["PHENOS_DIR"], exist_ok=True)

    script_name = os.path.splitext(os.path.basename(routine_path))[0]
    dose_log = os.path.join(samples_dir, "cumulative_dose.txt")

    command = [
        "upcxx-run", "-n", str(cfg["N_RANKS"]), cfg["CS2_BINARY"],
        "--config", cfg["CONFIG_PATH"],
        "--samples_dir", samples_dir,
        "--phenos_dir", cfg["PHENOS_DIR"],
        "--routines_dir", cfg["ROUTINES_DIR"],
        "--routine_script", script_name,
    ]
    env = os.environ.copy()
    env["CS2_DOSE_LOG"] = dose_log

    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=cfg["SIM_TIMEOUT"], env=env)
    except subprocess.TimeoutExpired:
        return TumorTrajectory([], [], 0.0, cfg=cfg)
    if result.returncode != 0:
        return TumorTrajectory([], [], 0.0, cfg=cfg)

    cell_files = sorted(
        glob.glob(os.path.join(samples_dir, "cells_*.csv")),
        key=lambda f: int(re.search(r'cells_(\d+)\.csv', f).group(1))
    )
    timesteps, counts = [], []
    for fp in cell_files:
        timesteps.append(int(re.search(r'cells_(\d+)\.csv', fp).group(1)))
        counts.append(count_alive_cells(fp))

    cum_dose = 0.0
    dose_per_step = []
    if os.path.exists(dose_log):
        try:
            with open(dose_log) as f:
                lines = f.readlines()
                cum_dose = float(lines[0].strip())
                if len(lines) > 1 and lines[1].strip():
                    dose_per_step = [float(x) for x in lines[1].strip().split(",")]
        except (ValueError, IOError, IndexError):
            pass

    return TumorTrajectory(timesteps, counts, cum_dose, dose_per_step, cfg)


# ─────────────────────────────────────────────────────────────────────────────
# FITNESS — mechanism-agnostic
# ─────────────────────────────────────────────────────────────────────────────

def compute_fitness(traj, cfg):
    if not traj.cell_counts:
        return -999999.0
    return (
        - cfg["BURDEN_WEIGHT"]    * traj.burden
        - cfg["TOXICITY_WEIGHT"]  * traj.cumulative_dose
        - cfg["OVERFLOW_PENALTY"] * traj.overflow_steps
#       - cfg["TERMINAL_WEIGHT"]  * traj.final
        + cfg["HOLIDAY_BONUS"]    * traj.holiday_steps
    )


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE + PREFLIGHT
# ─────────────────────────────────────────────────────────────────────────────

def _write_helper(path, body, cfg):
    os.makedirs(cfg["ROUTINES_DIR"], exist_ok=True)
    sp = cfg["SAMPLE_PERIOD"]
    with open(path, "w") as f:
        f.write(f"import sys, os\n_d=0.0\n_dl=[]\nSP={sp}\n" + body +
                "def get_drug_type(d): return 0\n"
                "def finalize():\n"
                "    try:\n"
                "        with open(os.environ.get('CS2_DOSE_LOG',"
                "'cumulative_dose.txt'),'w') as f:\n"
                "            f.write(str(_d)+'\\n')\n"
                "            f.write(','.join(str(x) for x in _dl)+'\\n')\n"
                "    except: pass\n")


def run_baseline(cfg):
    path = os.path.join(cfg["ROUTINES_DIR"], "baseline.py")
    _write_helper(path,
        "def drug_func(t,d,n,r):\n"
        "    if t%SP==0: _dl.append(0.0)\n"
        "    return 0.0\n", cfg)
    return run_simulation(path, os.path.join(cfg["SAMPLES_DIR"], "baseline"), cfg)


def run_maxdose(cfg):
    path = os.path.join(cfg["ROUTINES_DIR"], "maxdose.py")
    dmax = cfg["DOSAGE_MAX"]
    _write_helper(path,
        f"def drug_func(t,d,n,r):\n"
        f"    global _d; _d+={dmax}\n"
        f"    if t%SP==0: _dl.append({dmax})\n"
        f"    return {dmax}\n", cfg)
    return run_simulation(path, os.path.join(cfg["SAMPLES_DIR"], "maxdose"), cfg)


def preflight(cfg):
    cap = cfg["GRID_CAPACITY"]
    print("=" * 60)
    print(f"PREFLIGHT — Grid: {cfg['GRID_N']}³ = {cap:,} points")
    print(f"Danger ceiling: {cfg['DANGER_CEIL']:,} (50%)")
    print(f"Warmup: {cfg['WARMUP_PERIOD']} steps")
    print(f"Min threshold: {cfg['MIN_THRESHOLD']:,} cells")
    print("=" * 60, flush=True)

    bl = run_baseline(cfg)
    mx = run_maxdose(cfg)
    bl_fit = compute_fitness(bl, cfg)
    mx_fit = compute_fitness(mx, cfg)
    works = bl.final != mx.final
    print(f"  Baseline: fit={bl_fit:,.0f}  burden={bl.burden:,}  "
          f"final={bl.final:,}  peak={bl.peak:,}")
    print(f"  Maxdose:  fit={mx_fit:,.0f}  burden={mx.burden:,}  "
          f"final={mx.final:,}  dose={mx.cumulative_dose:,.0f}")
    print(f"  Drugs {'WORK' if works else 'NO EFFECT'}")

    # Auto-scale danger ceiling to actual tumor size
    actual_peak = bl.peak
    if actual_peak > 0 and actual_peak < cfg["GRID_CAPACITY"] * 0.5:
        old_danger = cfg["DANGER_CEIL"]
        cfg["DANGER_CEIL"] = int(min(0.50 * cfg["GRID_CAPACITY"], 2.0 * actual_peak))
        cfg["MIN_THRESHOLD"] = max(10, int(actual_peak * 0.20))
        print(f"  Auto-scaled to tumor size (peak={actual_peak:,}):")
        print(f"    Danger ceiling: {old_danger:,} -> {cfg['DANGER_CEIL']:,}")
        print(f"    Min threshold:  {cfg['MIN_THRESHOLD']:,}")

    # Recompute baseline with auto-scaled params
    bl_fit = compute_fitness(bl, cfg)

    print(f"  Baseline burden: {bl.burden:,}", flush=True)
    cfg["_baseline_burden"] = bl.burden
    return bl, bl_fit, works


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def _eval_one(args):
    i, genome_dict, cfg = args
    g = Genome(genome_dict["dosage"], genome_dict["cell_threshold"])
    path = os.path.join(cfg["ROUTINES_DIR"], f"routine_{i}.py")
    write_routine(g, path, cfg)
    traj = run_simulation(path, os.path.join(cfg["SAMPLES_DIR"], f"genome_{i}"), cfg)
    return i, compute_fitness(traj, cfg), traj


def evaluate(population, cfg):
    n = len(population)
    fitnesses    = [0.0] * n
    trajectories = [None] * n
    args = [(i, g.to_dict(), cfg) for i, g in enumerate(population)]

    if cfg["MAX_WORKERS"] > 1:
        with ProcessPoolExecutor(max_workers=cfg["MAX_WORKERS"]) as ex:
            futs = {ex.submit(_eval_one, a): a[0] for a in args}
            for fut in as_completed(futs):
                i, f, t = fut.result()
                fitnesses[i] = f
                trajectories[i] = t
                ns = len(t.cell_counts)
                print(f"  {i:02d}: fit={f:>10.0f}  "
                      f"final={t.final:>5,}  "
                      f"dose={t.cumulative_dose:>7,.0f}  "
                      f"burden={t.burden:>8,}  "
                      f"hol={t.holiday_steps}/{ns}  "
                      f"[d={population[i].dosage:.0f} t={population[i].cell_threshold}]",
                      flush=True)
    else:
        for a in args:
            i, f, t = _eval_one(a)
            fitnesses[i] = f
            trajectories[i] = t
            ns = len(t.cell_counts)
            print(f"  {i:02d}: fit={f:>10.0f}  "
                  f"final={t.final:>5,}  "
                  f"dose={t.cumulative_dose:>7,.0f}  "
                  f"burden={t.burden:>8,}  "
                  f"[d={population[i].dosage:.0f} t={population[i].cell_threshold}]",
                  flush=True)

    return fitnesses, trajectories


# ─────────────────────────────────────────────────────────────────────────────
# GENETIC OPERATORS
#
# 1-point crossover on 2-gene genome
# Modification-only mutation at 50% per gene
# ─────────────────────────────────────────────────────────────────────────────

def tournament_select(population, fitnesses, k):
    pairs = list(zip(population, fitnesses))
    return [copy.deepcopy(max(random.sample(pairs, min(k, len(pairs))),
                              key=lambda x: x[1])[0])
            for _ in range(len(population))]


def crossover_1point(p1, p2):
    """
    1-point crossover on a 2-gene genome.
    Cut point is between gene 0 and gene 1.
    
    Parent 1: [dosage_1, threshold_1]
    Parent 2: [dosage_2, threshold_2]
    
    Child 1:  [dosage_1, threshold_2]  (gene 0 from P1, gene 1 from P2)
    Child 2:  [dosage_2, threshold_1]  (gene 0 from P2, gene 1 from P1)
    """
    c1 = Genome(dosage=p1.dosage, cell_threshold=p2.cell_threshold)
    c2 = Genome(dosage=p2.dosage, cell_threshold=p1.cell_threshold)
    return c1, c2


def mutate(genome, cfg):
    """
    Modification-only mutation. Each gene mutates independently
    with 50% probability (expected 1 mutation per 2-gene individual).
    
    Gaussian perturbation:
      dosage:         N(0, DOSAGE_MAX * 0.1)
      cell_threshold: N(0, GRID_CAPACITY * 0.1)
    """
    mr = cfg["MUTATION_RATE_PER_GENE"]  # 0.50

    # Gene 0: dosage
    if random.random() < mr:
        scale = cfg["DOSAGE_MAX"] * 0.1
        genome.dosage = round(max(cfg["DOSAGE_MIN"], min(cfg["DOSAGE_MAX"],
            genome.dosage + random.gauss(0, scale))), 2)

    # Gene 1: cell_threshold
    if random.random() < mr:
        scale = max(1, cfg["GRID_CAPACITY"] // 10)
        genome.cell_threshold = max(cfg["MIN_THRESHOLD"], min(cfg["GRID_CAPACITY"],
            genome.cell_threshold + int(random.gauss(0, scale))))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN GA
# ─────────────────────────────────────────────────────────────────────────────

def genetic_algorithm(cfg):
    random.seed(cfg["RANDOM_SEED"])
    cap = cfg["GRID_CAPACITY"]
    t_start = time.time()

    print(f"Threshold-Only GA")
    print(f"Seed: {cfg['RANDOM_SEED']}")
    print(f"Grid: {cfg['GRID_N']}³ = {cap:,} points")
    print(f"Danger ceiling: {cfg['DANGER_CEIL']:,} (50%)")
    print(f"Warmup: {cfg['WARMUP_PERIOD']} steps")
    print(f"GA: pop={cfg['POP_SIZE']}  gen={cfg['GENERATIONS']}  "
          f"workers={cfg['MAX_WORKERS']}  ranks={cfg['N_RANKS']}")
    print(f"Genome: 2 genes [dosage, cell_threshold]")
    print(f"Crossover: 1-point  |  Mutation: 50% per gene (modification only)")
    print(f"Fitness = -{cfg['BURDEN_WEIGHT']}*B - {cfg['TOXICITY_WEIGHT']}*D "
          f"- {cfg['OVERFLOW_PENALTY']}*O - {cfg['TERMINAL_WEIGHT']}*F "
          f"+ {cfg['HOLIDAY_BONUS']}*H  (mechanism-agnostic)")
    print(flush=True)

    # Preflight
    if not cfg.get("SKIP_PREFLIGHT"):
        baseline_traj, baseline_fit, ok = preflight(cfg)
        if not ok:
            print("Drugs have no effect. Aborting.")
            return None, None
    else:
        baseline_traj = run_baseline(cfg)
        baseline_fit  = compute_fitness(baseline_traj, cfg)

    # Initialize population
    pop = [random_genome(cfg) for _ in range(cfg["POP_SIZE"])]
    fit, trajs = evaluate(pop, cfg)
    best_fit, best_genome, best_traj = float("-inf"), None, None
    history = []

    for gen in range(cfg["GENERATIONS"]):
        elapsed = time.time() - t_start
        print(f"\n{'='*60}")
        print(f"  Generation {gen+1} / {cfg['GENERATIONS']}  "
              f"[{elapsed/60:.1f} min elapsed]")
        print(f"{'='*60}", flush=True)

        # Track best
        gb = max(fit)
        ga = sum(fit) / len(fit)
        history.append((gb, ga))

        gi = fit.index(gb)
        if gb > best_fit:
            best_fit    = gb
            best_genome = copy.deepcopy(pop[gi])
            best_traj   = trajs[gi]

        ns = len(best_traj.cell_counts) if best_traj and best_traj.cell_counts else 0
        outcome, outcome_time = best_traj.classify_outcome(cfg) if best_traj else ("FATAL", 0)
        print(f"\n  >> Best: {gb:,.0f}  Avg: {ga:,.0f}  "
              f"vs baseline: {gb - baseline_fit:+,.0f}")
        print(f"     genome: dose={best_genome.dosage:.1f}  "
              f"thresh={best_genome.cell_threshold}")
        if best_traj and ns:
            print(f"     final={best_traj.final:,}  burden={best_traj.burden:,}  "
                  f"dose={best_traj.cumulative_dose:,.0f}  "
                  f"hol={best_traj.holiday_steps}/{ns}")
            ot = f"t={outcome_time}" if outcome_time else "end"
            print(f"     outcome={outcome} ({ot})", flush=True)

        # Elitism: preserve top 2
        top_idx = sorted(range(len(fit)), key=lambda k: fit[k], reverse=True)
        elites = [copy.deepcopy(pop[top_idx[0]]),
                  copy.deepcopy(pop[top_idx[1]])]

        # Selection
        parents = tournament_select(pop, fit, cfg["TOURNAMENT_SIZE"])

        # Build next generation
        new_pop = list(elites)  # elites pass through unmutated
        while len(new_pop) < cfg["POP_SIZE"]:
            p1, p2 = random.choice(parents), random.choice(parents)
            c1, c2 = crossover_1point(p1, p2)
            mutate(c1, cfg)
            mutate(c2, cfg)
            new_pop.append(c1)
            if len(new_pop) < cfg["POP_SIZE"]:
                new_pop.append(c2)

        pop = new_pop[:cfg["POP_SIZE"]]
        fit, trajs = evaluate(pop, cfg)

    # ── Final evaluation ─────────────────────────────────────────────────
    elapsed = time.time() - t_start

    # Re-evaluate best genome
    final_path = os.path.join(cfg["ROUTINES_DIR"], "best_routine_threshold.py")
    write_routine(best_genome, final_path, cfg)
    best_traj = run_simulation(final_path,
                                os.path.join(cfg["SAMPLES_DIR"], "best_final"), cfg)
    best_fit = compute_fitness(best_traj, cfg)
    ns = len(best_traj.cell_counts) if best_traj.cell_counts else 0
    outcome, outcome_time = best_traj.classify_outcome(cfg) if best_traj else ("FATAL", 0)

    danger_pct = int(100 * cfg["DANGER_CEIL"] / cap)
    print(f"\n{'='*60}")
    print(f"  BEST SCHEDULE (threshold-only)")
    print(f"  Grid: {cfg['GRID_N']}³ = {cap:,}  |  "
          f"Danger: {cfg['DANGER_CEIL']:,} ({danger_pct}%)")
    print(f"  Time: {elapsed/60:.1f} min")
    print(f"{'='*60}")
    print(f"  Fitness: {best_fit:,.0f}  (baseline: {baseline_fit:,.0f})")
    print(f"  Genome:  dose={best_genome.dosage:.1f}  "
          f"thresh={best_genome.cell_threshold}")
    if best_traj:
        bl_burden = cfg.get("_baseline_burden", 0)
        if bl_burden > 0:
            pct = ((best_traj.burden - bl_burden) / bl_burden) * 100
            print(f"  Burden:  {best_traj.burden:,} ({pct:+.1f}% vs untreated)")
        print(f"  Final:   {best_traj.final:,}  Peak: {best_traj.peak:,}")
        print(f"  Dose:    {best_traj.cumulative_dose:,.0f}")
        print(f"  Holidays: {best_traj.holiday_steps}/{ns}  "
              f"Overflow: {best_traj.overflow_steps}/{ns}")
        ot = f"t={outcome_time}" if outcome_time else "end"
        print(f"  Outcome: {outcome} ({ot})")
    print(f"\n  Saved: {final_path}", flush=True)

    # Plot
    if history:
        os.makedirs(cfg["OUTPUT_DIR"], exist_ok=True)
        bests, avgs = zip(*history)
        fig, axes = plt.subplots(2, 1, figsize=(12, 9))

        axes[0].axhline(y=baseline_fit, color='r', ls=':',
                        label=f"Baseline ({baseline_fit:,.0f})")
        axes[0].plot(range(1, len(bests)+1), bests, 'o-', label="Best")
        axes[0].plot(range(1, len(avgs)+1), avgs, 's--', label="Average")
        axes[0].set_ylabel("Fitness")
        axes[0].set_title(f"Threshold-Only GA — {cfg['GRID_N']}³ grid")
        axes[0].legend()

        if best_traj and best_traj.timesteps:
            ax2 = axes[1]
            ax2.plot(best_traj.timesteps, best_traj.cell_counts,
                     'g-', lw=2, label="Best schedule")
            if baseline_traj and baseline_traj.timesteps:
                ax2.plot(baseline_traj.timesteps, baseline_traj.cell_counts,
                         'r:', lw=1.5, label="Untreated")
            ax2.axhline(y=cfg["DANGER_CEIL"], color='orange', ls='--',
                        label=f"Danger ({cfg['DANGER_CEIL']:,})")
            # Markers
            if best_traj.dose_per_step:
                n_pts = min(len(best_traj.timesteps), len(best_traj.dose_per_step))
                on_t, on_c, off_t, off_c = [], [], [], []
                for j in range(n_pts):
                    if best_traj.dose_per_step[j] > 0:
                        on_t.append(best_traj.timesteps[j])
                        on_c.append(best_traj.cell_counts[j])
                    else:
                        off_t.append(best_traj.timesteps[j])
                        off_c.append(best_traj.cell_counts[j])
                if on_t:
                    ax2.scatter(on_t, on_c, c='blue', s=25, zorder=5,
                                label="Treating", marker='v')
                if off_t:
                    ax2.scatter(off_t, off_c, c='gray', s=25, zorder=5,
                                label="Holiday", marker='o')
            ax2.set_xlabel("Timestep")
            ax2.set_ylabel("Alive Tumor Cells")
            ax2.set_title("Tumor Trajectory: Best Schedule vs Untreated")
            ax2.legend()

        plt.tight_layout()
        plot_path = os.path.join(cfg["OUTPUT_DIR"], "fitness_plot_threshold.png")
        plt.savefig(plot_path, dpi=150)
        print(f"\nPlot: {plot_path}")

    return best_genome, best_fit


if __name__ == "__main__":
    cfg = get_config()
    genetic_algorithm(cfg)