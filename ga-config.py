"""
added to default.conf
--time_steps 50000
--warmup_period 5000
--sample_period 1000
"""

import random, copy, os, shutil, subprocess, re, sys, argparse, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION 
# ─────────────────────────────────────────────────────────────────────────────

def get_config():
    p = argparse.ArgumentParser(description="Adaptive Therapy GA — CancerSim2")
    p.add_argument("--grid-size", type=int, default=10,
                   help="Grid dimension N for NxNxN lattice (default=10)")
    p.add_argument("--containment-pct", type=float, default=0.50,
                   help="Containment ceiling as fraction of carrying capacity "
                        "(default=0.50, matching Zhang 50%% PSA threshold)")
    p.add_argument("--warmup-period", type=int, default=None,
                   help="Warmup steps before drug allowed (default: auto-scaled)")
    p.add_argument("--time-steps", type=int, default=50000)
    p.add_argument("--sample-period", type=int, default=1000)
    p.add_argument("--dosage-max", type=float, default=1000.0)
    p.add_argument("--pop-size", type=int, default=20)
    p.add_argument("--generations", type=int, default=25)
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--n-ranks", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-seeds", type=int, default=1,
                   help="Number of seeds to average per genome evaluation "
                        "(1=fast/local, 3+=reliable/HPC)")
    p.add_argument("--min-early-gen", type=int, default=3,
                   help="Earliest generation early termination can trigger")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint JSON to resume from")
    p.add_argument("--skip-preflight", action="store_true")
    p.add_argument("--cs2-binary", type=str, default=None)
    p.add_argument("--cs2-config", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="./ga_output")
    args = p.parse_args()

    grid_n    = args.grid_size
    capacity  = grid_n ** 3
    ceil_pct  = args.containment_pct
    ceil      = int(ceil_pct * capacity)

    # Auto-scale warmup: larger grids need more time for tumor to establish.
    # At 10³ the tumor fills in ~3000 steps. Scale proportionally.
    if args.warmup_period is not None:
        warmup = args.warmup_period
    elif grid_n <= 10:
        warmup = 5000
    elif grid_n <= 50:
        warmup = 10000
    else:
        warmup = 20000

    # Auto-scale UPC++ ranks to grid size if not specified
    if args.n_ranks is not None:
        n_ranks = args.n_ranks
    elif grid_n <= 10:
        n_ranks = 1
    elif grid_n <= 50:
        n_ranks = 4
    else:
        n_ranks = 16

    # Auto-scale simulation timeout
    if grid_n <= 10:
        timeout = 300
    elif grid_n <= 50:
        timeout = 900
    else:
        timeout = 3600

    # Minimum cell threshold for rules: must be meaningful fraction of capacity.
    # At 10³, min=50 (5%). At 50³, min=6250 (5%). Prevents trivial early-kill.
    min_threshold = max(10, int(0.20 * capacity))

    cfg = {
        # Grid
        "GRID_N":           grid_n,
        "GRID_CAPACITY":    capacity,
        "CONTAINMENT_PCT":  ceil_pct,
        "CONTAINMENT_CEIL": ceil,
        "MIN_THRESHOLD":    min_threshold,

        # Time
        "TIME_STEPS":       args.time_steps,
        "SAMPLE_PERIOD":    args.sample_period,
        "WARMUP_PERIOD":    warmup,
        "NUM_SAMPLES":      args.time_steps // args.sample_period,

        # Drug
        "DOSAGE_MIN":       0.0,
        "DOSAGE_MAX":       args.dosage_max,

        # GA
        "POP_SIZE":         args.pop_size,
        "GENERATIONS":      args.generations,
        "TOURNAMENT_SIZE":  3,
        "MR_INITIAL":       0.5,
        "MR_FINAL":         0.15,
        "STAGNATION_WIN":   4,
        "IMMIGRANT_COUNT":  5,
        "MAX_INSTRUCTIONS": 10,

        # Fitness — adaptive therapy preferred, any cure > no treatment
        #
        # ORDERING: Adaptive > Eradication > Carpet bomb > No treatment
        #
        # f(T,D) = α·M + κ·C + δ·H - γ·B - β·D - ω·O - τ·F
        #
        # POSITIVE (reward adaptive behavior):
        #   M = managed steps (0 < cells ≤ 50% capacity)
        #       Scaled to cap×1.5 → large enough to offset burden from alive tumor
        #   C = treatment cycles (on→off transitions)
        #       Scaled to cap×5 → THE differentiator: 3 cycles >> 1 cycle
        #   H = holiday steps (drug-free)
        #       Fixed at 50 → quality of life tiebreaker
        #
        # NEGATIVE (penalize bad outcomes):
        #   B = total burden (weight 0.1) → tiebreaker, NOT the driver
        #   D = cumulative dose (weight 0.0005) → VERY small so carpet bomb
        #       still beats no treatment (50M × 0.0005 = 25k < overflow penalty)
        #   O = overflow steps (cells > 50%) → scaled to cap×2, destroys no-treatment
        #   F = final cell count → residual disease penalty
        #
        # Verified ordering at 10³ and 20³:
        #   Adaptive (+62k/+371k) > Eradication (+11k/+33k) >
        #   Carpet bomb (-25k) > No treatment (-78k/-351k)
        "DANGER_CEIL":        int(0.50 * capacity),
        "MANAGED_BONUS":      max(1500, int(capacity * 1.5)),
        "CYCLING_BONUS":      max(5000, capacity * 5),
        "HOLIDAY_BONUS":      50,
        "BURDEN_WEIGHT":      0.1,
        "TOXICITY_WEIGHT":    0.0005,    # tiny: any cure > no treatment
        "OVERFLOW_PENALTY":   max(2000, capacity * 2),
        "TERMINAL_WEIGHT":    20,

        # ── Outcome classification (clinical parallel) ──
        # These define formal success/failure for each simulation trajectory.
        #
        # CURE: cells = 0 for CURE_WINDOW consecutive snapshots.
        #   Clinical: complete response sustained across follow-up visits.
        #   Default: 5 snapshots = 5,000 timesteps of zero cells.
        #
        # FATAL: cells > FATAL_CEIL for FATAL_WINDOW consecutive snapshots.
        #   Clinical: tumor burden incompatible with survival (organ failure).
        #   Default: 90% capacity for 3 snapshots = rapidly growing, uncontrolled.
        #
        # PROGRESSIVE: tumor growing on 3+ consecutive snapshots while drug
        #   is being applied. Resistant population has won; drug is futile.
        #   Clinical: RECIST progressive disease on therapy.
        #
        # MANAGED: none of the above. Tumor persists but is controlled.
        #   Clinical: stable disease under adaptive therapy.
        "CURE_WINDOW":        5,       # consecutive zero-cell snapshots
        "FATAL_CEIL":         int(0.90 * capacity),
        "FATAL_WINDOW":       3,       # consecutive snapshots above fatal
        "PROGRESSION_WINDOW": 3,       # consecutive growth-while-treating

        # ── GA-level termination (early stopping) ──
        # The GA stops early if the population has converged to a satisfactory
        # solution and further evolution is unlikely to improve outcomes.
        #
        # CONVERGED: best fitness unchanged for CONVERGENCE_WINDOW gens AND
        #   best genome's trajectory achieves cure or managed outcome.
        #
        # SOLVED: best genome achieves cure with cumulative dose below
        #   DOSE_EFFICIENCY_THRESHOLD * capacity. The cheapest possible cure.
        "CONVERGENCE_WINDOW": 6,       # gens without improvement → stop
        "DOSE_EFFICIENCY_THRESHOLD": 0.1,  # dose/capacity ratio for "solved"
        "MIN_EARLY_STOP_GEN": args.min_early_gen,  # no early stop before this gen

        # Multi-seed evaluation
        "EVAL_SEEDS":  args.eval_seeds,

        # Resume
        "RESUME_PATH": args.resume,

        # Fitness addition: terminal penalty for residual tumor
        "TERMINAL_WEIGHT":    10,      # penalty per final cell (residual disease)

        # Cycling rule params
        "REDUCTION_TARGET_RANGE": (0.2, 0.8),
        "REGROWTH_TRIGGER_RANGE": (0.5, 1.0),

        # Infra
        "MAX_WORKERS":  args.max_workers,
        "N_RANKS":      n_ranks,
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
# GENOME 
# ─────────────────────────────────────────────────────────────────────────────

class CyclingRule:
    """
    Stateful: tracks treating vs holiday. Treats until tumor shrinks to
    reduction_target * cycle_peak, pauses until regrowth_trigger * cycle_peak.
    """
    def __init__(self, dosage, reduction_target, regrowth_trigger, start_threshold):
        self.dosage           = dosage
        self.reduction_target = reduction_target
        self.regrowth_trigger = regrowth_trigger
        self.start_threshold  = start_threshold
        self.rule_type        = "cycling"

    def to_dict(self):
        return dict(type="cycling", dosage=self.dosage,
                    reduction_target=self.reduction_target,
                    regrowth_trigger=self.regrowth_trigger,
                    start_threshold=self.start_threshold)


class ThresholdRule:
    """
    Applies dosage when cells >= cell_threshold. Multiple threshold
    rules at different levels create a dose-escalation ladder.
    """
    def __init__(self, dosage, cell_threshold):
        self.dosage         = dosage
        self.cell_threshold = cell_threshold
        self.rule_type      = "threshold"

    def to_dict(self):
        return dict(type="threshold", dosage=self.dosage,
                    cell_threshold=self.cell_threshold)


class Genome:
    def __init__(self):
        self.rules = []
    def total_dosage(self):
        return sum(r.dosage for r in self.rules)


def random_cycling_rule(cfg, hints=None):
    h = hints or {}
    d_lo = h.get("dose_min", cfg["DOSAGE_MIN"])
    d_hi = h.get("dose_max", cfg["DOSAGE_MAX"])
    lo, hi = cfg["REDUCTION_TARGET_RANGE"]
    rlo, rhi = cfg["REGROWTH_TRIGGER_RANGE"]
    return CyclingRule(
        dosage           = round(random.uniform(d_lo, d_hi), 2),
        reduction_target = round(random.uniform(lo, hi), 2),
        regrowth_trigger = round(random.uniform(rlo, rhi), 2),
        start_threshold  = random.randint(cfg["MIN_THRESHOLD"],
                                          max(cfg["MIN_THRESHOLD"],
                                              cfg["CONTAINMENT_CEIL"])),
    )


def random_threshold_rule(cfg, hints=None):
    h = hints or {}
    d_lo = h.get("dose_min", cfg["DOSAGE_MIN"])
    d_hi = h.get("dose_max", cfg["DOSAGE_MAX"])
    ct   = max(h.get("ct_max", cfg["GRID_CAPACITY"]), cfg["MIN_THRESHOLD"])
    return ThresholdRule(
        dosage         = round(random.uniform(d_lo, d_hi), 2),
        cell_threshold = random.randint(cfg["MIN_THRESHOLD"], ct),
    )


def random_rule(cfg, hints=None):
    if random.random() < 0.6:
        return random_cycling_rule(cfg, hints)
    return random_threshold_rule(cfg, hints)


def random_genome(cfg, hints=None):
    g = Genome()
    g.rules.append(random_cycling_rule(cfg, hints))
    for _ in range(random.randint(1, 5)):
        g.rules.append(random_rule(cfg, hints))
    return g


def prune_genome(genome):
    cycling = [r for r in genome.rules if r.rule_type == "cycling"]
    threshold = [r for r in genome.rules if r.rule_type == "threshold"]
    pruned = []
    for i, r in enumerate(threshold):
        if not any(threshold[j].dosage >= r.dosage
                   and threshold[j].cell_threshold <= r.cell_threshold
                   and (threshold[j].dosage > r.dosage
                        or threshold[j].cell_threshold < r.cell_threshold)
                   for j in range(len(threshold)) if j != i):
            pruned.append(r)
    genome.rules = cycling + (pruned if pruned else threshold[:1])
    if not genome.rules:
        genome.rules = [cycling[0]] if cycling else [threshold[0]]
    return genome


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
        self._ceil = cfg["CONTAINMENT_CEIL"] if cfg else 500
        self._danger = cfg["DANGER_CEIL"] if cfg else 800

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
    def eradicated(self):
        return (len(self.cell_counts) >= 2
                and self.cell_counts[-1] == 0
                and self.cell_counts[-2] == 0)

    @property
    def managed_steps(self):
        """Timesteps where 0 < cells <= danger ceiling (contained but alive)."""
        return sum(1 for c in self.cell_counts if 0 < c <= self._danger)

    @property
    def eradicated_steps(self):
        """Timesteps where cells = 0 (sensitive population gone)."""
        return sum(1 for c in self.cell_counts if c == 0)

    @property
    def contained_steps(self):
        return sum(1 for c in self.cell_counts if c <= self._ceil)

    @property
    def overflow_steps(self):
        """Timesteps where tumor exceeds danger ceiling (uncontrolled growth)."""
        return sum(1 for c in self.cell_counts if c > self._danger)

    @property
    def holiday_steps(self):
        if not self.dose_per_step:
            return 0
        return sum(1 for d in self.dose_per_step if d == 0.0)

    @property
    def regrowth_steps(self):
        count, saw_decline = 0, False
        for i in range(1, len(self.cell_counts)):
            if self.cell_counts[i] < self.cell_counts[i-1]:
                saw_decline = True
            elif saw_decline and self.cell_counts[i] > self.cell_counts[i-1]:
                count += 1
        return count

    @property
    def num_cycles(self):
        if not self.dose_per_step or len(self.dose_per_step) < 2:
            return 0
        return sum(1 for i in range(1, len(self.dose_per_step))
                   if self.dose_per_step[i-1] > 0 and self.dose_per_step[i] == 0)

    def first_control_time(self):
        if not self.cell_counts or self.cell_counts[0] <= self._ceil:
            return 0
        for i, c in enumerate(self.cell_counts):
            if c <= self._ceil:
                return self.timesteps[i]
        return None

    def classify_outcome(self, cfg):
        """
        Classify the simulation trajectory into a clinical outcome category.

        Returns one of:
          "CURE"        — complete response: cells = 0 for CURE_WINDOW snapshots
          "FATAL"       — patient failure: cells > 90% capacity sustained
          "PROGRESSIVE" — treatment failure: growing despite active drug
          "MANAGED"     — stable disease: tumor persists, controlled

        Also returns the timestep at which the outcome was determined, or None
        if the outcome held through the end of simulation.
        """
        cc = self.cell_counts
        ds = self.dose_per_step
        n = len(cc)
        if n == 0:
            return "FATAL", 0

        cure_w = cfg.get("CURE_WINDOW", 5)
        fatal_ceil = cfg.get("FATAL_CEIL", int(0.90 * cfg["GRID_CAPACITY"]))
        fatal_w = cfg.get("FATAL_WINDOW", 3)
        prog_w = cfg.get("PROGRESSION_WINDOW", 3)

        # ── Check for CURE ──
        # Scan for CURE_WINDOW consecutive snapshots with cells = 0
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

        # ── Check for FATAL ──
        # Scan for FATAL_WINDOW consecutive snapshots with cells > fatal ceiling
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

        # ── Check for PROGRESSIVE DISEASE ──
        # Scan for PROGRESSION_WINDOW consecutive snapshots where:
        #   (a) cell count is increasing, AND
        #   (b) drug is being applied (dose > 0)
        progressive_time = None
        if ds and len(ds) >= n:
            consecutive_prog = 0
            for i in range(1, n):
                growing = cc[i] > cc[i-1]
                treating = ds[i] > 0 if i < len(ds) else False
                if growing and treating:
                    consecutive_prog += 1
                    if consecutive_prog >= prog_w:
                        progressive_time = self.timesteps[i] if self.timesteps else i
                        break
                else:
                    consecutive_prog = 0

        # ── Priority: FATAL > PROGRESSIVE > CURE > MANAGED ──
        # If fatal happened before cure, the patient died first.
        if fatal_time is not None:
            if cure_time is None or fatal_time <= cure_time:
                return "FATAL", fatal_time

        if progressive_time is not None:
            if cure_time is None or progressive_time < cure_time:
                return "PROGRESSIVE", progressive_time

        if cure_time is not None:
            return "CURE", cure_time

        return "MANAGED", None

    def time_to_progression(self, cfg):
        """
        Defined as the first timestep where cell count exceeds the pre-treatment
        peak while drug is being applied. Returns None if no progression.
        """
        cc = self.cell_counts
        ds = self.dose_per_step
        if not cc or len(cc) < 3:
            return None

        # Find peak during warmup/early growth
        warmup_end = cfg.get("WARMUP_PERIOD", 5000)
        sp = cfg.get("SAMPLE_PERIOD", 1000)
        warmup_samples = max(1, warmup_end // sp)
        pre_peak = max(cc[:min(warmup_samples + 2, len(cc))])

        # Find first post-treatment snapshot exceeding pre-treatment peak
        for i in range(warmup_samples, len(cc)):
            treating = ds[i] > 0 if ds and i < len(ds) else False
            if cc[i] > pre_peak and treating:
                return self.timesteps[i] if self.timesteps else i
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SEARCH HINTS
# ─────────────────────────────────────────────────────────────────────────────

class SearchHints:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dose_min = cfg["DOSAGE_MIN"]
        self.dose_max = cfg["DOSAGE_MAX"]
        self.ct_max   = cfg["GRID_CAPACITY"]
        self.log      = []

    def as_dict(self):
        return {"dose_min": self.dose_min, "dose_max": self.dose_max,
                "ct_max": self.ct_max}

    def update(self, trajectories, genomes, fitnesses, gen):
        if not trajectories:
            return
        ranked = sorted(zip(fitnesses, trajectories, genomes),
                        key=lambda x: x[0], reverse=True)
        top = ranked[:min(5, len(ranked))]
        _, top_t, top_g = zip(*top)

        doses = [r.dosage for g in top_g for r in g.rules if r.dosage > 0]
        if doses:
            med = sorted(doses)[len(doses) // 2]
            self.dose_min = max(self.cfg["DOSAGE_MIN"], med * 0.3)
            self.dose_max = min(self.cfg["DOSAGE_MAX"], med * 2.5)
            if self.dose_max - self.dose_min < 20:
                self.dose_max = min(self.cfg["DOSAGE_MAX"], self.dose_min + 50)

        peaks = [t.peak for t in top_t if t.peak > 0]
        if peaks:
            self.ct_max = min(self.cfg["GRID_CAPACITY"],
                              max(self.cfg["MIN_THRESHOLD"], max(peaks)))

        n = len(top_t)
        entry = dict(
            gen=gen+1,
            dose=f"[{self.dose_min:.0f}-{self.dose_max:.0f}]",
            mgd=f"{sum(t.managed_steps for t in top_t)/n:.1f}",
            cyc=f"{sum(t.num_cycles for t in top_t)/n:.1f}",
            hol=f"{sum(t.holiday_steps for t in top_t)/n:.1f}",
            ovf=f"{sum(t.overflow_steps for t in top_t)/n:.1f}",
            cdose=f"{sum(t.cumulative_dose for t in top_t)/n:.0f}",
            fin=f"{sum(t.final for t in top_t)/n:.0f}",
        )
        self.log.append(entry)
        print(f"  [adapt] dose={entry['dose']}  mgd={entry['mgd']}  "
              f"cyc={entry['cyc']}  hol={entry['hol']}  "
              f"ovf={entry['ovf']}  dose={entry['cdose']}")


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
# DRUG ROUTINE GENERATION — stateful Zhang-style cycling
# ─────────────────────────────────────────────────────────────────────────────

def write_routine(genome, filepath, cfg):
    cycling = [r.to_dict() for r in genome.rules if r.rule_type == "cycling"]
    threshold = [r.to_dict() for r in genome.rules if r.rule_type == "threshold"]
    warmup = cfg["WARMUP_PERIOD"]
    sample_period = cfg["SAMPLE_PERIOD"]

    script = f"""import sys, os

CYCLING_RULES = {cycling!r}
THRESHOLD_RULES = {threshold!r}
WARMUP = {warmup}
SAMPLE_PERIOD = {sample_period}

_treating = [False] * len(CYCLING_RULES)
_cycle_peak = [0] * len(CYCLING_RULES)
_cumulative_dose = 0.0
_dose_log = []
_interval_max_dose = 0.0    # track max dose within each sample interval
_interval_total_dose = 0.0  # track total dose within each sample interval

def drug_func(timestep, drug_idx, num_cells, num_resistant):
    global _cumulative_dose, _interval_max_dose, _interval_total_dose
    if isinstance(num_resistant, (list, tuple)):
        num_resistant = sum(num_resistant)
    num_cells = int(num_cells or 0)

    # At each sample boundary, log the interval's max dose then reset
    if timestep % SAMPLE_PERIOD == 0 and timestep > 0:
        _dose_log.append(_interval_max_dose)
        _interval_max_dose = 0.0
        _interval_total_dose = 0.0

    # No drug during warmup — tumor must establish first
    if timestep < WARMUP:
        if timestep == 0:
            _dose_log.append(0.0)  # first sample point
        return 0.0

    cycling_dose = 0.0
    for i, rule in enumerate(CYCLING_RULES):
        if not _treating[i]:
            # Holiday: start treating when cells reach start_threshold
            # and have regrown to regrowth_trigger of last cycle peak
            if num_cells >= rule["start_threshold"]:
                if _cycle_peak[i] == 0 or num_cells >= rule["regrowth_trigger"] * _cycle_peak[i]:
                    _treating[i] = True
                    _cycle_peak[i] = num_cells
        else:
            # Treating: track peak, stop when tumor drops to reduction_target
            if num_cells > _cycle_peak[i]:
                _cycle_peak[i] = num_cells
            if _cycle_peak[i] > 0 and num_cells <= rule["reduction_target"] * _cycle_peak[i]:
                _treating[i] = False
            else:
                candidate = rule["dosage"]
                # Resistance scaling (Gatenby 2009)
                if num_cells > 0 and num_resistant > 0:
                    res_frac = num_resistant / num_cells
                    candidate *= max(0.3, 1.0 - 0.7 * res_frac)
                cycling_dose = max(cycling_dose, candidate)

    threshold_dose = 0.0
    for rule in THRESHOLD_RULES:
        if num_cells >= rule["cell_threshold"]:
            candidate = rule["dosage"]
            if num_cells > 0 and num_resistant > 0:
                res_frac = num_resistant / num_cells
                candidate *= max(0.3, 1.0 - 0.7 * res_frac)
            threshold_dose = max(threshold_dose, candidate)

    dosage = max(cycling_dose, threshold_dose)
    _cumulative_dose += dosage
    _interval_max_dose = max(_interval_max_dose, dosage)
    _interval_total_dose += dosage
    return float(dosage)

def get_drug_type(drug_idx):
    return 0

def finalize():
    # Flush the last interval
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
# FITNESS — adaptive therapy preferred, any cure beats no treatment
#
# f(T,D) = α·M + κ·C + δ·H - γ·B - β·D - ω·O - τ·F
#
# Ordering: Adaptive > Eradication > Carpet bomb > No treatment
# Key: dose weight (0.0005) is tiny so even 50M dose carpet bomb
# beats untreated tumor's overflow + burden + terminal penalties.
# ─────────────────────────────────────────────────────────────────────────────

def compute_fitness(traj, cfg):
    if not traj.cell_counts:
        return -999999.0
    return (
        cfg["MANAGED_BONUS"]    * traj.managed_steps
        + cfg["CYCLING_BONUS"]  * traj.num_cycles
        + cfg["HOLIDAY_BONUS"]  * traj.holiday_steps
        - cfg["BURDEN_WEIGHT"]  * traj.burden
        - cfg["TOXICITY_WEIGHT"] * traj.cumulative_dose
        - cfg["OVERFLOW_PENALTY"] * traj.overflow_steps
        - cfg["TERMINAL_WEIGHT"] * traj.final
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


def _auto_generate_cs2_config(cfg):
    """Generate a CS2 config file matching --grid-size if none exists."""
    grid_n = cfg["GRID_N"]
    config_path = cfg["CONFIG_PATH"]

    # Check if config already has matching dims
    try:
        with open(config_path) as f:
            conf_text = f.read()
        for line in conf_text.splitlines():
            parts = line.strip().split()
            if len(parts) == 2 and parts[0] == "--dim_x":
                if int(parts[1]) == grid_n:
                    return  # already matches
    except IOError:
        pass

    # Generate matching config
    auto_path = os.path.join(os.path.dirname(config_path),
                             f"grid{grid_n}_auto.conf")
    if os.path.exists(auto_path):
        cfg["CONFIG_PATH"] = auto_path
        print(f"  Using auto-generated config: {auto_path}")
        return

    # Read base config for non-grid params
    base_params = {}
    try:
        with open(config_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2 and parts[0].startswith("--"):
                    key = parts[0]
                    if key not in ("--dim_x", "--dim_y", "--dim_z",
                                   "--rank_n_x", "--rank_n_y", "--rank_n_z",
                                   "--samples_dir", "--phenos_dir"):
                        base_params[key] = parts[1]
    except IOError:
        pass

    # Set warmup/timesteps from cfg (may have been auto-scaled)
    base_params["--warmup_period"] = str(cfg["WARMUP_PERIOD"])
    base_params["--time_steps"] = str(cfg["TIME_STEPS"])
    base_params["--sample_period"] = str(cfg["SAMPLE_PERIOD"])

    with open(auto_path, "w") as f:
        f.write(f"--dim_x {grid_n}\n--dim_y {grid_n}\n--dim_z {grid_n}\n")
        f.write("--rank_n_x 1\n--rank_n_y 1\n--rank_n_z 1\n")
        for k, v in base_params.items():
            f.write(f"{k} {v}\n")

    cfg["CONFIG_PATH"] = auto_path
    print(f"  Auto-generated CS2 config: {auto_path}")


def preflight(cfg):
    cap = cfg["GRID_CAPACITY"]
    ceil = cfg["CONTAINMENT_CEIL"]
    pct = cfg["CONTAINMENT_PCT"]
    print("=" * 60)
    print(f"PREFLIGHT — Grid: {cfg['GRID_N']}³ = {cap:,} points")
    print(f"Containment ceiling: {ceil:,} cells ({pct*100:.0f}% of capacity)")
    print(f"Warmup: {cfg['WARMUP_PERIOD']} steps")
    print(f"Min rule threshold: {cfg['MIN_THRESHOLD']:,} cells")
    print("=" * 60)

    # Auto-generate CS2 config if grid size doesn't match
    _auto_generate_cs2_config(cfg)

    bl = run_baseline(cfg)
    mx = run_maxdose(cfg)
    bl_fit = compute_fitness(bl, cfg)
    mx_fit = compute_fitness(mx, cfg)
    works = bl.final != mx.final
    print(f"  Baseline: fit={bl_fit:,.0f}  burden={bl.burden:,}  "
          f"final={bl.final:,}  peak={bl.peak:,}  "
          f"overflow={bl.overflow_steps}/{len(bl.cell_counts)}")
    print(f"  Maxdose:  fit={mx_fit:,.0f}  burden={mx.burden:,}  "
          f"final={mx.final:,}  dose={mx.cumulative_dose:,.0f}")
    print(f"  Drugs {'WORK' if works else 'NO EFFECT'}")

    # Auto-detect warmup from baseline: find when tumor reaches 80% of peak
    if bl.cell_counts and bl.peak > 0:
        threshold_80 = int(0.80 * bl.peak)
        auto_warmup = cfg["WARMUP_PERIOD"]
        for i, c in enumerate(bl.cell_counts):
            if c >= threshold_80:
                auto_warmup = bl.timesteps[i] if bl.timesteps else i * cfg["SAMPLE_PERIOD"]
                break
        if auto_warmup != cfg["WARMUP_PERIOD"]:
            old_w = cfg["WARMUP_PERIOD"]
            # Round up to nearest sample_period
            auto_warmup = ((auto_warmup // cfg["SAMPLE_PERIOD"]) + 1) * cfg["SAMPLE_PERIOD"]
            cfg["WARMUP_PERIOD"] = auto_warmup
            print(f"  Warmup auto-adjusted: {old_w} → {auto_warmup} "
                  f"(baseline reached 80% peak at ~t={auto_warmup})")

    print(f"  Baseline burden: {bl.burden:,} (GA must beat this)\n")
    cfg["_baseline_burden"] = bl.burden
    return bl, bl_fit, works


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def _eval_one(args):
    i, genome_rules, cfg = args
    # Reconstruct genome from serialized rules
    g = Genome()
    for rd in genome_rules:
        if rd["type"] == "cycling":
            g.rules.append(CyclingRule(rd["dosage"], rd["reduction_target"],
                                       rd["regrowth_trigger"], rd["start_threshold"]))
        else:
            g.rules.append(ThresholdRule(rd["dosage"], rd["cell_threshold"]))
    path = os.path.join(cfg["ROUTINES_DIR"], f"routine_{i}.py")
    write_routine(g, path, cfg)

    n_seeds = cfg.get("EVAL_SEEDS", 1)
    if n_seeds <= 1:
        traj = run_simulation(path, os.path.join(cfg["SAMPLES_DIR"], f"genome_{i}"), cfg)
        return i, compute_fitness(traj, cfg), traj

    # Multi-seed: run N times, average fitness, return median trajectory
    fits, trajs = [], []
    for s in range(n_seeds):
        t = run_simulation(path, os.path.join(cfg["SAMPLES_DIR"], f"genome_{i}_s{s}"), cfg)
        fits.append(compute_fitness(t, cfg))
        trajs.append(t)
    avg_fit = sum(fits) / len(fits)
    # Return trajectory closest to mean fitness (median-ish)
    median_idx = min(range(len(fits)), key=lambda k: abs(fits[k] - avg_fit))
    return i, avg_fit, trajs[median_idx]


def evaluate(population, cfg):
    n = len(population)
    fitnesses    = [0.0]  * n
    trajectories = [None] * n
    ceil = cfg["CONTAINMENT_CEIL"]

    # Serialize genomes for multiprocessing
    args = [(i, [r.to_dict() for r in g.rules], cfg)
            for i, g in enumerate(population)]

    if cfg["MAX_WORKERS"] > 1:
        with ProcessPoolExecutor(max_workers=cfg["MAX_WORKERS"]) as ex:
            futs = {ex.submit(_eval_one, a): a[0] for a in args}
            for fut in as_completed(futs):
                i, f, t = fut.result()
                fitnesses[i] = f
                trajectories[i] = t
                ns = len(t.cell_counts)
                tag = "ok" if t.managed_steps > 0 or t.num_cycles > 0 else "--"
                print(f"  {i:02d}: fit={f:>10.0f}  "
                      f"final={t.final:>5,}  mgd={t.managed_steps:>2}/{ns}  "
                      f"cyc={t.num_cycles}  hol={t.holiday_steps:>2}/{ns}  "
                      f"ovf={t.overflow_steps:>2}/{ns}  "
                      f"dose={t.cumulative_dose:>7,.0f}  {tag}")
    else:
        for a in args:
            i, f, t = _eval_one(a)
            fitnesses[i] = f
            trajectories[i] = t
            print(f"  {i:02d}: fit={f:.0f}")

    return fitnesses, trajectories


# ─────────────────────────────────────────────────────────────────────────────
# GENETIC OPERATORS
# ─────────────────────────────────────────────────────────────────────────────

def tournament_select(population, fitnesses, k):
    pairs = list(zip(population, fitnesses))
    return [copy.deepcopy(max(random.sample(pairs, min(k, len(pairs))),
                              key=lambda x: x[1])[0])
            for _ in range(len(population))]


def crossover(p1, p2, max_inst):
    r1, r2 = copy.deepcopy(p1.rules), copy.deepcopy(p2.rules)
    m = min(len(r1), len(r2))
    if m < 2:
        return copy.deepcopy(p1), copy.deepcopy(p2)
    a, b = sorted(random.sample(range(m), 2))
    c1, c2 = Genome(), Genome()
    c1.rules = (r1[:a] + r2[a:b] + r1[b:])[:max_inst]
    c2.rules = (r2[:a] + r1[a:b] + r2[b:])[:max_inst]
    return c1, c2


def mutate(genome, strength, cfg, hints=None):
    h = hints or {}
    d_lo = h.get("dose_min", cfg["DOSAGE_MIN"])
    d_hi = h.get("dose_max", cfg["DOSAGE_MAX"])
    mint = cfg["MIN_THRESHOLD"]
    ct   = max(h.get("ct_max", cfg["GRID_CAPACITY"]), mint)
    rtr  = cfg["REDUCTION_TARGET_RANGE"]
    rgr  = cfg["REGROWTH_TRIGGER_RANGE"]

    op = random.choice(["add", "remove", "modify"])
    if op == "add" and len(genome.rules) < cfg["MAX_INSTRUCTIONS"]:
        genome.rules.append(random_rule(cfg, hints))
    elif op == "remove" and len(genome.rules) > 1:
        genome.rules.pop(random.randint(0, len(genome.rules) - 1))
    elif op == "modify" and genome.rules:
        r = genome.rules[random.randint(0, len(genome.rules) - 1)]
        if r.rule_type == "cycling":
            field = random.choice(["dosage", "reduction_target",
                                    "regrowth_trigger", "start_threshold"])
            if field == "dosage":
                scale = max(1.0, (d_hi - d_lo) * 0.1 * strength)
                r.dosage = round(max(d_lo, min(d_hi,
                    r.dosage + random.gauss(0, scale))), 2)
            elif field == "reduction_target":
                r.reduction_target = round(max(rtr[0], min(rtr[1],
                    r.reduction_target + random.gauss(0, 0.1 * strength))), 2)
            elif field == "regrowth_trigger":
                r.regrowth_trigger = round(max(rgr[0], min(rgr[1],
                    r.regrowth_trigger + random.gauss(0, 0.1 * strength))), 2)
            elif field == "start_threshold":
                r.start_threshold = max(mint, min(cfg["GRID_CAPACITY"],
                    r.start_threshold + int(random.gauss(0,
                        cfg["CONTAINMENT_CEIL"] * 0.1 * strength))))
        else:
            field = random.choice(["dosage", "cell_threshold"])
            if field == "dosage":
                scale = max(1.0, (d_hi - d_lo) * 0.1 * strength)
                r.dosage = round(max(d_lo, min(d_hi,
                    r.dosage + random.gauss(0, scale))), 2)
            elif field == "cell_threshold":
                r.cell_threshold = max(mint, min(ct,
                    r.cell_threshold + int(random.gauss(0,
                        max(1, ct // 10) * strength))))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN GA
# ─────────────────────────────────────────────────────────────────────────────

def genetic_algorithm(cfg):
    random.seed(cfg["RANDOM_SEED"])
    cap  = cfg["GRID_CAPACITY"]
    ceil = cfg["CONTAINMENT_CEIL"]
    pct  = cfg["CONTAINMENT_PCT"]

    print(f"Seed: {cfg['RANDOM_SEED']}")
    print(f"Grid: {cfg['GRID_N']}³ = {cap:,} points")
    print(f"Containment: {ceil:,} cells ({pct*100:.0f}% capacity)")
    print(f"Warmup: {cfg['WARMUP_PERIOD']} steps | Min threshold: {cfg['MIN_THRESHOLD']:,}")
    print(f"GA: pop={cfg['POP_SIZE']}  gen={cfg['GENERATIONS']}  "
          f"workers={cfg['MAX_WORKERS']}  ranks={cfg['N_RANKS']}")
    print(f"Fitness = {cfg['MANAGED_BONUS']:,}*managed + {cfg['CYCLING_BONUS']:,}*cycles "
          f"+ {cfg['HOLIDAY_BONUS']}*holidays")
    print(f"         - {cfg['BURDEN_WEIGHT']}*burden - {cfg['TOXICITY_WEIGHT']}*dose "
          f"- {cfg['OVERFLOW_PENALTY']:,}*overflow(>{cfg['DANGER_CEIL']:,}) "
          f"- {cfg['TERMINAL_WEIGHT']}*final")
    print(f"Termination: SOLVED(dose/cap<{cfg['DOSE_EFFICIENCY_THRESHOLD']}) "
          f"or CONVERGED({cfg['CONVERGENCE_WINDOW']}gen) "
          f"after gen {cfg['MIN_EARLY_STOP_GEN']}")
    if cfg["EVAL_SEEDS"] > 1:
        print(f"Multi-seed eval: {cfg['EVAL_SEEDS']} seeds per genome")
    print()

    # Validate CS2 config matches --grid-size
    try:
        with open(cfg["CONFIG_PATH"]) as f:
            conf_text = f.read()
        cs2_dims = {}
        for line in conf_text.splitlines():
            parts = line.strip().split()
            if len(parts) == 2 and parts[0] in ("--dim_x", "--dim_y", "--dim_z"):
                cs2_dims[parts[0]] = int(parts[1])
        if cs2_dims:
            cs2_grid = (cs2_dims.get("--dim_x", 10),
                        cs2_dims.get("--dim_y", 10),
                        cs2_dims.get("--dim_z", 10))
            cs2_cap = cs2_grid[0] * cs2_grid[1] * cs2_grid[2]
            if cs2_cap != cap:
                print(f"  WARNING: --grid-size {cfg['GRID_N']} implies {cap:,} points, "
                      f"but CS2 config {cfg['CONFIG_PATH']} has "
                      f"dim={cs2_grid[0]}x{cs2_grid[1]}x{cs2_grid[2]} = {cs2_cap:,} points!")
                print(f"  The GA's thresholds/ceilings won't match CS2's actual grid.")
                print(f"  Create a matching config with --dim_x {cfg['GRID_N']} "
                      f"--dim_y {cfg['GRID_N']} --dim_z {cfg['GRID_N']}\n")
        elif cfg["GRID_N"] != 10:
            print(f"  NOTE: CS2 config has no dim_x/y/z — CS2 likely defaults to 10x10x10.")
            print(f"  Your --grid-size {cfg['GRID_N']} may not match. "
                  f"Add --dim_x/y/z to the config.\n")
    except (IOError, ValueError):
        pass

    # Preflight
    if not cfg.get("SKIP_PREFLIGHT"):
        baseline_traj, baseline_fit, ok = preflight(cfg)
        if not ok:
            print("Drugs have no effect. Aborting.")
            return None, None
    else:
        baseline_traj = run_baseline(cfg)
        baseline_fit  = compute_fitness(baseline_traj, cfg)

    hints = SearchHints(cfg)
    start_gen = 0

    # ── Resume from checkpoint ───────────────────────────────────────────
    if cfg.get("RESUME_PATH") and os.path.exists(cfg["RESUME_PATH"]):
        try:
            with open(cfg["RESUME_PATH"]) as jf:
                ckpt = json.load(jf)
            start_gen = ckpt["generation"]
            best_fit  = ckpt["best_fit"]
            history   = ckpt.get("history", [])
            recent    = ckpt.get("recent", [])
            # Restore best genome
            best_genome = Genome()
            for rd in ckpt["best_genome"]:
                if rd["type"] == "cycling":
                    best_genome.rules.append(CyclingRule(
                        rd["dosage"], rd["reduction_target"],
                        rd["regrowth_trigger"], rd["start_threshold"]))
                else:
                    best_genome.rules.append(ThresholdRule(
                        rd["dosage"], rd["cell_threshold"]))
            # Restore hints
            h = ckpt.get("hints", {})
            hints.dose_min = h.get("dose_min", hints.dose_min)
            hints.dose_max = h.get("dose_max", hints.dose_max)
            hints.ct_max   = h.get("ct_max", hints.ct_max)
            print(f"  Resumed from checkpoint: gen {start_gen}, "
                  f"best_fit={best_fit:,.0f}")
        except (json.JSONDecodeError, KeyError, IOError) as e:
            print(f"  Warning: failed to load checkpoint: {e}")
            start_gen = 0

    pop = [random_genome(cfg) for _ in range(cfg["POP_SIZE"])]
    fit, trajs = evaluate(pop, cfg)
    if start_gen == 0:
        best_fit, best_genome, best_traj = float("-inf"), None, None
        history, recent = [], []
    else:
        # Re-evaluate best genome from checkpoint
        best_traj = None
        gi = fit.index(max(fit))
        if fit[gi] > best_fit:
            best_fit = fit[gi]
            best_genome = copy.deepcopy(pop[gi])
            best_traj = trajs[gi]

    for gen in range(start_gen, cfg["GENERATIONS"]):
        print(f"\n{'='*60}")
        print(f"  Generation {gen+1} / {cfg['GENERATIONS']}")
        print(f"{'='*60}")

        total = cfg["GENERATIONS"]
        mr = cfg["MR_INITIAL"] + (cfg["MR_FINAL"] - cfg["MR_INITIAL"]) * gen / max(1, total-1)
        ms = 1.0 - (gen / max(1, total-1)) * 0.7

        hints.update(trajs, pop, fit, gen)
        hd = hints.as_dict()

        # Stagnation
        sw = cfg["STAGNATION_WIN"]
        if len(recent) >= sw:
            if all(abs(recent[-1] - recent[-j]) < 1.0 for j in range(1, sw)):
                ic = cfg["IMMIGRANT_COUNT"]
                print(f"  [stagnation] injecting {ic} explorers")
                worst = sorted(range(len(fit)), key=lambda k: fit[k])
                for j in range(min(ic, len(worst))):
                    pop[worst[j]] = random_genome(cfg, hd if j % 2 == 0 else None)

        # Elitism
        top2 = sorted(range(len(fit)), key=lambda k: fit[k], reverse=True)[:2]
        elites = [copy.deepcopy(pop[top2[0]]), copy.deepcopy(pop[top2[1]])]

        parents = tournament_select(pop, fit, cfg["TOURNAMENT_SIZE"])
        new_pop = list(elites)
        while len(new_pop) < cfg["POP_SIZE"]:
            p1, p2 = random.choice(parents), random.choice(parents)
            c1, c2 = crossover(p1, p2, cfg["MAX_INSTRUCTIONS"])
            if random.random() < mr:
                mutate(c1, ms, cfg, hd)
            if random.random() < mr:
                mutate(c2, ms, cfg, hd)
            new_pop.append(c1)
            if len(new_pop) < cfg["POP_SIZE"]:
                new_pop.append(c2)

        pop = new_pop[:cfg["POP_SIZE"]]
        fit, trajs = evaluate(pop, cfg)

        gb = max(fit)
        ga = sum(fit) / len(fit)
        history.append((gb, ga))
        recent.append(gb)

        gi = fit.index(gb)
        if gb > best_fit:
            best_fit    = gb
            best_genome = copy.deepcopy(pop[gi])
            best_traj   = trajs[gi]

        ns = len(best_traj.cell_counts) if best_traj and best_traj.cell_counts else 0
        outcome, outcome_time = best_traj.classify_outcome(cfg) if best_traj else ("FATAL", 0)
        ttp = best_traj.time_to_progression(cfg) if best_traj else None
        print(f"\n  >> Best: {gb:,.0f}  Avg: {ga:,.0f}  "
              f"vs baseline: {gb - baseline_fit:+,.0f}  "
              f"rules: {len(best_genome.rules)}")
        if best_traj and ns:
            print(f"     mgd={best_traj.managed_steps}/{ns}  "
                  f"cyc={best_traj.num_cycles}  "
                  f"hol={best_traj.holiday_steps}/{ns}  "
                  f"ovf={best_traj.overflow_steps}/{ns}  "
                  f"final={best_traj.final:,}  "
                  f"dose={best_traj.cumulative_dose:,.0f}")
            ttp_str = f"t={ttp}" if ttp else "none"
            ot_str = f"t={outcome_time}" if outcome_time else "end"
            print(f"     outcome={outcome} ({ot_str})  TTP={ttp_str}")

        # ── Save trajectory data for best genome this generation ───────────
        os.makedirs(cfg["OUTPUT_DIR"], exist_ok=True)
        if best_traj and best_traj.timesteps:
            gen_data = {
                "generation": gen + 1,
                "best_fitness": gb, "avg_fitness": ga,
                "burden": best_traj.burden, "final": best_traj.final,
                "peak": best_traj.peak, "dose": best_traj.cumulative_dose,
                "outcome": outcome, "outcome_time": outcome_time,
                "timesteps": best_traj.timesteps,
                "cell_counts": best_traj.cell_counts,
                "dose_per_step": best_traj.dose_per_step,
                "rules": [r.to_dict() for r in best_genome.rules],
            }
            gen_path = os.path.join(cfg["OUTPUT_DIR"],
                                    f"gen_{gen+1:03d}_best.json")
            with open(gen_path, "w") as jf:
                json.dump(gen_data, jf)

        # ── Checkpoint (resume support) ──────────────────────────────────────
        ckpt = {
            "generation": gen + 1, "best_fit": best_fit,
            "best_genome": [r.to_dict() for r in best_genome.rules],
            "history": history, "recent": recent,
            "hints": hints.as_dict(),
            "config_hash": f"{cfg['GRID_N']}_{cfg['CONTAINMENT_PCT']}_{cfg['RANDOM_SEED']}",
        }
        ckpt_path = os.path.join(cfg["OUTPUT_DIR"], "checkpoint.json")
        with open(ckpt_path, "w") as jf:
            json.dump(ckpt, jf)

        # ── Early termination checks ────────────────────────────────────────
        min_gen = cfg["MIN_EARLY_STOP_GEN"]

        # SOLVED: cure achieved with efficient drug use (after min_gen)
        dose_eff = best_traj.cumulative_dose / max(1, cap)
        if (gen + 1) >= min_gen and outcome == "CURE" \
                and dose_eff < cfg["DOSE_EFFICIENCY_THRESHOLD"]:
            print(f"\n  ** SOLVED: cure with dose/capacity = {dose_eff:.3f} "
                  f"(< {cfg['DOSE_EFFICIENCY_THRESHOLD']})")
            print(f"  ** Stopping early at generation {gen+1}.")
            break

        # CONVERGED: fitness stable and outcome is acceptable (after min_gen)
        conv_w = cfg["CONVERGENCE_WINDOW"]
        if (gen + 1) >= max(min_gen, conv_w) and len(recent) >= conv_w:
            window = recent[-conv_w:]
            spread = max(window) - min(window)
            rel_spread = spread / max(1, abs(best_fit))
            if rel_spread < 0.001 and outcome in ("CURE", "MANAGED"):
                print(f"\n  ** CONVERGED: fitness stable for {conv_w} gens "
                      f"(spread={spread:.1f}), outcome={outcome}")
                print(f"  ** Stopping early at generation {gen+1}.")
                break

    # ── Final ────────────────────────────────────────────────────────────────
    prune_genome(best_genome)

    final_path = os.path.join(cfg["ROUTINES_DIR"], "best_routine.py")
    write_routine(best_genome, final_path, cfg)
    best_traj = run_simulation(final_path,
                                os.path.join(cfg["SAMPLES_DIR"], "best_final"), cfg)
    best_fit = compute_fitness(best_traj, cfg)
    ns = len(best_traj.cell_counts) if best_traj.cell_counts else 0

    print("\n" + "=" * 60)
    danger_pct = int(100 * cfg["DANGER_CEIL"] / cap)
    print(f"  BEST SCHEDULE")
    print(f"  Grid: {cfg['GRID_N']}³ = {cap:,}  |  "
          f"Danger ceiling: {cfg['DANGER_CEIL']:,} ({danger_pct}%)")
    print("=" * 60)
    print(f"  Fitness:          {best_fit:,.0f}  (baseline: {baseline_fit:,.0f})")
    if best_traj:
        outcome, outcome_time = best_traj.classify_outcome(cfg)
        ttp = best_traj.time_to_progression(cfg)
        bl_burden = cfg.get("_baseline_burden", 0)
        if bl_burden > 0:
            pct_change = ((best_traj.burden - bl_burden) / bl_burden) * 100
            burden_label = f"({pct_change:+.1f}% vs untreated)"
        else:
            burden_label = ""
        print(f"  Total burden:     {best_traj.burden:,}  {burden_label}")
        print(f"  Managed steps:    {best_traj.managed_steps}/{ns}  "
              f"(0 < cells \u2264 {cfg['DANGER_CEIL']:,})")
        print(f"  Treatment cycles: {best_traj.num_cycles}")
        print(f"  Holiday steps:    {best_traj.holiday_steps}/{ns}")
        print(f"  Overflow steps:   {best_traj.overflow_steps}/{ns}  "
              f"(cells > {cfg['DANGER_CEIL']:,})")
        print(f"  Final cells:      {best_traj.final:,}")
        print(f"  Peak cells:       {best_traj.peak:,}")
        print(f"  Cumulative dose:  {best_traj.cumulative_dose:,.0f}")
        print(f"  Dose efficiency:  {best_traj.cumulative_dose / max(1, cap):.3f} "
              f"(dose / capacity)")
        print()
        ot_str = f"at t={outcome_time}" if outcome_time else "through end"
        print(f"  OUTCOME:          {outcome} ({ot_str})")
        if outcome == "CURE":
            print(f"    Complete response: cells = 0 for {cfg['CURE_WINDOW']} "
                  f"consecutive snapshots")
        elif outcome == "FATAL":
            print(f"    Patient failure: cells > {cfg['FATAL_CEIL']:,} "
                  f"for {cfg['FATAL_WINDOW']} consecutive snapshots")
        elif outcome == "PROGRESSIVE":
            print(f"    Progressive disease: tumor growing on treatment "
                  f"for {cfg['PROGRESSION_WINDOW']} consecutive snapshots")
        elif outcome == "MANAGED":
            print(f"    Stable disease: tumor controlled through end of simulation")
        if ttp is not None:
            print(f"  Time to progression: {ttp}")
        else:
            print(f"  Time to progression: none (no progression detected)")

    print(f"\n  Drug schedule ({len(best_genome.rules)} rules):")
    for r in best_genome.rules:
        if r.rule_type == "cycling":
            print(f"    [CYCLING] dose={r.dosage:>8.2f}  "
                  f"reduce_to={r.reduction_target:.0%}  "
                  f"restart_at={r.regrowth_trigger:.0%}  "
                  f"start>={r.start_threshold:,}")
        else:
            print(f"    [THRESH]  dose={r.dosage:>8.2f}  "
                  f"cells>={r.cell_threshold:,}")

    print(f"\n  Saved: {final_path}")

    # Adaptation log
    if hints.log:
        print("\n  Search adaptation log:")
        print(f"    {'Gen':>4s}  {'Dose range':>14s}  {'Managed':>8s}  "
              f"{'Cycles':>7s}  {'Hol':>5s}  {'Overflow':>8s}  "
              f"{'Cum dose':>10s}  {'Final':>6s}")
        for e in hints.log:
            print(f"    {e['gen']:>4d}  {e['dose']:>14s}  {e['mgd']:>8s}  "
                  f"{e['cyc']:>7s}  {e['hol']:>5s}  {e['ovf']:>8s}  "
                  f"{e['cdose']:>10s}  {e['fin']:>6s}")

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
        axes[0].set_title(f"Adaptive Therapy GA — {cfg['GRID_N']}³ grid "
                          f"(outcome-driven fitness)")
        axes[0].legend()

        if best_traj and best_traj.timesteps:
            ax2 = axes[1]
            ax2.plot(best_traj.timesteps, best_traj.cell_counts,
                     'g-', lw=2, label="Best schedule")
            if baseline_traj and baseline_traj.timesteps:
                ax2.plot(baseline_traj.timesteps, baseline_traj.cell_counts,
                         'r:', lw=1.5, label="No treatment")
            ax2.axhline(y=cfg["DANGER_CEIL"], color='orange', ls='--',
                        label=f"Danger ceiling ({cfg['DANGER_CEIL']:,})")
            if best_traj.dose_per_step and len(best_traj.dose_per_step) == len(best_traj.timesteps):
                on_t  = [best_traj.timesteps[j] for j in range(len(best_traj.timesteps))
                         if best_traj.dose_per_step[j] > 0]
                on_c  = [best_traj.cell_counts[j] for j in range(len(best_traj.timesteps))
                         if best_traj.dose_per_step[j] > 0]
                off_t = [best_traj.timesteps[j] for j in range(len(best_traj.timesteps))
                         if best_traj.dose_per_step[j] == 0]
                off_c = [best_traj.cell_counts[j] for j in range(len(best_traj.timesteps))
                         if best_traj.dose_per_step[j] == 0]
                if on_t:
                    ax2.scatter(on_t, on_c, c='blue', s=20, zorder=5,
                                label="Treating", marker='v')
                if off_t:
                    ax2.scatter(off_t, off_c, c='gray', s=20, zorder=5,
                                label="Holiday", marker='o')
            ax2.set_xlabel("Timestep")
            ax2.set_ylabel("Alive Cells")
            ax2.set_title("Tumor Trajectory: Best Schedule vs Untreated")
            ax2.legend()

        plt.tight_layout()
        plot_path = os.path.join(cfg["OUTPUT_DIR"], "fitness_plot_0.1warmup.png")
        plt.savefig(plot_path, dpi=150)
        print(f"\nPlot: {plot_path}")

    return best_genome, best_fit


if __name__ == "__main__":
    cfg = get_config()
    genetic_algorithm(cfg)