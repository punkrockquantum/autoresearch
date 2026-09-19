"""Paths and tunables. Everything is overridable by environment variable."""

import os
from dataclasses import dataclass, field

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Where the platform keeps its state. Defaults next to the autoresearch cache so
# the two live together, but tests and demos point it somewhere disposable.
HOME_DIR = os.environ.get(
    "ARP_HOME", os.path.join(os.path.expanduser("~"), ".cache", "autoresearch", "arp")
)
DB_PATH = os.environ.get("ARP_DB", os.path.join(HOME_DIR, "platform.db"))
WORK_DIR = os.environ.get("ARP_WORK", os.path.join(HOME_DIR, "work"))

# Agent/LLM layer. Absent keys are fine: every agent has a deterministic
# heuristic fallback, so the platform is fully functional offline.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ARP_MODEL", "claude-sonnet-5")
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
HTTP_TIMEOUT = float(os.environ.get("ARP_HTTP_TIMEOUT", "30"))


@dataclass
class DecisionPolicy:
    """The rules that turn noisy trials into a deterministic verdict.

    These are deliberately explicit numbers rather than agent judgement: given
    the same trial table and the same policy, `arp.stats.decide` always returns
    the same verdict, on any machine, forever.
    """

    alpha: float = 0.05             # false-positive budget for the screening stage
    alpha_confirm: float = 0.01     # stricter budget for the exhaustive stage
    min_replicates: int = 3         # per arm, before any verdict may be issued
    max_replicates: int = 24        # per arm, before we call it inconclusive
    mde: float = 0.002              # minimum detectable effect worth keeping (relative)
    rope: float = 0.001             # region of practical equivalence (relative)
    cs_rho: float = 1.0             # confidence-sequence mixture tuning parameter
    regression_tolerance: float = 0.0005  # per-cell slack in the exhaustive grid
    challenge_alpha_factor: float = 0.25  # each open challenge tightens alpha
    challenge_extra_replicates: int = 3   # ...and buys more evidence

    def tightened(self, open_challenges: int) -> "DecisionPolicy":
        """A stricter copy of this policy, one notch per unresolved challenge.

        This is the mechanical form of "the human second-guessed the result":
        the bar goes up and stays up until the thesis clears it.
        """
        if open_challenges <= 0:
            return self
        factor = self.challenge_alpha_factor ** open_challenges
        return DecisionPolicy(
            alpha=self.alpha * factor,
            alpha_confirm=self.alpha_confirm * factor,
            min_replicates=self.min_replicates + self.challenge_extra_replicates * open_challenges,
            max_replicates=self.max_replicates + self.challenge_extra_replicates * open_challenges * 2,
            mde=self.mde,
            rope=self.rope,
            cs_rho=self.cs_rho,
            regression_tolerance=self.regression_tolerance,
            challenge_alpha_factor=self.challenge_alpha_factor,
            challenge_extra_replicates=self.challenge_extra_replicates,
        )


@dataclass
class SearchPolicy:
    """How aggressively the bandit explores."""

    novelty_bonus: float = 0.35     # optimism for operators never tried on this subject
    min_baseline: int = 3           # measure the baseline this often before comparing anything
    baseline_ratio: float = 0.35    # keep baseline evidence this close to treatment evidence
    baseline_refresh_every: int = 12  # re-measure the baseline arm this often regardless, for drift
    exploration_floor: float = 0.1  # min probability of a purely exploratory pick
    max_open_hypotheses: int = 12   # keep the frontier bounded


@dataclass
class PlatformConfig:
    db_path: str = DB_PATH
    work_dir: str = WORK_DIR
    decision: DecisionPolicy = field(default_factory=DecisionPolicy)
    search: SearchPolicy = field(default_factory=SearchPolicy)

    def ensure_dirs(self) -> None:
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        os.makedirs(self.work_dir, exist_ok=True)


def default_config() -> PlatformConfig:
    cfg = PlatformConfig()
    cfg.ensure_dirs()
    return cfg
