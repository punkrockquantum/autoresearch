"""The loop that ties it all together.

One step of the loop is one trial's worth of compute, spent on the most valuable
thing available, in this priority order:

    1. a finding a human challenged   (their question outranks our curiosity)
    2. a supported finding's exhaustive confirmation grid
    3. the screening bandit over open hypotheses
    4. a fresh baseline measurement

After every trial the affected hypothesis is re-decided from scratch by
`arp.stats.decide`. When a hypothesis clears screening it is criticised (by the
platform itself) and pushed into the exhaustive stage; when it clears the whole
grid with no open challenges it becomes PROVEN, its parameters are adopted as
the new baseline, the capability memory learns from the outcome, and the web and
impact agents work out whether the result is worth money.

The loop is resumable: all state lives in SQLite, and every seed is derived from
the run salt, so `arp run --salt <same>` replays the same search.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from arp.agents import Critic, HypothesisProposer, LLMClient, ProposalContext
from arp.config import PlatformConfig, default_config
from arp.evolution import CapabilityMemory, operator_catalog
from arp.exhaustive import ConfirmationResult, cell_key, cell_overrides, evaluate_confirmation
from arp.impact import rank_opportunities
from arp.interventions import auto_challenge, challenge_axes, record_prompt, resolve_challenges
from arp.models import (
    Finding,
    Hypothesis,
    Run,
    Subject,
    Trial,
    TrialSpec,
    Verdict,
    now_iso,
)
from arp.runners import Runner, build_runner
from arp.search import ArmStats, ThompsonAllocator
from arp.stats import decide
from arp.store import Store
from arp.websearch import gather_leads, get_provider

MAX_CRASHES_PER_HYPOTHESIS = 3
MAX_CONSECUTIVE_FAILURES = 5


@dataclass
class StepResult:
    index: int
    action: str                      # screen | confirm | baseline | idle
    ok: bool = True
    hypothesis_id: Optional[str] = None
    arm: str = ""
    metric: Optional[float] = None
    verdict: Optional[str] = None
    note: str = ""
    cell: Dict[str, Any] = field(default_factory=dict)

    def line(self) -> str:
        bits = [f"[{self.index:04d}]", self.action]
        if self.arm:
            bits.append(self.arm)
        if self.metric is not None:
            bits.append(f"metric={self.metric:.6f}")
        elif not self.ok:
            bits.append("FAILED")
        if self.verdict:
            bits.append(f"-> {self.verdict}")
        if self.note:
            bits.append(f"({self.note})")
        return " ".join(bits)


@dataclass
class RunSummary:
    run_id: str
    steps: List[StepResult] = field(default_factory=list)
    proven: List[str] = field(default_factory=list)
    refuted: List[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    aborted: str = ""      # set when the run stopped because the setup is broken

    @property
    def trials(self) -> int:
        return sum(1 for s in self.steps if s.action in ("screen", "confirm", "baseline"))


class Orchestrator:
    def __init__(
        self,
        store: Store,
        subject: Subject,
        config: Optional[PlatformConfig] = None,
        runner: Optional[Runner] = None,
        llm: Optional[LLMClient] = None,
        salt: Optional[str] = None,
        search_web: bool = True,
    ):
        self.store = store
        self.subject = subject
        self.config = config or default_config()
        self.salt = salt or subject.config.get("salt") or subject.slug
        self.runner = runner or build_runner(subject, self.config.work_dir)
        self.llm = llm or LLMClient()
        self.memory = CapabilityMemory(store)
        self.proposer = HypothesisProposer(store, self.memory, self.llm)
        self.critic = Critic(store, self.llm)
        self.allocator = ThompsonAllocator(self.config.search, salt=self.salt)
        self.search_web = search_web
        self._step_index = 0

    # -- state helpers -----------------------------------------------------

    @property
    def epoch(self) -> int:
        return int(self.subject.config.get("epoch", 0))

    def baseline_params(self) -> Dict[str, Any]:
        return dict(self.subject.baseline_params)

    def _screen_trials(self, arm: str, hypothesis_id: Optional[str] = None) -> List[float]:
        trials = self.store.list_trials(
            subject_id=self.subject.id, hypothesis_id=hypothesis_id, arm=arm, stage="screen"
        )
        return [
            t.metric_value for t in trials
            if t.metric_value is not None and int(t.aux.get("epoch", 0)) == self.epoch
        ]

    def _extra_axes(self, finding: Optional[Finding]) -> Dict[str, Any]:
        """Confirmation axes: the subject's, plus any challenge's, plus the epoch.

        Tagging the grid with the epoch means that adopting a new baseline
        automatically invalidates confirmation evidence gathered against the old
        one, instead of quietly mixing the two.
        """
        axes: Dict[str, Any] = {"epoch": [self.epoch]}
        if finding is not None:
            axes.update(challenge_axes(self.store, finding))
        return axes

    def _arm_stats(self, hypotheses: Sequence[Hypothesis]) -> Dict[str, ArmStats]:
        baseline = self._screen_trials("baseline")
        baseline_mean = sum(baseline) / len(baseline) if baseline else None
        stats: Dict[str, ArmStats] = {}
        for hyp in hypotheses:
            values = self._screen_trials("treatment", hyp.id)
            gains = []
            if baseline_mean is not None:
                for v in values:
                    gains.append(self.subject.relative_gain(baseline_mean, v))
            stats[hyp.id] = ArmStats.from_values(gains)
        return stats

    def _confirm_progress(self, finding: Finding) -> int:
        """How many confirmation trials this finding's grid already holds."""
        return len(self.store.list_trials(
            subject_id=self.subject.id, hypothesis_id=finding.hypothesis_id, stage="confirm"
        ))

    def _crash_count(self, hypothesis_id: str) -> int:
        trials = self.store.list_trials(
            subject_id=self.subject.id, hypothesis_id=hypothesis_id, ok_only=False
        )
        return sum(1 for t in trials if not t.ok)

    def _boldness(self) -> float:
        """Get more adventurous after a dry spell, calmer after a win."""
        recent = self.store.list_findings(
            self.subject.id, [Verdict.REFUTED, Verdict.INCONCLUSIVE, Verdict.PROVEN]
        )
        if not recent:
            return 1.0
        recent_sorted = sorted(recent, key=lambda f: f.updated_at, reverse=True)[:6]
        wins = sum(1 for f in recent_sorted if f.verdict == Verdict.PROVEN)
        return 1.0 if wins else min(2.0, 1.0 + 0.2 * len(recent_sorted))

    # -- frontier ----------------------------------------------------------

    def ensure_frontier(self, minimum: int = 3) -> List[Hypothesis]:
        """Keep enough untested hypotheses around for the bandit to choose from."""
        open_hyps = [h for h in self.store.open_hypotheses(self.subject.id)
                     if h.status in (Verdict.PROPOSED, Verdict.TESTING)]
        want = max(0, min(minimum, self.config.search.max_open_hypotheses - len(open_hyps)))
        if len(open_hyps) >= minimum or want == 0:
            return open_hyps
        ctx = ProposalContext(
            subject=self.subject,
            best_params=self.baseline_params(),
            recent_findings=self.store.list_findings(self.subject.id)[:12],
            recent_prompts=self.store.list_prompts(self.subject.id, limit=10),
            boldness=self._boldness(),
            epoch=self.epoch,
        )
        for hyp in self.proposer.propose(ctx, k=want):
            self.store.add_hypothesis(hyp)
            open_hyps.append(hyp)
        return open_hyps

    # -- trial execution ---------------------------------------------------

    def _execute(
        self, hypothesis: Optional[Hypothesis], arm: str, seed: int, stage: str,
        cell: Optional[Dict[str, Any]] = None,
    ) -> Trial:
        cell = cell or {}
        params = self.baseline_params()
        if arm == "treatment" and hypothesis is not None:
            params.update(hypothesis.params)
        params.update(cell_overrides(self.subject, cell, self._extra_axes(None)))

        spec = TrialSpec(
            subject=self.subject, params=params, seed=seed, arm=arm, cell=cell,
            hypothesis_id=hypothesis.id if hypothesis else None,
            timeout_s=float(self.subject.config.get("runner_config", {}).get("timeout_s", 1800)),
        )
        result = self.runner.run(spec)
        trial = Trial(
            subject_id=self.subject.id,
            hypothesis_id=hypothesis.id if hypothesis else None,
            arm=arm,
            seed=seed,
            stage=stage,
            metric_value=result.metric,
            ok=result.ok,
            params=params,
            cell=cell,
            aux={**result.aux, "epoch": self.epoch, "log_path": result.log_path},
            error=result.error,
            duration_s=result.duration_s,
        )
        return self.store.add_trial(trial)

    # -- verdict bookkeeping ----------------------------------------------

    def _finding_for(self, hypothesis: Hypothesis) -> Finding:
        finding = self.store.finding_for_hypothesis(hypothesis.id)
        if finding is None:
            finding = Finding(subject_id=self.subject.id, hypothesis_id=hypothesis.id)
            self.store.save_finding(finding)
        return finding

    def _update_screening(self, hypothesis: Hypothesis) -> Finding:
        finding = self._finding_for(hypothesis)
        previous = finding.verdict
        open_challenges = len(self.store.open_challenges(finding.id))
        baseline = self._screen_trials("baseline")
        treatment = self._screen_trials("treatment", hypothesis.id)
        decision = decide(baseline, treatment, self.subject.direction,
                          self.config.decision, open_challenges)

        finding.effect = decision.estimate.effect
        finding.ci_low = decision.estimate.ci_low
        finding.ci_high = decision.estimate.ci_high
        finding.p_value = decision.estimate.p_value
        finding.n_trials = decision.estimate.n_treatment
        finding.reason = decision.reason
        finding.evidence = {**finding.evidence, "estimate": decision.estimate.as_dict(),
                            "policy": decision.stats, "stage": "screen"}
        # A contested finding never silently drops back to a softer verdict.
        if decision.status == Verdict.SUPPORTED and open_challenges:
            finding.verdict = Verdict.CONTESTED
        else:
            finding.verdict = decision.status
        self.store.save_finding(finding)

        hypothesis.status = finding.verdict
        self.store.save_hypothesis(hypothesis)

        if decision.status in (Verdict.REFUTED, Verdict.INCONCLUSIVE):
            self.memory.record_verdict(self.subject, hypothesis.operator, decision.status,
                                       decision.estimate.effect)
            resolve_challenges(self.store, finding, decision.status, decision.reason)
        elif decision.status == Verdict.SUPPORTED and previous != Verdict.SUPPORTED:
            # On promotion, and only on promotion, argue against it. A
            # high-severity concern becomes a challenge and raises the bar now,
            # rather than after we have told someone it works.
            auto_challenge(self.store, self.subject, finding, self.critic)
        return finding

    def _update_confirmation(self, hypothesis: Hypothesis) -> ConfirmationResult:
        finding = self._finding_for(hypothesis)
        was_proven = finding.verdict == Verdict.PROVEN
        open_challenges = len(self.store.open_challenges(finding.id))
        result = evaluate_confirmation(
            self.store, self.subject, hypothesis, self.config.decision, self.salt,
            self._extra_axes(finding), open_challenges,
        )
        finding.evidence = {**finding.evidence, "confirmation": result.as_dict()}
        if result.pooled.n_treatment:
            finding.effect = result.pooled.effect
            finding.ci_low = result.pooled.ci_low
            finding.ci_high = result.pooled.ci_high
            finding.p_value = result.pooled.p_value
            finding.n_trials = result.pooled.n_treatment
        finding.verdict = result.status
        finding.reason = result.reason
        if result.status == Verdict.PROVEN:
            finding.proven_at = now_iso()
        self.store.save_finding(finding)

        hypothesis.status = result.status
        self.store.save_hypothesis(hypothesis)

        if result.status in Verdict.TERMINAL:
            self.memory.record_verdict(self.subject, hypothesis.operator, result.status, finding.effect)
            resolve_challenges(self.store, finding, result.status, result.reason)
        if result.status == Verdict.PROVEN and not was_proven:
            # Only on the transition: re-evaluating a proven finding must not
            # adopt it (and bump the epoch) a second time.
            self._adopt(hypothesis, finding)
        return result

    def _adopt(self, hypothesis: Hypothesis, finding: Finding) -> None:
        """A proven change becomes the new baseline, and the next epoch begins.

        This is the platform's version of `git commit` in the manual autoresearch
        loop: subsequent hypotheses are measured against the improved setup, not
        against the original one.
        """
        previous = self.baseline_params()
        params = dict(previous)
        params.update(hypothesis.params)
        self.subject.config["baseline_params"] = params
        self.subject.config["epoch"] = self.epoch + 1
        history = list(self.subject.config.get("adopted", []))
        history.append({
            "hypothesis_id": hypothesis.id, "finding_id": finding.id,
            "title": hypothesis.title, "effect": round(finding.effect, 6), "at": now_iso(),
            # What the baseline held before this change, so a later challenge can
            # roll it back and re-run the comparison honestly.
            "previous": {k: previous.get(k) for k in hypothesis.params},
        })
        self.subject.config["adopted"] = history
        self.store.save_subject(self.subject)
        self._retire_superseded(hypothesis)
        record_prompt(
            self.store, self.subject,
            f"Adopted proven change: {hypothesis.title} ({finding.effect:+.3%} on {self.subject.metric}). "
            f"Baseline is now epoch {self.epoch}.",
            kind="note", role="platform", meta={"finding_id": finding.id},
        )
        self._value_pass(finding, hypothesis)

    def _retire_superseded(self, adopted: Hypothesis) -> None:
        """Close open hypotheses that the adoption just made obsolete.

        A proposal like "set lr to 0.015" was a question about the old baseline.
        Once lr has moved to 0.022 the question is not merely unanswered, it is
        the wrong question — and its screening evidence belongs to a previous
        epoch anyway. Retiring it frees the bandit to propose against the setup
        that now exists, which is what a human would do after committing a win.
        """
        touched = set(adopted.params or {})
        if not touched:
            return
        for hyp in self.store.open_hypotheses(self.subject.id):
            if hyp.id == adopted.id or hyp.status not in (Verdict.PROPOSED, Verdict.TESTING):
                continue
            if not touched & set(hyp.params or {}):
                continue
            finding = self._finding_for(hyp)
            finding.verdict = Verdict.INCONCLUSIVE
            finding.reason = (
                f"superseded when '{adopted.title}' was adopted: this proposal was measured "
                f"against the epoch-{self.epoch - 1} baseline for "
                f"{', '.join(sorted(touched & set(hyp.params)))}"
            )
            self.store.save_finding(finding)
            hyp.status = Verdict.INCONCLUSIVE
            self.store.save_hypothesis(hyp)

    def _value_pass(self, finding: Finding, hypothesis: Hypothesis) -> None:
        """Ask the outside world what this result is worth, then score it."""
        if self.search_web:
            try:
                provider = get_provider()
                if provider.available:
                    gather_leads(self.store, self.subject, finding, hypothesis, provider)
            except Exception as exc:  # network agents must never kill a research run
                record_prompt(self.store, self.subject, f"web search failed: {exc}",
                              kind="note", role="platform")
        rank_opportunities(self.store, self.subject)

    # -- the step ----------------------------------------------------------

    def step(self) -> StepResult:
        self._step_index += 1
        index = self._step_index
        # Re-read the subject: adoption in an earlier step (or a prompt from
        # another process) may have moved the baseline under us. The runner gets
        # the fresh copy too, or it would keep measuring an obsolete baseline.
        self.subject = self.store.get_subject(self.subject.id) or self.subject
        self.runner.subject = self.subject

        # 1. Challenged findings first: a human is waiting on the answer.
        # 2. Then confirmation grids for supported findings.
        #
        # Both are worked *one at a time*, most-advanced first. Spreading the
        # budget evenly over nine half-finished grids produces nine findings that
        # are nearly proven, which is worth nothing; finishing one produces an
        # answer. The rest lose nothing by waiting — their evidence keeps.
        for verdict in (Verdict.CONTESTED, Verdict.SUPPORTED):
            candidates = self.store.list_findings(self.subject.id, [verdict])
            candidates.sort(key=lambda f: (self._confirm_progress(f), f.effect), reverse=True)
            for finding in candidates:
                hyp = self.store.get_hypothesis(finding.hypothesis_id)
                if hyp is None:
                    continue
                result = self._run_confirmation_trial(hyp, finding, index, stage_label="confirm")
                if result is not None:
                    return result

        # 3. Screening bandit.
        hypotheses = self.ensure_frontier()
        stats = self._arm_stats(hypotheses)
        tried = {h.operator for h in self.store.list_hypotheses(self.subject.id)
                 if stats.get(h.id, ArmStats()).n > 0}
        priors = self.memory.priors(self.subject, operator_catalog(self.subject))
        baseline_n = len(self._screen_trials("baseline"))
        treatment_n = sum(stats.get(h.id, ArmStats()).n for h in hypotheses)
        allocation = self.allocator.allocate(
            self.subject, hypotheses, stats, priors, tried, step=index,
            baseline_n=baseline_n, treatment_n=treatment_n,
        )

        if not hypotheses and baseline_n >= self.config.search.min_baseline:
            # Nothing left to test and the baseline is already well measured.
            # Spending the rest of the budget re-measuring it would look like
            # progress in the log and be worth nothing.
            return StepResult(
                index, "idle",
                note="no open hypotheses and no confirmation work; the frontier is exhausted",
            )

        if allocation.arm == "baseline" or allocation.hypothesis is None:
            trial = self._execute(None, "baseline", allocation.seed, "screen")
            # A new baseline changes every open comparison, so re-decide them all.
            for hyp in hypotheses:
                self._update_screening(hyp)
            note = allocation.reason if trial.ok else f"{allocation.reason} | error: {trial.error[:200]}"
            return StepResult(index, "baseline", ok=trial.ok, arm="baseline",
                              metric=trial.metric_value, note=note)

        hyp = allocation.hypothesis
        trial = self._execute(hyp, "treatment", allocation.seed, "screen")
        if not trial.ok:
            return self._handle_crash(hyp, trial, index, allocation.reason)
        finding = self._update_screening(hyp)
        return StepResult(
            index, "screen", ok=True, hypothesis_id=hyp.id, arm="treatment",
            metric=trial.metric_value, verdict=finding.verdict,
            note=f"{hyp.title} | {finding.reason}",
        )

    def _run_confirmation_trial(
        self, hypothesis: Hypothesis, finding: Finding, index: int, stage_label: str,
    ) -> Optional[StepResult]:
        """Run the next missing trial of this finding's grid, if any.

        The queue comes from `evaluate_confirmation` rather than from a plain
        plan, because that is what knows about escalation: a cell whose interval
        still spans the regression tolerance asks for replicates beyond the
        floor, and planning without that would leave every grid stalled one step
        short of an answer.
        """
        result = self._update_confirmation(hypothesis)
        if not result.pending:
            return StepResult(
                index, "confirm", hypothesis_id=hypothesis.id, verdict=result.status,
                note=result.reason,
            )
        request = result.pending[0]
        trial = self._execute(
            hypothesis if request.arm == "treatment" else None,
            request.arm, request.seed, "confirm", request.cell,
        )
        if not trial.ok and request.arm == "treatment":
            crashed = self._handle_crash(hypothesis, trial, index, "confirmation trial failed")
            return crashed
        result = self._update_confirmation(hypothesis)
        return StepResult(
            index, "confirm", ok=trial.ok, hypothesis_id=hypothesis.id, arm=request.arm,
            metric=trial.metric_value, verdict=result.status, cell=request.cell,
            note=f"cell {cell_key(request.cell)} | {result.reason}",
        )

    def _handle_crash(self, hypothesis: Hypothesis, trial: Trial, index: int, note: str) -> StepResult:
        crashes = self._crash_count(hypothesis.id)
        if crashes >= MAX_CRASHES_PER_HYPOTHESIS:
            finding = self._finding_for(hypothesis)
            finding.verdict = Verdict.REFUTED
            finding.reason = f"runner failed {crashes} times: {trial.error[:200]}"
            self.store.save_finding(finding)
            hypothesis.status = Verdict.REFUTED
            self.store.save_hypothesis(hypothesis)
            self.memory.record_verdict(self.subject, hypothesis.operator, Verdict.REFUTED, 0.0)
            resolve_challenges(self.store, finding, Verdict.REFUTED, finding.reason)
            verdict = Verdict.REFUTED
        else:
            verdict = None
        return StepResult(
            index, "screen", ok=False, hypothesis_id=hypothesis.id, arm="treatment",
            verdict=verdict, note=f"{note} | error: {trial.error[:160]}",
        )

    # -- the loop ----------------------------------------------------------

    def run(
        self, steps: int = 20, budget_seconds: Optional[float] = None,
        on_step: Optional[Callable[[StepResult], None]] = None,
    ) -> RunSummary:
        run = self.store.add_run(Run(subject_id=self.subject.id, salt=self.salt, steps_requested=steps))
        summary = RunSummary(run_id=run.id)
        t0 = time.time()
        idle_streak = 0
        failure_streak = 0
        try:
            for _ in range(steps):
                if budget_seconds is not None and time.time() - t0 >= budget_seconds:
                    run.notes = "time budget reached"
                    break
                result = self.step()
                summary.steps.append(result)
                if on_step:
                    on_step(result)
                idle_streak = idle_streak + 1 if result.action == "idle" else 0
                if idle_streak >= 3:
                    run.notes = "frontier exhausted: no hypotheses left to test"
                    break

                # A setup that cannot run is not a research result. Five failures
                # in a row means the command, the environment or the parameter
                # space is wrong, and grinding through the budget would just bury
                # the error under a hundred identical ones.
                failure_streak = failure_streak + 1 if not result.ok else 0
                if failure_streak >= MAX_CONSECUTIVE_FAILURES:
                    run.status = "failed"
                    run.notes = (
                        f"aborted after {failure_streak} consecutive failed trials — "
                        f"last error: {result.note[:300]}"
                    )
                    summary.aborted = run.notes
                    break
                if result.verdict == Verdict.PROVEN and result.hypothesis_id:
                    summary.proven.append(result.hypothesis_id)
                elif result.verdict == Verdict.REFUTED and result.hypothesis_id:
                    summary.refuted.append(result.hypothesis_id)
            if run.status == "running":
                run.status = "finished"
        except KeyboardInterrupt:
            run.status = "interrupted"
            run.notes = "interrupted by user"
        finally:
            run.steps_done = len(summary.steps)
            run.ended_at = now_iso()
            self.store.save_run(run)
            summary.elapsed_s = time.time() - t0
        return summary

    # -- reporting ---------------------------------------------------------

    def state(self) -> Dict[str, Any]:
        findings = self.store.list_findings(self.subject.id)
        by_verdict: Dict[str, int] = {}
        for f in findings:
            by_verdict[f.verdict] = by_verdict.get(f.verdict, 0) + 1
        skill = self.memory.skill_profile(self.subject)
        return {
            "subject": self.subject.slug,
            "epoch": self.epoch,
            "baseline_params": self.baseline_params(),
            "trials": self.store.count_trials(self.subject.id),
            "findings": by_verdict,
            "open_challenges": len(self.store.list_challenges(subject_id=self.subject.id, status="open")),
            "skill_level": skill.level,
            "top_operators": skill.top_operators(),
            "salt": self.salt,
        }
