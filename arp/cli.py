"""`arp` — the command line for the agentic research platform.

    arp init
    arp subject add --preset autoresearch
    arp prompt nanochat-pretrain "focus on the LR schedule, depth is a dead end"
    arp run nanochat-pretrain --steps 50
    arp status nanochat-pretrain
    arp challenge nanochat-pretrain --finding fnd_abc123 --reason "smells like noise"
    arp suggest
    arp report nanochat-pretrain --html report.html
    arp demo
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from arp import __version__
from arp.config import PlatformConfig, default_config
from arp.evolution import CapabilityMemory
from arp.impact import rank_opportunities
from arp.interventions import apply_prompt, open_challenge
from arp.models import Subject, Verdict
from arp.orchestrator import Orchestrator
from arp.report import write_report
from arp.store import Store
from arp.suggest import polish, suggest
from arp.websearch import gather_leads, get_provider


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

def preset_subject(name: str) -> Subject:
    if name == "autoresearch":
        from arp.adapters.autoresearch_train import autoresearch_subject

        return autoresearch_subject()
    if name == "demo":
        return demo_subject()
    raise SystemExit(f"unknown preset {name!r}; try 'autoresearch' or 'demo'")


def demo_subject(slug: str = "demo-optimizer") -> Subject:
    """A synthetic subject with a known ground truth, for offline end-to-end runs.

    `lr` genuinely helps (up to a point), `momentum` does nothing. A correct
    platform proves the first and refuses to prove the second.
    """
    return Subject(
        slug=slug,
        title="Tune a synthetic optimiser to minimise simulated validation loss",
        metric="val_loss",
        direction="minimize",
        runner="simulated",
        description=(
            "Offline demonstration subject with a known response surface: the learning "
            "rate has a real effect with diminishing returns, momentum has none."
        ),
        config={
            "param_space": {
                "lr": {"type": "float", "default": 0.01, "low": 0.0005, "high": 0.5, "log": True,
                       "group": "lr", "step": 0.35},
                "momentum": {"type": "float", "default": 0.9, "low": 0.5, "high": 0.99, "log": False,
                             "scale": 0.1, "group": "momentum", "step": 0.2},
                "width": {"type": "int", "default": 256, "low": 64, "high": 1024, "step": 64,
                          "group": "architecture"},
            },
            "baseline_params": {"lr": 0.01, "momentum": 0.9, "width": 256},
            "stress_axes": {"scale": {"nominal": {}, "larger": {"width": 512}}},
            "simulation": {
                "baseline": 1.0,
                "noise": 0.0025,
                # lr genuinely helps (with diminishing returns), width helps a
                # little, momentum does nothing at all. A correct platform proves
                # the first two and refuses to prove the third.
                "coefs": {"lr": -0.03, "momentum": 0.0, "width": -0.006},
                "curvature": {"lr": 0.02},
                "cell_shift": {"scale": {"larger": -0.005}},
            },
            "value_model": {"reference_effect": 0.01, "effort_days": 3,
                            "value_per_relative_point": 250000},
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config(args: argparse.Namespace) -> PlatformConfig:
    cfg = default_config()
    if getattr(args, "db", None):
        cfg.db_path = args.db
        cfg.ensure_dirs()
    return cfg


def _store(args: argparse.Namespace) -> Store:
    return Store(_config(args).db_path)


def _subject(store: Store, slug: str) -> Subject:
    subject = store.get_subject(slug)
    if subject is None:
        raise SystemExit(f"no such subject: {slug}. Try `arp subject list`.")
    return subject


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str, sort_keys=True))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_init(args: argparse.Namespace) -> int:
    cfg = _config(args)
    store = Store(cfg.db_path)
    store.close()
    print(f"platform database ready at {cfg.db_path}")
    print(f"work directory: {cfg.work_dir}")
    return 0


def cmd_subject_add(args: argparse.Namespace) -> int:
    store = _store(args)
    if args.preset:
        subject = preset_subject(args.preset)
        if args.slug:
            subject.slug = args.slug
    else:
        if not args.slug or not args.title:
            raise SystemExit("--slug and --title are required without --preset")
        subject = Subject(
            slug=args.slug, title=args.title, metric=args.metric, direction=args.direction,
            runner=args.runner, description=args.description or "",
        )
    if args.config:
        with open(args.config) as f:
            subject.config.update(json.load(f))

    if store.get_subject(subject.slug):
        raise SystemExit(f"subject {subject.slug!r} already exists")
    store.add_subject(subject)

    # A brand-new subject inherits what similar subjects already taught us.
    seeded = CapabilityMemory(store).seed_from_similar(subject)
    print(f"created subject {subject.slug} ({subject.id})")
    print(f"  metric: {subject.metric} ({subject.direction}), runner: {subject.runner}")
    if seeded:
        print(f"  warm-started {seeded} operator prior(s) from similar subjects")
    return 0


def cmd_subject_list(args: argparse.Namespace) -> int:
    store = _store(args)
    subjects = store.list_subjects()
    if not subjects:
        print("no subjects yet — try `arp subject add --preset demo`")
        return 0
    for subject in subjects:
        findings = store.list_findings(subject.id, [Verdict.PROVEN])
        print(
            f"{subject.slug:28s} {subject.metric:12s} {subject.direction:8s} "
            f"trials={store.count_trials(subject.id):4d} proven={len(findings):2d}  {subject.title[:60]}"
        )
    return 0


def cmd_subject_show(args: argparse.Namespace) -> int:
    store = _store(args)
    subject = _subject(store, args.slug)
    _print_json({
        "id": subject.id, "slug": subject.slug, "title": subject.title,
        "metric": subject.metric, "direction": subject.direction, "runner": subject.runner,
        "description": subject.description, "config": subject.config,
    })
    return 0


def cmd_prompt(args: argparse.Namespace) -> int:
    store = _store(args)
    subject = _subject(store, args.slug)
    text = " ".join(args.text)
    outcome = apply_prompt(store, subject, text)
    print(f"recorded {outcome['kind']} prompt ({outcome['prompt_id']})")
    for action in outcome["actions"]:
        for key, value in action.items():
            print(f"  {key}: {value}")
    if not outcome["actions"]:
        print("  stored as context for the next planning step")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _config(args)
    store = Store(cfg.db_path)
    subject = _subject(store, args.slug)
    orch = Orchestrator(
        store, subject, cfg, salt=args.salt or subject.config.get("salt") or subject.slug,
        search_web=not args.no_web,
    )
    if args.salt:
        subject.config["salt"] = args.salt
        store.save_subject(subject)

    def on_step(result) -> None:
        if not args.quiet:
            print(result.line(), flush=True)

    summary = orch.run(
        steps=args.steps,
        budget_seconds=args.minutes * 60 if args.minutes else None,
        on_step=on_step,
    )
    print(
        f"\n{summary.trials} trials in {summary.elapsed_s:.1f}s — "
        f"{len(summary.proven)} proven, {len(summary.refuted)} refuted"
    )
    _print_json(orch.state())
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    cfg = _config(args)
    store = Store(cfg.db_path)
    subject = _subject(store, args.slug)
    orch = Orchestrator(store, subject, cfg, search_web=False)
    state = orch.state()
    _print_json(state)

    challenges = store.list_challenges(subject_id=subject.id, status="open")
    if challenges:
        print("\nopen challenges:")
        for challenge in challenges:
            print(f"  {challenge.id} on {challenge.finding_id}: {challenge.reason[:100]}")
    return 0


def cmd_findings(args: argparse.Namespace) -> int:
    store = _store(args)
    subject = _subject(store, args.slug)
    verdicts = args.verdict or None
    findings = store.list_findings(subject.id, verdicts)
    if not findings:
        print("no findings yet")
        return 0
    for finding in findings:
        hyp = store.get_hypothesis(finding.hypothesis_id)
        title = hyp.title if hyp else finding.hypothesis_id
        print(
            f"{finding.id}  {finding.verdict:12s} {finding.effect:+8.3%} "
            f"[{finding.ci_low:+7.3%}, {finding.ci_high:+7.3%}] n={finding.n_trials:3d}  {title[:60]}"
        )
        if args.verbose:
            print(f"    {finding.reason}")
    return 0


def cmd_challenge(args: argparse.Namespace) -> int:
    store = _store(args)
    subject = _subject(store, args.slug)
    finding = store.get_finding(args.finding) if args.finding else None
    if finding is None:
        candidates = store.list_findings(subject.id, [Verdict.PROVEN, Verdict.SUPPORTED])
        if not candidates:
            raise SystemExit("nothing to challenge yet")
        finding = candidates[0]
    axis: Dict[str, List[Any]] = {}
    for spec in args.axis or []:
        if "=" not in spec:
            raise SystemExit(f"--axis expects name=label1,label2 (got {spec!r})")
        name, labels = spec.split("=", 1)
        axis[name.strip()] = [l.strip() for l in labels.split(",") if l.strip()]
    challenge = open_challenge(store, subject, finding, args.reason, axis or None)
    print(f"opened challenge {challenge.id} against {finding.id}")
    print(f"  finding is now {Verdict.CONTESTED}; the bar tightens and the stress grid is re-run")
    print(f"  extra axis: {challenge.axis}")
    refreshed = store.get_finding(finding.id)
    for note in ((refreshed.evidence if refreshed else {}).get("challenge_notes") or [])[-1:]:
        print(f"  baseline: {note}")
    print(f"  run `arp run {subject.slug}` to settle it")
    return 0


def cmd_leads(args: argparse.Namespace) -> int:
    store = _store(args)
    subject = _subject(store, args.slug)
    if args.refresh:
        provider = get_provider(args.provider)
        if not provider.available:
            print("no search provider configured (set ANTHROPIC_API_KEY, BRAVE_API_KEY or TAVILY_API_KEY)")
        else:
            found = gather_leads(store, subject, provider=provider)
            print(f"gathered {len(found)} leads via {provider.name}")
    leads = store.list_leads(subject.id)
    for lead in leads[: args.limit]:
        s = lead.scores or {}
        print(
            f"[rel {s.get('relevance', 0):.2f} auth {s.get('authority', 0):.2f} "
            f"comm {s.get('commercial', 0):.2f}] {lead.title[:70]}\n    {lead.url}"
        )
    if not leads:
        print("no leads stored")
    return 0


def cmd_impact(args: argparse.Namespace) -> int:
    store = _store(args)
    subject = _subject(store, args.slug)
    ranked = rank_opportunities(store, subject)
    if not ranked:
        print("nothing scored yet — no supported or proven findings")
        return 0
    for finding, score in ranked:
        hyp = store.get_hypothesis(finding.hypothesis_id)
        title = hyp.title if hyp else finding.hypothesis_id
        money = f"  ~{score.monetary:,.0f} {score.currency}/yr" if score.monetary else ""
        print(f"{score.total:6.1f}  {finding.verdict:10s} {title[:60]}{money}")
        print(f"        {score.rationale}")
    return 0


def cmd_suggest(args: argparse.Namespace) -> int:
    store = _store(args)
    subject = store.get_subject(args.slug) if args.slug else None
    ideas = suggest(store, subject, limit=args.limit)
    if args.polish:
        ideas = polish(ideas)
    if not ideas:
        print("nothing to suggest yet — run some experiments or send some prompts first")
        return 0
    for idea in ideas:
        print(idea.line())
        if idea.seed_prompt and idea.subject_slug:
            print(f'         arp prompt {idea.subject_slug} "{idea.seed_prompt}"')
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    cfg = _config(args)
    store = Store(cfg.db_path)
    subject = _subject(store, args.slug)
    written = write_report(store, subject, args.md, args.html, cfg)
    if args.md or args.html:
        for kind, path in written.items():
            if kind != "text":
                print(f"wrote {kind}: {path}")
    else:
        print(written["text"])
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """End-to-end offline demonstration: propose, screen, confirm, challenge, prove."""
    cfg = _config(args)
    cfg.db_path = args.db or os.path.join(cfg.work_dir, "demo.db")
    store = Store(cfg.db_path)
    subject = store.get_subject("demo-optimizer")
    if subject is None:
        subject = store.add_subject(demo_subject())
        print(f"created demo subject {subject.slug}")

    orch = Orchestrator(store, subject, cfg, salt="demo", search_web=False)
    print(f"running {args.steps} steps against a simulated response surface...\n")
    summary = orch.run(steps=args.steps, on_step=lambda r: print(r.line(), flush=True))
    print(f"\n{summary.trials} trials in {summary.elapsed_s:.1f}s")
    print("\n--- findings ---")
    for finding in store.list_findings(subject.id):
        hyp = store.get_hypothesis(finding.hypothesis_id)
        print(f"{finding.verdict:12s} {finding.effect:+8.3%}  {(hyp.title if hyp else '')[:60]}")
    print("\n--- what the platform learned ---")
    profile = CapabilityMemory(store).skill_profile(subject)
    print(f"capability level {profile.level} over {profile.trials} trials, {profile.proven} proven")
    for row in profile.operators:
        print(f"  {row['operator']:20s} n={row['n']:3d} hit={row['success_rate']:.2f} "
              f"gain={row['mean_reward']:+.4%}")
    print(f"\ndatabase: {cfg.db_path}")
    print(f"try:  arp --db {cfg.db_path} report demo-optimizer")
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arp", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"arp {__version__}")
    parser.add_argument("--db", help="path to the platform database")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="create the platform database")
    p.set_defaults(func=cmd_init)

    p_subject = sub.add_parser("subject", help="manage research subjects")
    subject_sub = p_subject.add_subparsers(dest="subject_command", required=True)

    p = subject_sub.add_parser("add", help="create a subject")
    p.add_argument("--slug")
    p.add_argument("--title")
    p.add_argument("--metric", default="score")
    p.add_argument("--direction", default="minimize", choices=["minimize", "maximize"])
    p.add_argument("--runner", default="simulated", help="simulated | command | train")
    p.add_argument("--description", default="")
    p.add_argument("--preset", help="autoresearch | demo")
    p.add_argument("--config", help="JSON file merged into the subject config")
    p.set_defaults(func=cmd_subject_add)

    p = subject_sub.add_parser("list", help="list subjects")
    p.set_defaults(func=cmd_subject_list)

    p = subject_sub.add_parser("show", help="show a subject's full config")
    p.add_argument("slug")
    p.set_defaults(func=cmd_subject_show)

    p = sub.add_parser("prompt", help="send a human prompt to a subject")
    p.add_argument("slug")
    p.add_argument("text", nargs="+")
    p.set_defaults(func=cmd_prompt)

    p = sub.add_parser("run", help="run the research loop")
    p.add_argument("slug")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--minutes", type=float, default=None, help="wall-clock budget")
    p.add_argument("--salt", help="replay a previous run deterministically")
    p.add_argument("--no-web", action="store_true", help="skip web search agents")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("status", help="show a subject's current state")
    p.add_argument("slug")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("findings", help="list findings")
    p.add_argument("slug")
    p.add_argument("--verdict", action="append", help="filter (repeatable)")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_findings)

    p = sub.add_parser("challenge", help="second-guess a finding; forces re-testing")
    p.add_argument("slug")
    p.add_argument("--finding", help="finding id (default: the strongest current claim)")
    p.add_argument("--reason", required=True)
    p.add_argument("--axis", action="append", help="extra stress axis, name=label1,label2")
    p.set_defaults(func=cmd_challenge)

    p = sub.add_parser("leads", help="web leads for a subject")
    p.add_argument("slug")
    p.add_argument("--refresh", action="store_true", help="search the web now")
    p.add_argument("--provider", help="anthropic | brave | tavily")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_leads)

    p = sub.add_parser("impact", help="rank findings by business impact")
    p.add_argument("slug")
    p.set_defaults(func=cmd_impact)

    p = sub.add_parser("suggest", help="suggest what to research next")
    p.add_argument("slug", nargs="?")
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--polish", action="store_true", help="let the LLM rewrite the phrasing")
    p.set_defaults(func=cmd_suggest)

    p = sub.add_parser("report", help="write a report")
    p.add_argument("slug")
    p.add_argument("--md", help="write Markdown here")
    p.add_argument("--html", help="write HTML here")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("demo", help="offline end-to-end demonstration")
    p.add_argument("--steps", type=int, default=60)
    p.set_defaults(func=cmd_demo)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:
        # `arp findings ... | head` closes the pipe early; that is not an error.
        try:
            sys.stdout.close()
        finally:
            return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
