"""Reports: what the platform believes, why, and what it is worth.

Markdown is the source of truth; the HTML writer is a small self-contained
renderer so a report can be opened in a browser or mailed to someone without
pulling in a Markdown dependency.
"""

import html
import re
from typing import Any, Dict, List, Optional

from arp.evolution import CapabilityMemory
from arp.exhaustive import family_wise_filter
from arp.impact import business_brief, rank_opportunities
from arp.models import Subject, Verdict
from arp.store import Store
from arp.suggest import suggest


def subject_report(store: Store, subject: Subject, config=None, max_suggestions: int = 5) -> str:
    from arp.config import default_config

    config = config or default_config()
    memory = CapabilityMemory(store)
    skill = memory.skill_profile(subject)
    findings = store.list_findings(subject.id)
    by_verdict: Dict[str, List] = {}
    for f in findings:
        by_verdict.setdefault(f.verdict, []).append(f)
    survives = family_wise_filter(store, subject, config.decision)

    lines: List[str] = [
        f"# {subject.title}",
        "",
        f"`{subject.slug}` — metric **{subject.metric}** ({subject.direction}), "
        f"runner `{subject.runner}`, epoch {subject.config.get('epoch', 0)}",
        "",
        subject.description or "",
        "",
        "## Where things stand",
        "",
        f"- Trials run: **{store.count_trials(subject.id)}**",
        f"- Hypotheses: **{len(store.list_hypotheses(subject.id))}** "
        f"({len(store.open_hypotheses(subject.id))} open)",
        f"- Proven: **{len(by_verdict.get(Verdict.PROVEN, []))}** · "
        f"supported {len(by_verdict.get(Verdict.SUPPORTED, []))} · "
        f"contested {len(by_verdict.get(Verdict.CONTESTED, []))} · "
        f"refuted {len(by_verdict.get(Verdict.REFUTED, []))} · "
        f"inconclusive {len(by_verdict.get(Verdict.INCONCLUSIVE, []))}",
        f"- Open challenges: **{len(store.list_challenges(subject_id=subject.id, status='open'))}**",
        f"- Capability level: **{skill.level}** "
        f"(best operators: {', '.join(skill.top_operators()) or 'none yet'})",
        "",
    ]

    baseline = subject.baseline_params
    if baseline:
        lines += ["### Current baseline", "", "```", *(f"{k} = {v}" for k, v in sorted(baseline.items())), "```", ""]

    adopted = subject.config.get("adopted") or []
    if adopted:
        lines += ["### Adopted so far", ""]
        for entry in adopted:
            suffix = ""
            if entry.get("reverted_at"):
                suffix = f" — **rolled back** after a challenge on {entry['reverted_at']}"
            lines.append(
                f"- {entry.get('title')} ({entry.get('effect', 0):+.3%}) — {entry.get('at')}{suffix}"
            )
        lines.append("")

    # -- findings ----------------------------------------------------------
    for verdict, heading in (
        (Verdict.PROVEN, "Proven"),
        (Verdict.CONTESTED, "Contested (re-testing under a stricter bar)"),
        (Verdict.SUPPORTED, "Supported (exhaustive stage in progress)"),
        (Verdict.REFUTED, "Refuted"),
        (Verdict.INCONCLUSIVE, "Inconclusive"),
    ):
        group = by_verdict.get(verdict) or []
        if not group:
            continue
        lines += [f"## {heading}", ""]
        for finding in group:
            hyp = store.get_hypothesis(finding.hypothesis_id)
            title = hyp.title if hyp else finding.hypothesis_id
            lines.append(f"### {title}")
            lines.append("")
            lines.append(
                f"- Effect: **{finding.effect:+.4%}** on {subject.metric} "
                f"(interval [{finding.ci_low:+.4%}, {finding.ci_high:+.4%}], "
                f"p={finding.p_value:.4g}, n={finding.n_trials})"
            )
            lines.append(f"- Verdict rationale: {finding.reason}")
            if finding.id in survives:
                lines.append(
                    f"- Survives Holm-Bonferroni across this subject's findings: "
                    f"**{'yes' if survives[finding.id] else 'no'}**"
                )
            if hyp and hyp.rationale:
                lines.append(f"- Hypothesis rationale: {hyp.rationale}")
            confirmation = (finding.evidence or {}).get("confirmation") or {}
            cells = confirmation.get("cells") or []
            if cells:
                lines += ["", "| stress cell | effect | interval | n | passed |", "|---|---|---|---|---|"]
                for cell in cells:
                    lines.append(
                        f"| `{cell.get('cell')}` | {cell.get('effect', 0):+.4%} | "
                        f"[{cell.get('ci_low', 0):+.4%}, {cell.get('ci_high', 0):+.4%}] | "
                        f"{cell.get('n_treatment', 0)}v{cell.get('n_baseline', 0)} | "
                        f"{'yes' if cell.get('passed') else 'no'} |"
                    )
            critiques = (finding.evidence or {}).get("critiques") or []
            if critiques:
                lines += ["", "**Concerns raised against this result:**", ""]
                for note in critiques:
                    lines.append(f"- [{note.get('severity', '?')}] {note.get('concern')}")
            challenges = store.list_challenges(finding_id=finding.id)
            if challenges:
                lines += ["", "**Human challenges:**", ""]
                for challenge in challenges:
                    state = challenge.status
                    lines.append(
                        f"- [{state}] {challenge.reason[:200]}"
                        + (f" → {challenge.resolution}" if challenge.resolution else "")
                    )
            lines.append("")

    # -- opportunities -----------------------------------------------------
    opportunities = rank_opportunities(store, subject)
    if opportunities:
        lines += ["## Business impact", ""]
        for finding, score in opportunities[:8]:
            hyp = store.get_hypothesis(finding.hypothesis_id)
            leads = store.list_leads(subject.id, finding.id)
            lines += [business_brief(subject, finding, score, hyp, leads), ""]

    # -- capability --------------------------------------------------------
    if skill.operators:
        lines += ["## What the platform has learned", "",
                  "| operator | outcomes | hit rate | mean gain/trial |", "|---|---|---|---|"]
        for row in skill.operators:
            lines.append(
                f"| {row['operator']} | {row['n']} | {row['success_rate']:.2f} | "
                f"{row['mean_reward']:+.4%} |"
            )
        lines.append("")

    # -- suggestions -------------------------------------------------------
    ideas = suggest(store, subject, limit=max_suggestions)
    if ideas:
        lines += ["## What to look at next", ""]
        for idea in ideas:
            lines.append(f"- **{idea.title}** ({idea.kind}, score {idea.score:.1f}) — {idea.rationale}")
            if idea.seed_prompt:
                lines.append(f"  - prompt: `arp prompt {subject.slug} \"{idea.seed_prompt}\"`")
        lines.append("")

    # -- conversation ------------------------------------------------------
    prompts = store.list_prompts(subject.id, limit=12)
    if prompts:
        lines += ["## Recent conversation", ""]
        for prompt in reversed(prompts):
            lines.append(f"- `{prompt.created_at}` **{prompt.role}/{prompt.kind}**: {prompt.text[:300]}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Minimal Markdown -> HTML
# ---------------------------------------------------------------------------

def markdown_to_html(md: str, title: str = "arp report") -> str:
    """Render the subset of Markdown this module emits. No dependencies."""
    body: List[str] = []
    in_code = False
    in_list = False
    in_table = False

    def close_blocks() -> None:
        nonlocal in_list, in_table
        if in_list:
            body.append("</ul>")
            in_list = False
        if in_table:
            body.append("</table>")
            in_table = False

    for raw in md.splitlines():
        line = raw.rstrip()
        if line.startswith("```"):
            close_blocks()
            body.append("</pre>" if in_code else "<pre>")
            in_code = not in_code
            continue
        if in_code:
            body.append(html.escape(line))
            continue
        if not line.strip():
            close_blocks()
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            close_blocks()
            level = len(heading.group(1))
            body.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
            continue

        if line.lstrip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue  # separator row
            if not in_table:
                close_blocks()
                body.append("<table>")
                in_table = True
            tag = "th" if len(body) and body[-1] == "<table>" else "td"
            body.append("<tr>" + "".join(f"<{tag}>{_inline(c)}</{tag}>" for c in cells) + "</tr>")
            continue

        bullet = re.match(r"^(\s*)[-*]\s+(.*)$", line)
        if bullet:
            if not in_list:
                body.append("<ul>")
                in_list = True
            body.append(f"<li>{_inline(bullet.group(2))}</li>")
            continue

        close_blocks()
        body.append(f"<p>{_inline(line)}</p>")

    close_blocks()
    if in_code:
        body.append("</pre>")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  :root {{ color-scheme: light dark; --fg: #16181d; --bg: #ffffff; --muted: #5a6270;
           --line: #e2e5ea; --accent: #2f6f4f; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --fg: #e7e9ee; --bg: #14161a; --muted: #9aa3b2; --line: #2a2e36; --accent: #7fd1a6; }}
  }}
  body {{ margin: 0 auto; padding: 32px 16px 96px; max-width: 860px; background: var(--bg);
          color: var(--fg); font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
  h1 {{ font-size: 1.9rem; margin-bottom: .2em; }}
  h2 {{ margin-top: 2em; border-bottom: 1px solid var(--line); padding-bottom: .25em; }}
  h3 {{ margin-top: 1.6em; color: var(--accent); }}
  code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .88em; }}
  pre {{ background: color-mix(in srgb, var(--fg) 6%, transparent); padding: 12px 14px;
         border-radius: 8px; overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: .92em; }}
  th, td {{ border: 1px solid var(--line); padding: 6px 10px; text-align: left; }}
  th {{ background: color-mix(in srgb, var(--fg) 5%, transparent); }}
  a {{ color: var(--accent); }}
  li {{ margin: .2em 0; }}
  p {{ margin: .6em 0; }}
</style>
</head>
<body>
{chr(10).join(body)}
</body>
</html>
"""


def _inline(text: str) -> str:
    out = html.escape(text)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", out)
    out = re.sub(r"_([^_]+)_", r"<em>\1</em>", out)
    out = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', out)
    return out


def write_report(
    store: Store, subject: Subject, md_path: Optional[str] = None,
    html_path: Optional[str] = None, config=None,
) -> Dict[str, Any]:
    md = subject_report(store, subject, config)
    written = {}
    if md_path:
        with open(md_path, "w") as f:
            f.write(md)
        written["markdown"] = md_path
    if html_path:
        with open(html_path, "w") as f:
            f.write(markdown_to_html(md, title=subject.title))
        written["html"] = html_path
    written["text"] = md
    return written
