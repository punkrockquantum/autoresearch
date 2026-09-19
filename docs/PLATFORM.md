# `arp` — the agentic research platform

`train.py` + `program.md` is one agent, one metric, one loop, and a human reading
a TSV in the morning. This is the same idea generalised: **any subject, many
hypotheses at once, a decision procedure instead of a judgement call, a memory
that carries across subjects, and a human who can interrupt with a prompt at any
point.**

It is standard library only. No GPU, no network and no new dependencies are
needed to run it, develop it, or test it — `uv sync` is still only for `train.py`.

---

## The shape of it

```
          human prompts ────────────────┐
                                        ▼
  subject ──► proposer ──► bandit ──► runner ──► trials ──► decision ──► verdict
                 ▲          (Thompson)  (train.py /          (anytime-    │
                 │                        shell /             valid CS     │
        capability memory                 simulated)          + SPRT)      │
                 ▲                                                         │
                 └───────────── outcomes ◄── exhaustive grid ◄─────────────┘
                                                  │
                                    challenges ───┘        proven ──► web search
                                    (human or self)                   ──► impact
```

Eight modules do the work:

| module | responsibility |
|---|---|
| `arp/stats.py` | Welch's t-test, anytime-valid confidence sequences, Wald's SPRT, Holm-Bonferroni, and `decide()` — the one function that issues a screening verdict |
| `arp/exhaustive.py` | the confirmation grid: stress cells, replicate planning, and the four conditions for PROVEN |
| `arp/search.py` | Thompson sampling over hypotheses; deterministic seed derivation |
| `arp/evolution.py` | operator catalogue, mutation, subject similarity, the capability memory |
| `arp/agents.py` | proposer, critic, prompt parsing — LLM-backed when a key is present, deterministic otherwise |
| `arp/interventions.py` | human prompts, challenges, roll-backs |
| `arp/websearch.py` + `arp/impact.py` | outside evidence, and the business-impact score built from it |
| `arp/orchestrator.py` | the loop that spends the budget on whatever matters most right now |

---

## Probabilistic search, deterministic results

The search is stochastic; the conclusions are not. Three mechanisms keep those
apart.

**1. Every random draw is reproducible.** A run has a salt. The allocator's RNG
is seeded from it, and each trial's seed is `sha256(salt, subject, what-this-
trial-tests, replicate)` — derived from the *content* of the experiment, not from
a database id. `arp run <subject> --salt s` twice gives the same trials in the
same order.

**2. Verdicts are arithmetic, not opinion.** `decide()` is a pure function of
(baseline values, treatment values, direction, policy). No agent, no LLM and no
heuristic can promote a finding. The checks run in a fixed order:

1. below the replicate floor → keep testing
2. anytime-valid lower bound ≥ MDE → **supported**
3. anytime-valid upper bound ≤ −MDE → **refuted**
4. whole interval inside the region of practical equivalence → **inconclusive**
5. SPRT boundary crossed (and the interval agrees) → supported / inconclusive
6. replicate budget spent → refuted if it can no longer reach the MDE, else inconclusive
7. otherwise → keep testing

The interval is a normal-mixture confidence sequence (Howard et al. 2021): valid
at every sample size *simultaneously*, which is what makes it safe for a bandit
that looks at the numbers after every single trial. Because σ is estimated from a
handful of runs rather than known, the radius is inflated by the ratio of the
t critical value to the normal one; `tests/test_stats.py` checks empirically that
coverage holds under continuous peeking.

**3. "Proven" means exhaustively tested.** Clearing the screening stage only gets
a hypothesis into the confirmation stage, where it is re-run across a finite,
enumerable grid of stress cells declared by the subject (scale, data slice,
whatever matters), with a *fresh baseline measured inside every cell*. PROVEN
requires all four of:

1. every cell filled to the replicate floor, on both arms,
2. no cell regressing beyond tolerance,
3. the pooled effect clearing the MDE at the stricter confirmation alpha,
4. no open challenges.

Cells are pooled *stratified* — rescaled to the grand baseline mean — because
cells sit at different metric levels by construction, and charging that spread to
the treatment as noise would hide real effects.

A cell whose interval still spans the tolerance is **unresolved**, not failed:
the platform buys more replicates in that cell, up to the budget, and only then
calls it inconclusive. Refutation is reserved for an actual measured regression.

---

## Learning and evolving

Each operator family (`tune:lr`, `tune:architecture`, `combine`, `revert`,
`simplify`, …) carries a Beta posterior over "produces a real win" plus running
reward moments, **per subject**. The operator catalogue is derived from the
subject's declared parameter space, so the same machinery works on training
hyperparameters, serving costs, or anything else with knobs and a metric.

Those posteriors do three things:

- they bias the proposer's choice of what to try next (Thompson sampling),
- they prior the bandit's belief about a brand-new hypothesis,
- they transfer. A new subject is warm-started from the capability tables of
  subjects it resembles (Jaccard overlap of title, description, metric and
  parameter names), discounted so inherited evidence never counts as first-hand.

`arp status` reports a capability level: evidence gathered × hit rate + proven
findings. It is a blunt number, and it is honest about which knobs have paid off
on this subject.

**Epochs.** When a hypothesis is proven, its change is adopted as the new
baseline and the subject's epoch increments — the platform's equivalent of
committing in the manual loop. The epoch is part of every confirmation cell key,
so evidence gathered against an old baseline is automatically invalidated rather
than silently mixed in. Open proposals that touch the same parameters are retired
as superseded: they were questions about a baseline that no longer exists.

---

## Human intervention

Everything a person types is stored, and most of it does something.

```bash
arp prompt <subject> "focus on the schedule, depth is a dead end"   # steer
arp prompt <subject> "try DEPTH=12"                                 # becomes a real hypothesis
arp prompt <subject> "never exceed 40GB of VRAM"                    # recorded constraint
arp challenge <subject> --finding fnd_abc --reason "smells like a lucky baseline"
```

A **challenge** is the important one, and it is deliberately expensive for the
claim and cheap for the human:

1. the finding drops to CONTESTED and loses its `proven_at`,
2. the decision policy tightens — alpha shrinks by a factor per open challenge,
   the replicate floor rises,
3. a fresh axis is bolted onto the confirmation grid, so *every* cell key changes
   and the whole grid is re-measured from scratch,
4. if the change had already been adopted into the baseline, it is **rolled back
   out of it first** — otherwise the re-test would be comparing the change
   against itself. (If a later adoption has since taken over those parameters,
   the platform says so and declines to roll back, rather than quietly undoing
   the newer result.)

The challenge closes only on a terminal verdict — proven *or* refuted. The human
asked a question; both answers count as an answer.

The platform also challenges itself. On promotion out of screening, the critic
looks for reasons the result might be an artefact (effect smaller than run-to-run
noise, a parameter sitting exactly on a declared bound, no stress axes at all).
High-severity concerns become real challenges before anything is called proven.

---

## Web search and business impact

When a finding is proven, the search agents go and look at what the outside world
is doing with the same idea. Providers are tried in order — Anthropic's
server-side web search, Brave, Tavily — and with no keys configured the platform
simply skips the step and scores findings on its own evidence.

Hits are scored on relevance, source authority, commercial language and recency,
and stored as leads. The impact score is an explicit formula over five measured
inputs:

```
impact = 100 × confidence × magnitude × (0.5 + 0.5 durability) × (0.4 + 0.6 demand) × effort
```

- **confidence** — where the finding sits on the verdict ladder (refuted scores 0
  by construction: unproven things cannot be sold)
- **magnitude** — the *conservative* end of the interval against the subject's
  reference effect, not the point estimate
- **durability** — fraction of stress cells survived
- **demand** — authority-weighted market signal from the leads
- **effort** — declared cost of shipping it

If the subject declares a `value_model`, the score is also converted to money.
Two runs over the same evidence rank identically; nothing here is a model's
opinion.

---

## Subjects

A subject is a JSON-ish config:

```json
{
  "param_space": {
    "MATRIX_LR": {"type": "float", "default": 0.04, "low": 0.005, "high": 0.2,
                  "log": true, "group": "lr", "step": 0.25},
    "DEPTH":     {"type": "int", "default": 8, "low": 4, "high": 16, "step": 1,
                  "group": "architecture"},
    "WINDOW_PATTERN": {"type": "choice", "default": "SSSL",
                       "values": ["SSSL", "SSLL", "L"], "group": "attention"}
  },
  "baseline_params": {"MATRIX_LR": 0.04, "DEPTH": 8},
  "stress_axes": {"scale": {"nominal": {}, "smaller": {"DEPTH": 6}, "larger": {"DEPTH": 10}}},
  "runner_config": {"command": "uv run train.py", "metric_key": "val_bpb", "timeout_s": 900},
  "value_model": {"reference_effect": 0.01, "effort_days": 5, "value_per_relative_point": 250000}
}
```

`group` defines the operator families. `stress_axes` defines the exhaustive grid;
a label maps to parameter overrides (or to `{}` for a pure replication axis).

Three runners ship:

- **`train`** — patches the constants in this repo's `train.py`, repoints
  `manual_seed` at the trial's seed, runs `uv run`, and reads `val_bpb` back.
  `train.py` itself is never modified; each trial writes and deletes its own
  patched copy next to it so `from prepare import …` still resolves.
- **`command`** — any shell command. Parameters arrive as `{placeholders}` and as
  `ARP_PARAM_*` environment variables; any `key: value` line in stdout is parsed.
  This is how you point the platform at a subject it has never seen.
- **`simulated`** — a seeded synthetic response surface, for tests and `arp demo`.

---

## Command line

```bash
python3 -m arp init
python3 -m arp subject add --preset autoresearch     # the nanochat val_bpb subject
python3 -m arp subject add --preset demo             # offline synthetic subject
python3 -m arp run <subject> --steps 100 [--minutes 60] [--salt s] [--no-web]
python3 -m arp status <subject>
python3 -m arp findings <subject> [--verdict proven] [--verbose]
python3 -m arp prompt <subject> "…"
python3 -m arp challenge <subject> --finding <id> --reason "…" [--axis name=a,b]
python3 -m arp leads <subject> [--refresh]
python3 -m arp impact <subject>
python3 -m arp suggest [<subject>] [--polish]
python3 -m arp report <subject> [--md out.md] [--html out.html]
python3 -m arp demo --steps 150
```

State lives in one SQLite file (`--db`, or `$ARP_DB`, default
`~/.cache/autoresearch/arp/platform.db`). Runs are resumable: stop the loop
whenever, start it again, it picks up the pending confirmation trials.

---

## Configuration

| variable | meaning |
|---|---|
| `ARP_HOME` | base directory for the database and work files |
| `ARP_DB` | database path |
| `ANTHROPIC_API_KEY` | enables the LLM proposer/critic and Anthropic web search |
| `ARP_MODEL` | model id for those agents (default `claude-sonnet-5`) |
| `BRAVE_API_KEY` / `TAVILY_API_KEY` | alternative search providers |

Every one of these is optional. With none of them set the platform runs fully
offline on its deterministic paths — which is also how the test suite runs it.

---

## Limits worth knowing

- **Cost is real.** A confirmation grid is `cells × arms × replicates` runs. With
  the autoresearch subject's 3 stress cells and 3 replicates that is 18 five-minute
  trainings per proven finding — an hour and a half of GPU time to turn "looks
  good" into "holds up". That is the price of the word *proven*; lower
  `min_replicates` or trim `stress_axes` if you want the cheaper claim.
- **The MDE is a choice, not a discovery.** `DecisionPolicy.mde` defaults to 0.2%
  relative. Effects smaller than that are deliberately not worth the compute, and
  the platform will call them inconclusive forever if you let it.
- **Stress axes are only as good as you declare them.** The platform can only
  prove robustness along axes someone thought to write down; the critic flags a
  subject with none, but it cannot invent the right ones.
- **The capability memory can be wrong in a useful way.** Transfer between
  subjects is keyed on vocabulary overlap, which is a cheap proxy. It biases
  exploration order, never verdicts, so the worst case is wasted compute rather
  than a false result.
- **Web leads are evidence about the world, not about your result.** They move
  the impact score's `demand` term only; they can never make an unproven finding
  look proven.
