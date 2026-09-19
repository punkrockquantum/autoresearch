"""SQLite persistence. One file, no ORM, no migrations beyond `CREATE IF NOT EXISTS`.

The store is the platform's memory: subjects, the prompt/chat log, hypotheses,
every trial ever run, findings and their evidence, human challenges, the evolved
capability table, and web leads. Anything the orchestrator needs to resume after
a crash lives here.
"""

import os
import sqlite3
from typing import Any, Dict, Iterable, List, Optional

from arp.models import (
    Capability,
    Challenge,
    Finding,
    Hypothesis,
    Lead,
    Prompt,
    Run,
    Subject,
    Trial,
    Verdict,
    now_iso,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS subjects (
    id TEXT PRIMARY KEY,
    slug TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    metric TEXT NOT NULL,
    direction TEXT NOT NULL,
    description TEXT,
    runner TEXT,
    status TEXT,
    config TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS prompts (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    role TEXT,
    kind TEXT,
    text TEXT,
    meta TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_prompts_subject ON prompts(subject_id);

CREATE TABLE IF NOT EXISTS hypotheses (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    title TEXT,
    operator TEXT,
    params TEXT,
    rationale TEXT,
    origin TEXT,
    parent_id TEXT,
    status TEXT,
    meta TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_hyp_subject ON hypotheses(subject_id, status);

CREATE TABLE IF NOT EXISTS trials (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    hypothesis_id TEXT,
    arm TEXT,
    stage TEXT,
    seed INTEGER,
    metric_value REAL,
    ok INTEGER,
    params TEXT,
    cell TEXT,
    aux TEXT,
    error TEXT,
    duration_s REAL,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_trials_hyp ON trials(hypothesis_id, stage);
CREATE INDEX IF NOT EXISTS idx_trials_subject ON trials(subject_id, arm);

CREATE TABLE IF NOT EXISTS findings (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    hypothesis_id TEXT NOT NULL UNIQUE,
    verdict TEXT,
    effect REAL,
    ci_low REAL,
    ci_high REAL,
    p_value REAL,
    n_trials INTEGER,
    reason TEXT,
    evidence TEXT,
    impact TEXT,
    proven_at TEXT,
    created_at TEXT,
    updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_findings_subject ON findings(subject_id, verdict);

CREATE TABLE IF NOT EXISTS challenges (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    finding_id TEXT NOT NULL,
    reason TEXT,
    axis TEXT,
    status TEXT,
    resolution TEXT,
    created_at TEXT,
    resolved_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_challenges_finding ON challenges(finding_id, status);

CREATE TABLE IF NOT EXISTS capabilities (
    id TEXT PRIMARY KEY,
    subject_id TEXT,
    operator TEXT,
    alpha REAL,
    beta REAL,
    reward_sum REAL,
    reward_sq REAL,
    n INTEGER,
    updated_at TEXT,
    UNIQUE(subject_id, operator)
);

CREATE TABLE IF NOT EXISTS leads (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    finding_id TEXT,
    title TEXT,
    url TEXT,
    snippet TEXT,
    source TEXT,
    query TEXT,
    scores TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_leads_subject ON leads(subject_id);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    salt TEXT,
    steps_requested INTEGER,
    steps_done INTEGER,
    status TEXT,
    notes TEXT,
    started_at TEXT,
    ended_at TEXT
);
"""


class Store:
    def __init__(self, db_path: str):
        self.db_path = db_path
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # -- plumbing ----------------------------------------------------------

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _insert(self, table: str, row: Dict[str, Any]) -> None:
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        self.conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", list(row.values()))
        self.conn.commit()

    def _upsert(self, table: str, row: Dict[str, Any], key: str = "id") -> None:
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        updates = ", ".join(f"{c}=excluded.{c}" for c in row if c != key)
        self.conn.execute(
            f"INSERT INTO {table} ({cols}) VALUES ({marks}) "
            f"ON CONFLICT({key}) DO UPDATE SET {updates}",
            list(row.values()),
        )
        self.conn.commit()

    def _rows(self, sql: str, args: Iterable[Any] = ()) -> List[Dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(sql, tuple(args)).fetchall()]

    def _row(self, sql: str, args: Iterable[Any] = ()) -> Optional[Dict[str, Any]]:
        rows = self._rows(sql, args)
        return rows[0] if rows else None

    # -- subjects ----------------------------------------------------------

    def add_subject(self, subject: Subject) -> Subject:
        self._insert("subjects", subject.to_row())
        return subject

    def save_subject(self, subject: Subject) -> Subject:
        self._upsert("subjects", subject.to_row())
        return subject

    def get_subject(self, slug_or_id: str) -> Optional[Subject]:
        row = self._row(
            "SELECT * FROM subjects WHERE slug = ? OR id = ?", (slug_or_id, slug_or_id)
        )
        return Subject.from_row(row) if row else None

    def list_subjects(self, status: Optional[str] = None) -> List[Subject]:
        if status:
            rows = self._rows("SELECT * FROM subjects WHERE status = ? ORDER BY created_at", (status,))
        else:
            rows = self._rows("SELECT * FROM subjects ORDER BY created_at")
        return [Subject.from_row(r) for r in rows]

    # -- prompts / chat log ------------------------------------------------

    def add_prompt(self, prompt: Prompt) -> Prompt:
        self._insert("prompts", prompt.to_row())
        return prompt

    def list_prompts(self, subject_id: Optional[str] = None, limit: int = 500) -> List[Prompt]:
        if subject_id:
            rows = self._rows(
                "SELECT * FROM prompts WHERE subject_id = ? ORDER BY created_at DESC LIMIT ?",
                (subject_id, limit),
            )
        else:
            rows = self._rows("SELECT * FROM prompts ORDER BY created_at DESC LIMIT ?", (limit,))
        return [Prompt.from_row(r) for r in rows]

    # -- hypotheses --------------------------------------------------------

    def add_hypothesis(self, hyp: Hypothesis) -> Hypothesis:
        self._insert("hypotheses", hyp.to_row())
        return hyp

    def save_hypothesis(self, hyp: Hypothesis) -> Hypothesis:
        self._upsert("hypotheses", hyp.to_row())
        return hyp

    def get_hypothesis(self, hyp_id: str) -> Optional[Hypothesis]:
        row = self._row("SELECT * FROM hypotheses WHERE id = ?", (hyp_id,))
        return Hypothesis.from_row(row) if row else None

    def list_hypotheses(
        self, subject_id: str, statuses: Optional[Iterable[str]] = None
    ) -> List[Hypothesis]:
        if statuses:
            statuses = list(statuses)
            marks = ", ".join("?" for _ in statuses)
            rows = self._rows(
                f"SELECT * FROM hypotheses WHERE subject_id = ? AND status IN ({marks}) "
                "ORDER BY created_at",
                [subject_id, *statuses],
            )
        else:
            rows = self._rows(
                "SELECT * FROM hypotheses WHERE subject_id = ? ORDER BY created_at", (subject_id,)
            )
        return [Hypothesis.from_row(r) for r in rows]

    def open_hypotheses(self, subject_id: str) -> List[Hypothesis]:
        return self.list_hypotheses(subject_id, Verdict.OPEN)

    # -- trials ------------------------------------------------------------

    def add_trial(self, trial: Trial) -> Trial:
        self._insert("trials", trial.to_row())
        return trial

    def list_trials(
        self,
        subject_id: Optional[str] = None,
        hypothesis_id: Optional[str] = None,
        arm: Optional[str] = None,
        stage: Optional[str] = None,
        ok_only: bool = True,
    ) -> List[Trial]:
        sql = "SELECT * FROM trials WHERE 1=1"
        args: List[Any] = []
        if subject_id:
            sql += " AND subject_id = ?"
            args.append(subject_id)
        if hypothesis_id:
            sql += " AND hypothesis_id = ?"
            args.append(hypothesis_id)
        if arm:
            sql += " AND arm = ?"
            args.append(arm)
        if stage:
            sql += " AND stage = ?"
            args.append(stage)
        if ok_only:
            sql += " AND ok = 1"
        sql += " ORDER BY created_at"
        return [Trial.from_row(r) for r in self._rows(sql, args)]

    def baseline_values(self, subject_id: str, cell: Optional[Dict[str, Any]] = None) -> List[float]:
        """Baseline metric values, optionally restricted to one exhaustive-grid cell."""
        trials = self.list_trials(subject_id=subject_id, arm="baseline")
        values = []
        for t in trials:
            if t.metric_value is None:
                continue
            if cell is not None and t.cell != cell:
                continue
            values.append(t.metric_value)
        return values

    def count_trials(self, subject_id: str) -> int:
        row = self._row("SELECT COUNT(*) AS c FROM trials WHERE subject_id = ?", (subject_id,))
        return int(row["c"]) if row else 0

    # -- findings ----------------------------------------------------------

    def save_finding(self, finding: Finding) -> Finding:
        finding.updated_at = now_iso()
        self._upsert("findings", finding.to_row())
        return finding

    def get_finding(self, finding_id: str) -> Optional[Finding]:
        row = self._row("SELECT * FROM findings WHERE id = ?", (finding_id,))
        return Finding.from_row(row) if row else None

    def finding_for_hypothesis(self, hypothesis_id: str) -> Optional[Finding]:
        row = self._row("SELECT * FROM findings WHERE hypothesis_id = ?", (hypothesis_id,))
        return Finding.from_row(row) if row else None

    def list_findings(
        self, subject_id: Optional[str] = None, verdicts: Optional[Iterable[str]] = None
    ) -> List[Finding]:
        sql = "SELECT * FROM findings WHERE 1=1"
        args: List[Any] = []
        if subject_id:
            sql += " AND subject_id = ?"
            args.append(subject_id)
        if verdicts:
            verdicts = list(verdicts)
            marks = ", ".join("?" for _ in verdicts)
            sql += f" AND verdict IN ({marks})"
            args.extend(verdicts)
        sql += " ORDER BY effect DESC"
        return [Finding.from_row(r) for r in self._rows(sql, args)]

    # -- challenges --------------------------------------------------------

    def add_challenge(self, challenge: Challenge) -> Challenge:
        self._insert("challenges", challenge.to_row())
        return challenge

    def save_challenge(self, challenge: Challenge) -> Challenge:
        self._upsert("challenges", challenge.to_row())
        return challenge

    def list_challenges(
        self, finding_id: Optional[str] = None, subject_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Challenge]:
        sql = "SELECT * FROM challenges WHERE 1=1"
        args: List[Any] = []
        if finding_id:
            sql += " AND finding_id = ?"
            args.append(finding_id)
        if subject_id:
            sql += " AND subject_id = ?"
            args.append(subject_id)
        if status:
            sql += " AND status = ?"
            args.append(status)
        sql += " ORDER BY created_at"
        return [Challenge.from_row(r) for r in self._rows(sql, args)]

    def open_challenges(self, finding_id: str) -> List[Challenge]:
        return self.list_challenges(finding_id=finding_id, status="open")

    # -- capabilities ------------------------------------------------------

    def get_capability(self, operator: str, subject_id: str = "") -> Optional[Capability]:
        row = self._row(
            "SELECT * FROM capabilities WHERE operator = ? AND subject_id = ?",
            (operator, subject_id),
        )
        return Capability.from_row(row) if row else None

    def save_capability(self, cap: Capability) -> Capability:
        cap.updated_at = now_iso()
        row = cap.to_row()
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        updates = ", ".join(f"{c}=excluded.{c}" for c in row if c not in ("id",))
        self.conn.execute(
            f"INSERT INTO capabilities ({cols}) VALUES ({marks}) "
            f"ON CONFLICT(subject_id, operator) DO UPDATE SET {updates}",
            list(row.values()),
        )
        self.conn.commit()
        return cap

    def list_capabilities(self, subject_id: Optional[str] = None) -> List[Capability]:
        if subject_id is None:
            rows = self._rows("SELECT * FROM capabilities ORDER BY operator")
        else:
            rows = self._rows(
                "SELECT * FROM capabilities WHERE subject_id = ? ORDER BY operator", (subject_id,)
            )
        return [Capability.from_row(r) for r in rows]

    # -- leads -------------------------------------------------------------

    def add_lead(self, lead: Lead) -> Lead:
        existing = self._row(
            "SELECT id FROM leads WHERE subject_id = ? AND url = ?", (lead.subject_id, lead.url)
        )
        if existing:
            lead.id = existing["id"]
            self._upsert("leads", lead.to_row())
        else:
            self._insert("leads", lead.to_row())
        return lead

    def list_leads(self, subject_id: str, finding_id: Optional[str] = None) -> List[Lead]:
        if finding_id:
            rows = self._rows(
                "SELECT * FROM leads WHERE subject_id = ? AND finding_id = ? ORDER BY created_at DESC",
                (subject_id, finding_id),
            )
        else:
            rows = self._rows(
                "SELECT * FROM leads WHERE subject_id = ? ORDER BY created_at DESC", (subject_id,)
            )
        return [Lead.from_row(r) for r in rows]

    # -- runs --------------------------------------------------------------

    def add_run(self, run: Run) -> Run:
        self._insert("runs", run.to_row())
        return run

    def save_run(self, run: Run) -> Run:
        self._upsert("runs", run.to_row())
        return run

    def list_runs(self, subject_id: str, limit: int = 20) -> List[Run]:
        rows = self._rows(
            "SELECT * FROM runs WHERE subject_id = ? ORDER BY started_at DESC LIMIT ?",
            (subject_id, limit),
        )
        return [Run.from_row(r) for r in rows]


def open_store(db_path: Optional[str] = None) -> Store:
    from arp.config import DB_PATH

    return Store(db_path or DB_PATH)
