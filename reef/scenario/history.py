"""Read retained records and persisted commit metadata without interpreting learning outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from typing import Any

from reef.core.artifact_ref import encode_artifact_ref
from reef.scenario.scenario import Scenario
from reef.storage.commits import CommitRecord
from reef.storage.records import RecordReference, StoredRecord


@dataclass(frozen=True)
class LearningQuery:
    """Bounds for a read-only learning history page."""

    record_id: str | None = None
    commit_step: int | None = None
    after_sequence: int = 0
    after_step: int = 0
    input_offset: int = 0
    scan_after: int | None = None
    show_payload: bool = False

    def __post_init__(self) -> None:
        if self.record_id is not None and (not self.record_id or len(self.record_id) > 256):
            raise ValueError("record_id must contain between 1 and 256 characters")
        if self.commit_step is not None and (self.commit_step < 1 or self.record_id is not None):
            raise ValueError("commit_step must be positive and cannot be combined with record_id")
        if min(self.after_sequence, self.after_step, self.input_offset) < 0:
            raise ValueError("page cursors must be non-negative")
        if self.scan_after is not None and (self.scan_after < 0 or self.record_id is None):
            raise ValueError("scan_after requires record_id and a non-negative sequence")
        if self.show_payload and self.record_id is None:
            raise ValueError("show_payload requires record_id")


@dataclass(frozen=True)
class LearningPage:
    """A bounded view of stored facts, rendered after releasing the scenario lock."""

    scenario: str
    durable: bool
    records: tuple[StoredRecord, ...] = ()
    record: StoredRecord | None = None
    record_known: bool = False
    commits: tuple[CommitRecord, ...] = ()
    commit: CommitRecord | None = None
    release_index: int | None = None
    next_sequence: int | None = None
    next_step: int | None = None
    references: tuple[RecordReference, ...] | None = None
    next_scan: int | None = None
    scanned: int = 0


def read_learning_page(scenario: Scenario, query: LearningQuery) -> LearningPage:
    """Read only the selected page; reverse links scan at most 100 retained rows plus lookahead."""
    durable = scenario.store.durable
    history = scenario.store.history() if durable else ()
    if query.commit_step is not None:
        for index, commit in enumerate(history, start=1):
            if commit.step == query.commit_step:
                return LearningPage(
                    scenario=scenario.name,
                    durable=durable,
                    commit=commit,
                    release_index=index if scenario.surface.harness is not None else None,
                )
        return LearningPage(scenario=scenario.name, durable=durable)

    commits = tuple(
        islice(
            (
                commit
                for commit in history
                if commit.step > query.after_step
                and (query.record_id is None or query.record_id in commit.consumed_ids)
            ),
            21,
        )
    )
    next_step = commits[19].step if len(commits) > 20 else None
    if query.record_id is None:
        records = scenario.records.audit_page(scenario.name, after_sequence=query.after_sequence, limit=21)
        return LearningPage(
            scenario=scenario.name,
            durable=durable,
            records=records[:20],
            commits=commits[:20],
            next_sequence=records[19].sequence if len(records) > 20 else None,
            next_step=next_step,
        )

    record = scenario.records.get_for_audit(scenario.name, query.record_id)
    known = record is not None or any(query.record_id in commit.consumed_ids for commit in history)
    references = None
    next_scan = None
    scanned = 0
    if query.scan_after is not None:
        rows = scenario.records.reference_page(scenario.name, after_sequence=query.scan_after, limit=101)
        if rows is not None:
            scanned = len(rows[:100])
            references = tuple(row for row in rows[:100] if query.record_id in row.references)
            next_scan = rows[99].sequence if len(rows) > 100 else None
            known = known or bool(references)
    return LearningPage(
        scenario=scenario.name,
        durable=durable,
        record=record,
        record_known=known,
        commits=commits[:20],
        next_step=next_step,
        references=references,
        next_scan=next_scan,
        scanned=scanned,
    )


def record_metadata(stored: StoredRecord) -> dict[str, Any]:
    item = stored.item
    return {
        "sequence": stored.sequence,
        "agent_record_id": item.agent_record_id,
        "request_type": item.request_type.value,
        "created_at": item.created_at,
        "compacted_at": stored.compacted_at,
        "references": list(item.references),
        "artifact_ref": encode_artifact_ref(item.artifact_ref) if item.artifact_ref else None,
        "score": item.payload.get("score"),
    }


def read_records(scenario: Scenario, *, after_sequence: int, limit: int) -> dict[str, Any]:
    if after_sequence < 0 or not 1 <= limit <= 100:
        raise ValueError("after_sequence must be non-negative and limit must be between 1 and 100")
    retained = scenario.records.audit_page(scenario.name, after_sequence=after_sequence, limit=limit + 1)
    page = retained[:limit]
    return {
        "scenario": scenario.name,
        "records": [record_metadata(stored) for stored in page],
        "next_after_sequence": page[-1].sequence if len(retained) > limit else None,
    }


def read_record(scenario: Scenario, record_id: str) -> dict[str, Any] | None:
    stored = scenario.records.get_for_audit(scenario.name, record_id)
    return None if stored is None else {**record_metadata(stored), "payload": stored.item.payload}


def read_commits(scenario: Scenario, *, after_step: int, limit: int, record_ids: tuple[str, ...]) -> dict[str, Any]:
    """Filter committed consumed_ids by exact record IDs; preserve their stored meaning."""
    if after_step < 0 or not 1 <= limit <= 100:
        raise ValueError("after_step must be non-negative and limit must be between 1 and 100")
    if len(record_ids) > 100 or any(not value or len(value) > 256 for value in record_ids):
        raise ValueError("at most 100 non-empty record_id values of at most 256 characters are accepted")
    requested = frozenset(record_ids)
    commits = scenario.store.history() if scenario.store.durable else ()
    matches = tuple(
        islice(
            (
                commit
                for commit in commits
                if commit.step > after_step and (not requested or not requested.isdisjoint(commit.consumed_ids))
            ),
            limit + 1,
        )
    )
    return {
        "scenario": scenario.name,
        "commits": [
            {
                "step": commit.step,
                "operation": commit.operation,
                "operation_verified": commit.operation_verified,
                "recorded_at": commit.recorded_at,
                "artifact_ref": encode_artifact_ref(commit.artifact_ref),
                "pending": commit.pending,
                "consumed_ids": sorted(commit.consumed_ids),
                "metrics": commit.metrics,
            }
            for commit in matches[:limit]
        ],
        "next_after_step": matches[limit - 1].step if len(matches) > limit else None,
    }
