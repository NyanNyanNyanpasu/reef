"""Browse exact stored relationships without changing learning or retention."""

from __future__ import annotations

import asyncio
import html
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlencode

import pytest
from aiohttp.test_utils import TestClient, TestServer

from reef.core import AgentRecord, RequestType
from reef.dispatcher import Dispatcher, build_default_dispatcher
from reef.scenario.history import LearningPage, LearningQuery
from reef.service.app import create_app
from reef.service.learning_page import build_learning_page, learning_href
from reef.storage.commit_log import CommitLogScenarioStore
from reef.storage.commits import CommitRecord
from reef.storage.observer import ObservedRecordStore, RecordObserver
from reef.storage.records import RecordStore, StoredRecord
from reef.storage.sqlite import SQLiteScenarioStorage


@pytest.fixture
def history_dispatcher(tmp_path: Path) -> Iterator[Dispatcher]:
    dispatcher = build_default_dispatcher(
        agent_record_dir=tmp_path / "records", scenario_storage=SQLiteScenarioStorage(tmp_path / "records")
    )
    scenario = dispatcher.get_or_create_scenario("one")
    other = dispatcher.get_or_create_scenario("two")
    scenario.records.append(
        AgentRecord.create(
            scenario="one",
            request_type=RequestType.INFERENCE,
            agent_record_id="inference",
            payload={"text": "<script>private_trace()</script>"},
            artifact_ref=scenario.current_artifact_ref(),
        )
    )
    scenario.records.append(
        AgentRecord.create(
            scenario="one",
            request_type=RequestType.REPORT,
            agent_record_id="feedback",
            payload={"score": 0},
            references=("inference",),
        )
    )
    other.records.append(
        AgentRecord.create(
            scenario="two",
            request_type=RequestType.INFERENCE,
            agent_record_id="other-secret",
            payload={"text": "other-private-trace"},
        )
    )
    assert isinstance(scenario.store, CommitLogScenarioStore)
    log = scenario.store.commit_log
    assert log is not None
    log.append(
        CommitRecord(
            scenario="one",
            step=1,
            artifact_ref=scenario.current_artifact_ref(),
            checkpoint=False,
            algorithm_state={"private_state": "private-checkpoint-data"},
            high_water_sequence=2,
            high_water_offset=2,
            consumed_ids=frozenset({"feedback"}),
            compacted_ids=frozenset({"feedback", "inference"}),
            metrics={"selected": False, "selection": {"reason": "regressed"}},
        )
    )
    try:
        yield dispatcher
    finally:
        dispatcher.close()


def test_pages_link_only_recorded_relationships_and_survive_retention(history_dispatcher: Dispatcher) -> None:
    async def run() -> None:
        scenario = history_dispatcher.get_or_create_scenario("one")
        before = scenario.records.count("one")
        commits_before = scenario.store.history()
        async with TestClient(TestServer(create_app(history_dispatcher))) as client:
            root = "/reef/scenarios/one/learning/page"
            response = await client.get(root, headers={"x-reef-scenario": "two"})
            body = await response.text()
            assert response.status == 200
            assert response.headers["Cache-Control"] == "no-store"
            assert response.headers["Referrer-Policy"] == "no-referrer"
            assert "record_id=inference" in body and "commit_step=1" in body
            assert "private_trace" not in body and "other-secret" not in body

            body = await (await client.get(f"{root}?record_id=feedback")).text()
            assert "record_id=inference" in body and "commit_step=1" in body
            body = await (await client.get(f"{root}?record_id=inference")).text()
            assert "commit_step=1" not in body  # Retiring an input is not consuming it.
            assert "private_trace" not in body and "scan_after=0" in body
            body = await (await client.get(f"{root}?record_id=inference&show_payload=1")).text()
            assert "&lt;script&gt;private_trace()&lt;/script&gt;" in body
            assert "<script>private_trace" not in body
            body = await (await client.get(f"{root}?record_id=inference&scan_after=0")).text()
            assert "record_id=feedback" in body and "End of currently retained records" in body

            body = await (await client.get(f"{root}?commit_step=1")).text()
            assert "record_id=feedback" in body and "regressed" in body
            assert "private-checkpoint-data" not in body
            assert "/reef/harness/releases/" not in body  # A record-only scenario has no harness surface.
            assert scenario.records.count("one") == before
            assert scenario.store.history() == commits_before

            scenario.records.compact("one", frozenset({"feedback", "inference"}))
            body = await (await client.get(f"{root}?record_id=inference&scan_after=0")).text()
            assert "record_id=feedback" in body
            scenario.records.purge_compacted("one", before=10**12)
            response = await client.get(f"{root}?record_id=feedback")
            body = await response.text()
            assert response.status == 200 and "Record body unavailable" in body and "commit_step=1" in body
            assert (await client.get(f"{root}?record_id=other-secret")).status == 404
            assert (await client.get(f"{root}?commit_step=2")).status == 404
            assert (await client.get("/reef/scenarios/missing/learning/page")).status == 404
            assert not history_dispatcher.has_scenario("missing")

    asyncio.run(run())


def test_page_authentication_and_query_validation(history_dispatcher: Dispatcher) -> None:
    async def run() -> None:
        async with TestClient(TestServer(create_app(history_dispatcher, tokens="page-secret"))) as client:
            root = "/reef/scenarios/one/learning/page"
            assert (await client.get(root)).status == 401
            response = await client.get(f"{root}?token=page-secret")
            assert response.status == 200
            assert "token=page-secret" in await response.text()
            headers = {"Authorization": "Bearer wrong"}
            assert (await client.get(f"{root}?token=page-secret", headers=headers)).status == 401
            headers = {"Authorization": "Bearer page-secret"}
            body = await (await client.get(f"{root}?token=unused", headers=headers)).text()
            assert "page-secret" not in body and "token=unused" not in body
            assert (await client.get("/reef/scenarios/one/records?token=page-secret")).status == 401
            assert (await client.post(f"{root}?token=page-secret")).status == 401
            for query in (
                "record_id=",
                "record_id=x&commit_step=1",
                "commit_step=0",
                "commit_step=no",
                "after_sequence=-1",
                "after_step=-1",
                "input_offset=-1",
                "scan_after=0",
                "record_id=x&scan_after=-1",
                "show_payload=1",
                "record_id=x&show_payload=yes",
                "record_id=x&record_id=y",
                "record_id=" + "x" * 257,
            ):
                assert (await client.get(f"{root}?{query}", headers=headers)).status == 400, query

    asyncio.run(run())


def test_sparse_reference_scan_advances_over_unmatched_rows(history_dispatcher: Dispatcher) -> None:
    scenario = history_dispatcher.get_or_create_scenario("one")
    for index in range(100):
        scenario.records.append(
            AgentRecord.create(
                scenario="one",
                request_type=RequestType.INFERENCE,
                agent_record_id=f"unrelated-{index}",
                payload={},
            )
        )
    scenario.records.append(
        AgentRecord.create(
            scenario="one",
            request_type=RequestType.REPORT,
            agent_record_id="late-feedback",
            payload={},
            references=("feedback",),
        )
    )
    first = history_dispatcher.read_learning_page("one", LearningQuery(record_id="feedback", scan_after=0))
    assert first.references == () and first.scanned == 100 and first.next_scan is not None
    second = history_dispatcher.read_learning_page(
        "one", LearningQuery(record_id="feedback", scan_after=first.next_scan)
    )
    assert second.references is not None
    assert [row.agent_record_id for row in second.references] == ["late-feedback"]
    assert second.next_scan is None
    page = history_dispatcher.read_learning_page("one", LearningQuery())
    assert len(page.records) == 20 and page.next_sequence is not None
    tail = history_dispatcher.read_learning_page("one", LearningQuery(after_sequence=page.next_sequence))
    assert tail.records[0].sequence > page.records[-1].sequence


def test_commit_pages_bound_inputs_and_keep_missing_body_links(history_dispatcher: Dispatcher) -> None:
    scenario = history_dispatcher.get_or_create_scenario("one")
    assert isinstance(scenario.store, CommitLogScenarioStore)
    log = scenario.store.commit_log
    assert log is not None
    original = scenario.store.history()[0]
    for step in range(2, 22):
        log.append(replace(original, step=step))
    page = history_dispatcher.read_learning_page("one", LearningQuery(record_id="feedback"))
    assert len(page.commits) == 20 and page.next_step == 20
    tail = history_dispatcher.read_learning_page("one", LearningQuery(record_id="feedback", after_step=20))
    assert [commit.step for commit in tail.commits] == [21]
    commit = replace(original, consumed_ids=frozenset(f"input-{index:02}" for index in range(21)))
    page = LearningPage(scenario="one", durable=True, commit=commit, release_index=3)
    body = build_learning_page(page, LearningQuery(commit_step=1), "secret & token")
    assert "record_id=input-19" in body and "record_id=input-20" not in body
    assert "input_offset=20" in body and "/reef/harness/releases/3/page?" in body
    body = build_learning_page(page, LearningQuery(commit_step=1, input_offset=20))
    assert "record_id=input-20" in body and "record_id=input-19" not in body


def test_unsupported_history_and_escaped_bounded_previews() -> None:
    page = LearningPage(scenario="one", durable=False)
    body = build_learning_page(page, LearningQuery(record_id="missing", scan_after=0))
    assert "Durable commit history is unavailable" in body
    assert "does not support reference scans" in body
    record = AgentRecord.create(
        scenario="one",
        request_type=RequestType.INFERENCE,
        agent_record_id='x&"<tag>',
        payload={"text": "a" * 20_000},
    )
    page = LearningPage(scenario="one", durable=True, record=StoredRecord(1, record, None), record_known=True)
    body = build_learning_page(page, LearningQuery(record_id=record.agent_record_id, show_payload=True))
    assert "Preview truncated at 16,000 characters" in body and "a" * 16_001 not in body
    assert 'x&"<tag>' not in body
    href = learning_href("space & name", {"record_id": record.agent_record_id})
    assert href == "/reef/scenarios/space%20%26%20name/learning/page?" + urlencode(
        {"record_id": record.agent_record_id}
    )
    preview_href = learning_href("one", {"record_id": record.agent_record_id, "show_payload": 1})
    assert html.escape(preview_href, quote=True) in build_learning_page(
        page, LearningQuery(record_id=record.agent_record_id)
    )


def test_observed_store_delegates_optional_reference_reads(history_dispatcher: Dispatcher) -> None:
    scenario = history_dispatcher.get_or_create_scenario("one")
    observed = ObservedRecordStore(scenario.records, RecordObserver())
    assert observed.reference_page("one") == scenario.records.reference_page("one")
    assert RecordStore.reference_page(scenario.records, "one") is None
