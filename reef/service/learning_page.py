"""Small server-rendered views connecting retained records and committed learning steps."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from urllib.parse import quote, urlencode

from reef.core.artifact_ref import encode_artifact_ref
from reef.scenario.history import LearningPage, LearningQuery
from reef.service.page_chrome import document, escape, stamp


def learning_href(scenario: str, parameters: Mapping[str, str | int] | None = None) -> str:
    """Build a same-origin URL, keeping identifiers out of path and query syntax."""
    path = f"/reef/scenarios/{quote(scenario, safe='')}/learning/page"
    return f"{path}?{urlencode(parameters)}" if parameters else path


def bounded_json(value: object) -> str:
    """Escape a JSON preview capped at 16,000 characters, visibly marking truncation."""
    parts: list[str] = []
    remaining = 16_000
    for part in json.JSONEncoder(ensure_ascii=False, indent=2).iterencode(value):
        if len(part) > remaining:
            parts.append(part[:remaining])
            parts.append("\n[Preview truncated at 16,000 characters]")
            break
        parts.append(part)
        remaining -= len(part)
    return f"<pre>{escape(''.join(parts))}</pre>"


def build_learning_page(page: LearningPage, query: LearningQuery, token: str | None = None) -> str:
    """Render only persisted relationships; missing commits never imply a learning decision."""
    credentials: dict[str, str | int] = {} if token is None else {"token": token}

    def link(label: str, **parameters: str | int) -> str:
        href = learning_href(page.scenario, {**credentials, **parameters})
        return f'<a href="{escape(href)}">{escape(label)}</a>'

    def record_links(ids: Sequence[str], **selection: str | int) -> str:
        visible = ids[query.input_offset : query.input_offset + 20]
        content = "<ul>" + "".join(f"<li>{link(value, record_id=value)}</li>" for value in visible) + "</ul>"
        if query.input_offset + 20 < len(ids):
            content += link("More input records", **selection, input_offset=query.input_offset + 20)
        return content if ids else "<p>No input record IDs were recorded.</p>"

    def commit_table() -> str:
        rows = "".join(
            f"<tr><td>{link(str(commit.step), commit_step=commit.step)}</td>"
            f"<td>{escape(commit.operation)}</td><td>{stamp(commit.recorded_at)}</td>"
            f"<td>{len(commit.consumed_ids)}</td><td>{escape(commit.artifact_ref.release_id)}</td></tr>"
            for commit in page.commits
        )
        table = (
            "<table><thead><tr><th>Commit step</th><th>Operation</th><th>Recorded</th>"
            f"<th>Consumed records</th><th>Associated release</th></tr></thead><tbody>{rows}</tbody></table>"
        )
        if not page.commits:
            table += (
                "<p>No matching commits in this page. This does not establish whether a record will be learned.</p>"
            )
        if page.next_step is not None:
            parameters: dict[str, str | int] = {"after_step": page.next_step}
            if query.record_id is not None:
                parameters["record_id"] = query.record_id
            else:
                parameters["after_sequence"] = query.after_sequence
            table += link("More commits", **parameters)
        return table

    body = f'<nav>{link("All learning history")}</nav>'
    if not page.durable:
        body += "<p>Durable commit history is unavailable for this scenario.</p>"

    if query.commit_step is not None:
        heading = f"Commit step {query.commit_step}"
        commit = page.commit
        if commit is None:
            body += "<p>This commit is unavailable in the scenario's durable history.</p>"
        else:
            body += "<h2>Recorded result</h2>" + bounded_json(
                {
                    "step": commit.step,
                    "operation": commit.operation,
                    "operation_verified": commit.operation_verified,
                    "pending": commit.pending,
                    "recorded_at": commit.recorded_at,
                    "artifact_ref": encode_artifact_ref(commit.artifact_ref),
                }
            )
            body += "<p>The associated release may be unchanged; consumption does not imply an improvement.</p>"
            if page.release_index is not None:
                parameters = {**credentials, "scenario": page.scenario}
                href = f"/reef/harness/releases/{page.release_index}/page?{urlencode(parameters)}"
                body += f'<p><a href="{escape(href)}">Open harness step and evaluation</a></p>'
            body += f"<h2>Consumed records ({len(commit.consumed_ids)})</h2>"
            body += record_links(sorted(commit.consumed_ids), commit_step=commit.step)
            body += "<h2>Recorded metrics</h2>" + bounded_json(commit.metrics)
    elif query.record_id is not None:
        heading = f"Record {query.record_id}"
        stored = page.record
        if stored is None:
            body += "<p>Record body unavailable. It may have expired or never been retained.</p>"
        else:
            item = stored.item
            body += "<h2>Record</h2>" + bounded_json(
                {
                    "agent_record_id": item.agent_record_id,
                    "sequence": stored.sequence,
                    "request_type": item.request_type.value,
                    "created_at": item.created_at,
                    "compacted_at": stored.compacted_at,
                    "score": item.payload.get("score"),
                    "artifact_ref": encode_artifact_ref(item.artifact_ref) if item.artifact_ref else None,
                }
            )
            body += "<p>Compaction means retirement from training reads, not proof of learning. "
            body += "An inference artifact identifies the version that served that request.</p>"
            body += "<h2>Referenced records</h2>" + record_links(item.references, record_id=query.record_id)
            if query.show_payload:
                body += "<h2>Payload preview</h2>" + bounded_json(item.payload)
            else:
                body += f'<p>{link("Load payload preview", record_id=query.record_id, show_payload=1)}</p>'
        body += "<h2>Commits consuming this exact record ID</h2>" + commit_table()
        body += "<h2>Records referencing this ID</h2>"
        if query.scan_after is None:
            body += f'<p>{link("Scan retained references", record_id=query.record_id, scan_after=0)}</p>'
        elif page.references is None:
            body += "<p>This storage adapter does not support reference scans.</p>"
        else:
            body += f"<p>Scanned {page.scanned} retained records after sequence {query.scan_after}.</p><ul>"
            body += (
                "".join(
                    f"<li>{link(row.agent_record_id, record_id=row.agent_record_id)} ({escape(row.request_type.value)})</li>"
                    for row in page.references
                )
                + "</ul>"
            )
            if not page.references:
                body += "<p>No matching references in this scanned slice.</p>"
            if page.next_scan is not None:
                body += link("Scan next 100 records", record_id=query.record_id, scan_after=page.next_scan)
            else:
                body += "<p>End of currently retained records. Purged records are not included.</p>"
    else:
        heading = "Learning history"
        body += "<h2>Retained records</h2><table><thead><tr><th>Sequence</th><th>Record</th>"
        body += "<th>Type</th><th>Created</th><th>Compacted</th></tr></thead><tbody>"
        body += (
            "".join(
                f"<tr><td>{stored.sequence}</td>"
                f"<td>{link(stored.item.agent_record_id, record_id=stored.item.agent_record_id)}</td>"
                f"<td>{escape(stored.item.request_type.value)}</td><td>{stamp(stored.item.created_at)}</td>"
                f"<td>{'Yes' if stored.compacted_at is not None else 'No'}</td></tr>"
                for stored in page.records
            )
            + "</tbody></table>"
        )
        if not page.records:
            body += "<p>No retained records in this page.</p>"
        if page.next_sequence is not None:
            body += link("More records", after_sequence=page.next_sequence, after_step=query.after_step)
        body += "<h2>Committed steps</h2>" + commit_table()

    return document(
        title=heading,
        style="pre{white-space:pre-wrap;overflow-wrap:anywhere}td{overflow-wrap:anywhere}"
        "table{width:100%;table-layout:fixed}h2{margin-top:2rem}nav{margin-bottom:1rem}",
        breadcrumb="History",
        context=page.scenario,
        state="read only",
        eyebrow="Records and commits",
        heading=heading,
        subtitle="Browse stored relationships. Pages are live reads, not a historical snapshot.",
        body=body,
        home=learning_href(page.scenario, credentials),
        section="Learning",
        footer_label="Learning history",
    )
