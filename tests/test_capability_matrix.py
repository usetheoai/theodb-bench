"""The capability matrix, generated rather than written.

B-073 asks for a report that lists every declared capability and which adapters
exercise it, and asks for it to be *generated* — because the version that gets
written by hand is the version that drifts. The measurement that opened the item
was exactly this crossing done manually: six of fourteen capabilities reachable,
eight with no adapter at all. A month later nobody would know which number is
current, and the README would still say six.

Generated has a second property that matters more here: a capability an adapter
*declares* but cannot reach is invisible to a hand-written table and visible to
this one, because both sides come from the same registry the runs use.
"""

from __future__ import annotations

import pytest
from theodb_bench.adapters.base import CAPABILITIES
from theodb_bench.capabilities import capability_matrix, render_capability_matrix


def test_every_declared_capability_appears_exactly_once() -> None:
    matrix = capability_matrix()

    assert [row.capability for row in matrix] == list(CAPABILITIES)


def test_a_capability_names_the_adapters_that_declare_it() -> None:
    matrix = {row.capability: row for row in capability_matrix()}

    assert "theodb" in matrix["vector_hnsw"].adapters
    assert "pgvector" in matrix["vector_hnsw"].adapters


def test_a_capability_nobody_declares_is_reported_as_such_not_omitted() -> None:
    """An absent row reads as "not a thing"; an empty row reads as "nothing
    reaches this yet", which is the true and useful statement."""
    matrix = {row.capability: row for row in capability_matrix()}

    for capability in ("rerank", "vectorizer", "ai_sql"):
        assert matrix[capability].adapters == ()
        assert not matrix[capability].reachable


def test_the_fake_adapter_does_not_count_as_reach() -> None:
    """The fake exists to test the harness, and counting it would let the matrix
    report a pillar as reachable when no real system implements it."""
    matrix = {row.capability: row for row in capability_matrix()}

    assert all("fake" not in row.adapters for row in matrix.values())


def test_the_summary_counts_what_is_reachable() -> None:
    matrix = capability_matrix()

    reachable = sum(1 for row in matrix if row.reachable)

    assert reachable == 11, f"expected 11 of {len(CAPABILITIES)} reachable, got {reachable}"


def test_the_rendered_table_names_every_capability() -> None:
    rendered = render_capability_matrix()

    for capability in CAPABILITIES:
        assert capability in rendered


def test_the_rendered_table_carries_the_count_it_is_read_for() -> None:
    rendered = render_capability_matrix()

    assert f"11 of {len(CAPABILITIES)}" in rendered


def test_the_readme_table_is_the_generated_one() -> None:
    """The check that makes this worth generating: the README's table must be
    what the registry says today, not what it said when someone typed it."""
    from pathlib import Path

    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    rendered = render_capability_matrix()

    assert rendered.strip() in readme, (
        "the README's capability matrix is stale; regenerate it with "
        "`theodb-bench capabilities --markdown`"
    )


def test_the_cli_prints_the_matrix(capsys: pytest.CaptureFixture[str]) -> None:
    from theodb_bench.cli import main

    assert main(["capabilities"]) == 0
    assert "11 of 14" in capsys.readouterr().out
