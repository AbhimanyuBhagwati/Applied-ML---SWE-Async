import asyncio
import sqlite3

import pytest

from app.history import HistoryStore
from app.schemas import ChatResponse, Citation, ToolCall


def answer(query="who was second?", response="Buzz Aldrin.", **extra):
    return ChatResponse(query=query, response=response, finish_reason="COMPLETE", **extra)


@pytest.fixture
def store(tmp_path):
    store = HistoryStore(str(tmp_path / "history.db"))
    yield store
    store.close()


@pytest.mark.asyncio
async def test_a_turn_comes_back_as_it_went_in(store):
    await store.record(
        answer(
            tool_plan="I will look this up.",
            tool_calls=[
                ToolCall(
                    name="search_wikipedia",
                    arguments={"query": "moon"},
                    result_count=1,
                    results=[{"title": "Buzz Aldrin"}],
                )
            ],
            citations=[Citation(start=0, end=11, text="Buzz Aldrin", sources=["u"])],
        )
    )

    entries, total = await store.page(limit=50, offset=0)

    assert total == 1
    entry = entries[0]
    assert entry["query"] == "who was second?"
    assert entry["response"] == "Buzz Aldrin."
    assert entry["finish_reason"] == "COMPLETE"
    assert entry["tool_plan"] == "I will look this up."
    # The grounding survives the round trip, not just the answer.
    assert entry["tool_calls"][0]["name"] == "search_wikipedia"
    assert entry["tool_calls"][0]["results"][0]["title"] == "Buzz Aldrin"
    assert entry["citations"][0]["sources"] == ["u"]
    assert entry["created_at"].endswith("+00:00"), "timestamps are UTC"


@pytest.mark.asyncio
async def test_history_reads_newest_first(store):
    for i in range(3):
        await store.record(answer(query=f"Q{i}"))

    entries, _ = await store.page(limit=50, offset=0)

    assert [e["query"] for e in entries] == ["Q2", "Q1", "Q0"]


@pytest.mark.asyncio
async def test_an_empty_history_is_not_an_error(store):
    assert await store.page(limit=50, offset=0) == ([], 0)


@pytest.mark.asyncio
async def test_paging_walks_the_whole_history(store):
    for i in range(5):
        await store.record(answer(query=f"Q{i}"))

    seen = []
    offset = 0
    while True:
        entries, total = await store.page(limit=2, offset=offset)
        if not entries:
            break
        seen += [e["query"] for e in entries]
        offset += len(entries)

    assert seen == ["Q4", "Q3", "Q2", "Q1", "Q0"]
    assert total == 5, "the total is the whole history, not the page"


@pytest.mark.asyncio
async def test_an_offset_past_the_end_is_empty_not_an_error(store):
    await store.record(answer())

    assert await store.page(limit=50, offset=99) == ([], 1)


@pytest.mark.asyncio
async def test_history_survives_a_restart(tmp_path):
    """The reason this is SQLite and not a list."""
    path = str(tmp_path / "history.db")
    first = HistoryStore(path)
    await first.record(answer(query="Q"))
    first.close()

    second = HistoryStore(path)
    entries, total = await second.page(limit=50, offset=0)
    second.close()

    assert total == 1
    assert entries[0]["query"] == "Q"


@pytest.mark.asyncio
async def test_a_storage_failure_never_reaches_the_caller(store, caplog):
    """The answer is already produced; losing a row isn't worth a 500."""
    store.close()  # every write from here on raises

    with caplog.at_level("WARNING"):
        await store.record(answer())

    assert "Could not record the turn" in caplog.text


@pytest.mark.asyncio
async def test_reading_still_raises_if_the_store_is_broken(store):
    """Unlike recording. A history endpoint that quietly returned nothing
    would be indistinguishable from an empty history."""
    store.close()

    with pytest.raises(sqlite3.Error):
        await store.page(limit=50, offset=0)


@pytest.mark.asyncio
async def test_concurrent_writes_all_land(tmp_path):
    """One connection is shared across worker threads, so it needs the lock."""
    store = HistoryStore(str(tmp_path / "history.db"))
    await asyncio.gather(*(store.record(answer(query=f"Q{i}")) for i in range(50)))

    entries, total = await store.page(limit=200, offset=0)
    store.close()

    assert total == 50
    assert sorted(e["query"] for e in entries) == sorted(f"Q{i}" for i in range(50))
