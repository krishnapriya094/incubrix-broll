"""Unit tests: pure functions, no I/O."""
from app.adapters import _hash_vector, cosine
from app.pipeline import split_sentences, _timed_from_text, write_srt


def test_split_sentences_basic():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_split_sentences_single():
    assert split_sentences("No terminator here") == ["No terminator here"]


def test_timed_from_text_covers_full_duration():
    scenes = _timed_from_text(["a b c", "d e", "f"], 10.0)
    assert scenes[0]["start"] == 0.0
    assert scenes[-1]["end"] == 10.0
    # scenes are contiguous
    for prev, nxt in zip(scenes, scenes[1:]):
        assert abs(prev["end"] - nxt["start"]) < 1e-6


def test_embeddings_are_deterministic_and_normalised():
    v1 = _hash_vector("save money every month")
    v2 = _hash_vector("save money every month")
    assert v1 == v2
    assert abs(sum(x * x for x in v1) - 1.0) < 1e-6
    # related text should score higher than unrelated
    assert cosine(_hash_vector("save money coins budget"),
                  _hash_vector("money savings finance")) > \
           cosine(_hash_vector("save money coins budget"),
                  _hash_vector("nature trees outdoors"))


def test_write_srt_format(tmp_path):
    scenes = [{"index": 0, "text": "Hello world.", "start": 0.0, "end": 1.5}]
    path = tmp_path / "c.srt"
    write_srt(scenes, path)
    body = path.read_text()
    assert "00:00:00,000 --> 00:00:01,500" in body
    assert "Hello world." in body
