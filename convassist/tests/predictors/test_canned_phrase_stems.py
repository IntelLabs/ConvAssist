# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

"""Direct matching characterization using temporary SQLite data and mocked models."""

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy
import pytest

from convassist.predictor import canned_phrases_predictor as module
from convassist.predictor.utilities.prediction import Prediction


@pytest.fixture
def make_predictor(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "SentenceTransformer", Mock())
    monkeypatch.setattr(module.hnswlib, "Index", Mock())

    def make(phrases=()):
        corpus = tmp_path / "phrases.txt"
        corpus.write_text("".join(f"{phrase}\n" for phrase, _ in phrases), encoding="utf-8")
        with sqlite3.connect(tmp_path / "phrases.db") as conn:
            conn.execute("CREATE TABLE sentences (sentence TEXT PRIMARY KEY, count INTEGER)")
            conn.executemany("INSERT INTO sentences VALUES (?, ?)", phrases)
        module.joblib.dump(
            {"sentences": [phrase for phrase, _ in phrases],
             "embeddings": numpy.zeros((len(phrases), 4))},
            tmp_path / "embeddings.pkl",
        )
        (tmp_path / "semantic.index").touch()
        predictor = object.__new__(module.CannedPhrasesPredictor)
        predictor.device = "cpu"
        predictor.logger = Mock()
        predictor._predictor_name = "canned-test"
        predictor._static_resources_path = str(tmp_path)
        predictor._personalized_resources_path = str(tmp_path)
        predictor._personalized_cannedphrases = "phrases.txt"
        predictor._sentences_db = "phrases.db"
        predictor._embedding_cache_path = "embeddings.pkl"
        predictor._index_path = "semantic.index"
        predictor._sbertmodel = "unused-mocked-model"
        predictor._learn = True
        predictor.context_tracker = SimpleNamespace(context="", token_count=0)
        predictor.configure()
        with sqlite3.connect(predictor.sentences_db_path) as conn:
            conn.executemany("UPDATE sentences SET count=? WHERE sentence=?",
                             [(count, phrase) for phrase, count in phrases])
        return predictor

    return make


def direct(predictor, context):
    return [(s.word, s.probability)
            for s in predictor._find_direct_matches(context, Prediction())]


def legacy_result(predictor, context):
    phrases = predictor.cannedData.all_phrases_as_dict()
    total = sum(phrases.values())
    context_stems = [predictor.stemmer.stem(w) for w in module.word_tokenize(context)]
    rows = []
    for phrase, count in phrases.items():
        stems = [predictor.stemmer.stem(w) for w in module.word_tokenize(phrase)]
        matches = sum(stem in stems for stem in context_stems)
        if matches:
            rows.append((phrase, matches, count / total))
    rows.sort(key=lambda row: (row[1], row[2]), reverse=True)
    return [(phrase, matches + probability) for phrase, matches, probability in rows]


@pytest.mark.parametrize("context", ["go go", "running runs", "HELLO, hello!", "", "  ", "missing", "!"])
def test_direct_characterization(make_predictor, context):
    predictor = make_predictor([
        ("go home", 2), ("go go away", 2), ("runs fast", 1),
        ("running slowly", 3), ("Hello, world!", 2), ("unrelated", 10),
    ])
    assert direct(predictor, context) == legacy_result(predictor, context)
    predictor.logger.error.assert_not_called()


def test_duplicate_context_stems_count_but_phrase_duplicates_do_not(make_predictor):
    predictor = make_predictor([("running running", 1), ("runs", 1)])
    assert direct(predictor, "running runs") == [("running running", 2.5), ("runs", 2.5)]


def test_ties_live_order_and_weighted_denominator(make_predictor):
    predictor = make_predictor([("go first", 2), ("go second", 2), ("unrelated", 6)])
    assert direct(predictor, "go") == [("go first", 1.2), ("go second", 1.2)]
    with sqlite3.connect(predictor.sentences_db_path) as conn:
        conn.execute("DELETE FROM sentences WHERE sentence='go first'")
        conn.execute("INSERT INTO sentences VALUES ('go first', 2)")
    assert direct(predictor, "go") == [("go second", 1.2), ("go first", 1.2)]


def test_empty_data_and_no_matches(make_predictor):
    predictor = make_predictor()
    assert direct(predictor, "anything") == []
    assert direct(predictor, "") == []
    predictor.logger.error.assert_not_called()


def test_direct_semantic_dedup_precedes_final_limit(make_predictor):
    predictor = make_predictor([("go first", 1), ("go second", 1), ("go third", 1)])
    predictor.corpus_phrases = ["go third", "semantic only"]
    predictor.index.knn_query.return_value = (numpy.array([[0, 1]]), numpy.array([[0.0, 0.1]]))
    predictor.context_tracker.context = "go now"
    predictor.context_tracker.token_count = 2
    semantic = predictor._find_semantic_matches
    predictor._find_semantic_matches = Mock(wraps=semantic)
    sentences, words = predictor.predict(2)
    assert [s.word for s in sentences] == ["go first", "go second"]
    combined = predictor._find_semantic_matches.call_args.args[1]
    assert [s.word for s in combined] == ["go first", "go second", "go third", "semantic only"]
    assert words == []


def test_blank_context_frequency_limit_and_one_token_skips_semantics(make_predictor):
    predictor = make_predictor([("go first", 1), ("go second", 3)])
    predictor._find_semantic_matches = Mock(side_effect=AssertionError("Unexpected semantic search"))
    sentences, words = predictor.predict(1)
    assert [(s.word, s.probability) for s in sentences] == [("go second", 0.75)]
    assert words == []
    predictor.context_tracker.context = "go"
    predictor.context_tracker.token_count = 1
    assert [s.word for s in predictor.predict(1)[0]] == ["go second"]


def test_learn_and_external_count_changes_are_live(make_predictor):
    predictor = make_predictor([("go existing", 1), ("other", 3)])
    predictor.embedder.encode.return_value = numpy.ones(4)
    predictor.learn("go learned")
    assert dict(direct(predictor, "go"))["go learned"] == 1.2
    predictor.learn("go learned")
    assert dict(direct(predictor, "go"))["go learned"] == pytest.approx(1 + 2 / 6)
    with sqlite3.connect(predictor.sentences_db_path) as conn:
        conn.execute("UPDATE sentences SET count=6 WHERE sentence='go existing'")
        conn.execute("INSERT INTO sentences VALUES ('go external', 1)")
    assert direct(predictor, "go") == legacy_result(predictor, "go")
    assert direct(predictor, "go")[0] == ("go existing", 1.5)


def test_configure_precomputes_stem_membership(make_predictor):
    predictor = make_predictor([("running running", 1), ("runs fast", 1)])
    assert predictor._phrase_stems == {"running running": {"run"}, "runs fast": {"run", "fast"}}


def test_unchanged_phrases_reuse_cache_and_read_database_each_prediction(make_predictor, monkeypatch):
    predictor = make_predictor([("go first", 1), ("go second", 1)])
    cache_entries = dict(predictor._phrase_stems)
    tokenize = Mock(wraps=module.word_tokenize)
    monkeypatch.setattr(module, "word_tokenize", tokenize)
    stem = Mock(wraps=predictor.stemmer.stem)
    monkeypatch.setattr(predictor.stemmer, "stem", stem)
    read = Mock(wraps=predictor.cannedData.all_phrases_as_dict)
    monkeypatch.setattr(predictor.cannedData, "all_phrases_as_dict", read)
    for _ in range(3):
        assert direct(predictor, "go go") == [("go first", 2.5), ("go second", 2.5)]
    assert [call.args for call in tokenize.call_args_list] == [("go go",)] * 3
    assert [call.args for call in stem.call_args_list] == [("go",)] * 6
    assert read.call_count == 3
    for phrase, stems in cache_entries.items():
        assert predictor._phrase_stems[phrase] is stems


def test_external_add_remove_readd_and_count_only_changes(make_predictor, monkeypatch):
    predictor = make_predictor([("go first", 1), ("go second", 1)])
    original = predictor._phrase_stems["go first"]
    tokenize = Mock(wraps=module.word_tokenize)
    monkeypatch.setattr(module, "word_tokenize", tokenize)
    with sqlite3.connect(predictor.sentences_db_path) as conn:
        conn.execute("UPDATE sentences SET count=3 WHERE sentence='go first'")
    assert direct(predictor, "go") == [("go first", 1.75), ("go second", 1.25)]
    assert predictor._phrase_stems["go first"] is original
    assert [call.args for call in tokenize.call_args_list] == [("go",)]

    tokenize.reset_mock()
    with sqlite3.connect(predictor.sentences_db_path) as conn:
        conn.execute("DELETE FROM sentences WHERE sentence='go first'")
        conn.execute("INSERT INTO sentences VALUES ('go external', 1)")
    assert direct(predictor, "go") == [("go second", 1.5), ("go external", 1.5)]
    assert set(predictor._phrase_stems) == {"go second", "go external"}
    assert [call.args for call in tokenize.call_args_list] == [("go external",), ("go",)]

    tokenize.reset_mock()
    with sqlite3.connect(predictor.sentences_db_path) as conn:
        conn.execute("INSERT INTO sentences VALUES ('go first', 1)")
    assert direct(predictor, "go") == [(phrase, 1 + 1 / 3)
                                       for phrase in ["go second", "go external", "go first"]]
    assert predictor._phrase_stems["go first"] is not original
    assert [call.args for call in tokenize.call_args_list] == [("go first",), ("go",)]


def test_learn_refreshes_only_new_phrase_on_next_direct_match(make_predictor, monkeypatch):
    predictor = make_predictor([("go existing", 1)])
    predictor.embedder.encode.return_value = numpy.ones(4)
    tokenize = Mock(wraps=module.word_tokenize)
    monkeypatch.setattr(module, "word_tokenize", tokenize)
    predictor.learn("go learned")
    assert "go learned" not in predictor._phrase_stems
    assert direct(predictor, "go") == [("go existing", 1.5), ("go learned", 1.5)]
    assert [call.args for call in tokenize.call_args_list] == [("go learned",), ("go",)]
    tokenize.reset_mock()
    predictor.learn("go learned")
    assert direct(predictor, "go")[0] == ("go learned", 1 + 2 / 3)
    assert [call.args for call in tokenize.call_args_list] == [("go",)]


def test_empty_live_snapshot_discards_cache_and_can_repopulate(make_predictor):
    predictor = make_predictor([("go original", 1)])
    with sqlite3.connect(predictor.sentences_db_path) as conn:
        conn.execute("DELETE FROM sentences")
    assert direct(predictor, "go") == []
    assert predictor._phrase_stems == {}
    with sqlite3.connect(predictor.sentences_db_path) as conn:
        conn.execute("INSERT INTO sentences VALUES ('go new', 1)")
    assert direct(predictor, "go") == [("go new", 2.0)]
    assert set(predictor._phrase_stems) == {"go new"}


def test_configure_resets_all_entries_including_empty_data(make_predictor, monkeypatch):
    predictor = make_predictor([("go original", 1)])
    old = predictor._phrase_stems["go original"]
    predictor.configure()
    assert predictor._phrase_stems["go original"] == old
    assert predictor._phrase_stems["go original"] is not old
    Path(predictor.personalized_cannedphrases).write_text("", encoding="utf-8")
    # A true empty snapshot (an empty corpus file is otherwise read as ['']).
    monkeypatch.setattr(predictor.cannedData.__class__, "_read_personalized_corpus", lambda *args: [])
    predictor.configure()
    assert predictor._phrase_stems == {}
    assert direct(predictor, "go") == []
    monkeypatch.setattr(predictor.cannedData.__class__, "_read_personalized_corpus", lambda *args: ["go replacement"])
    predictor.configure()
    assert set(predictor._phrase_stems) == {"go replacement"}
    assert direct(predictor, "go") == [("go replacement", 2.0)]
