# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

from convassist.utilities.databaseutils.sqllite_dbconnector import SQLiteDatabaseConnector


class cannedData:
    def __init__(self, db_path, canned_data_path):
        self.sentence_db = SQLiteDatabaseConnector(db_path)
        self.canned_data_path = canned_data_path
        self._add_data = []
        self._remove_data = []

        # Create sentences_db
        self.sentence_db.connect()
        columns = ["sentence TEXT PRIMARY KEY", "count INTEGER"]
        self.sentence_db.create_table("sentences", columns)

        # Close sentences_db
        self.sentence_db.close()

        self.update()

    @property
    def length(self):
        return len(self.all_phrases_as_list())

    @property
    def add_data(self):
        return self._add_data

    @property
    def remove_data(self):
        return self._remove_data

    @property
    def add_remove_data(self):
        return self._add_data, self._remove_data

    def update(self):
        # check if any new sentences have been added to or removed from the
        # canned data.  If so, update the sentences_db
        personalized_data = self._read_personalized_corpus(self.canned_data_path)
        existing_data = self.all_phrases_as_dict() or {}

        self._add_data = list(set(personalized_data) - set(existing_data.keys()))
        self._remove_data = list(set(existing_data.keys()) - set(personalized_data))

        if self._add_data:
            for phrase in self._add_data:
                self.learn(phrase)

        if self._remove_data:
            for phrase in self._remove_data:
                self.remove(phrase)

    def retrieve(self, phrase) -> dict:
        result_dict = {}
        self.sentence_db.connect()
        query = "SELECT * FROM sentences WHERE sentence=?"
        res = self.sentence_db.fetch_one(query, (phrase,))
        self.sentence_db.close()
        if res:
            result_dict = {res[0]: res[1]}

        return result_dict

    def learn(self, phrase, count=1):
        existing_data = self.retrieve(phrase)

        self.sentence_db.connect()
        if existing_data:
            count += existing_data[phrase]
            query = """UPDATE sentences SET count = ? WHERE sentence = ?"""
            phraseToInsert = (count, phrase)
        else:
            query = """INSERT INTO sentences (sentence, count) VALUES (?,?)"""
            phraseToInsert = (phrase, count)
        self.sentence_db.execute_query(query, phraseToInsert)
        self.sentence_db.close()

    def remove(self, phrase):
        self.sentence_db.connect()
        query = "DELETE FROM sentences WHERE sentence=?"
        self.sentence_db.execute_query(query, (phrase,))
        self.sentence_db.close()

    def all_phrases_as_dict(self) -> dict:
        self.sentence_db.connect()
        res = self.sentence_db.fetch_all("SELECT * FROM sentences")
        self.sentence_db.close()
        result_dict = {row[0]: row[1] for row in res}
        return result_dict

    def all_phrases_as_list(self) -> list:
        phrases = self.all_phrases_as_dict()
        return list(phrases.keys())

    def _read_personalized_corpus(self, corpus_path):
        corpus = []

        with open(corpus_path) as f:
            corpus = f.readlines()
            corpus = [s.strip() for s in corpus]

        return corpus
