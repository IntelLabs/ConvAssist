# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import os

from tqdm import tqdm

from convassist.predictor.utilities.nlp import NLP
from convassist.utilities.ngram.ngram_map import NgramMap
from convassist.utilities.ngram.ngramutil import NGramUtil

from convassist.predictor.smoothed_ngram_predictor.smoothed_ngram_predictor import SmoothedNgramPredictor
from convassist.predictor.utilities.svo_util import SVOUtil


class CannedWordPredictor(SmoothedNgramPredictor):

    def configure(self):
        self.svo_utils = SVOUtil(self.stopwordsFile)

        super().configure()

    def extract_svo(self, sent) -> str:
        return self.svo_utils.extract_svo(sent)


    def recreate_database(self):
        """
        Recreates the sentence and n-gram databases by adding new phrases and removing outdated ones.
        """
        try:
            with open(self.personalized_cannedphrases) as f:
                phrases = f.read().splitlines()

            for cardinality in range(1, self.cardinality + 1):

                with NGramUtil(self.database, self.cardinality) as ngramutil:

                    query = ngramutil.generate_ngram_insert_query(cardinality, False)

                    data = []
                    for phrase in tqdm(phrases, desc=f"Processing {cardinality}-grams", unit=" phrases", leave=False):
                        ngram_map = NgramMap(cardinality, phrase)
                        for ngram, count in ngram_map.items():
                            data.append((*ngram, count))

                    batch_size = 3000
                    for i in tqdm(range(0, len(data), batch_size), desc=f"Inserting {cardinality}-grams", unit=" batches", leave=True):
                        batch = data[i:i + batch_size]
                        ngramutil.connection.execute_many(query, batch)

        except Exception as e:
            self.logger.error(f"exception in creating personalized db : {e}")

    @property
    def startwords(self):
        return os.path.join(self._personalized_resources_path, self._startwords)


