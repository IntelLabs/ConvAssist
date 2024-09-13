from configparser import ConfigParser
import logging
import os

from ConvAssist.context_tracker import ContextTracker
from ConvAssist.predictor.smoothed_ngram_predictor import SmoothedNgramPredictor
from ConvAssist.predictor.utilities.nlp import NLP
from ConvAssist.utilities.databaseutils.sqllite_dbconnector import SQLiteDatabaseConnector
from ConvAssist.predictor.utilities.ngram_map import NgramMap


class CannedWordPredictor(SmoothedNgramPredictor):
    def __init__(
            self, 
            config: ConfigParser, 
            context_tracker: ContextTracker, 
            predictor_name: str, 
            logger: logging.Logger | None = None
    ):
        super().__init__(
            config, context_tracker, 
            predictor_name, logger=logger
        )

        # load the natural language processing model
        self.nlp = NLP().get_nlp()

        # object and subject constants
        self.OBJECT_DEPS = {"dobj","pobj", "dative", "attr", "oprd", "npadvmod", "amod","acomp","advmod"}
        self.SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "agent", "expl"}
        # tags that define wether the word is wh-
        self.WH_WORDS = {"WP", "WP$", "WRB"}
        self.stopwordsList = []
        stoplist = open(self.stopwordsFile,"r").readlines()
        for s in stoplist:
            self.stopwordsList.append(s.strip())

# Override default properties
    @property
    def sentences_db(self):
        return os.path.join(self._personalized_resources_path, self._sentences_db)

    # @property
    # def database(self):
    #     return os.path.join(self._personalized_resources_path, self._database)

    def extract_svo(self, sent):
        doc = self.nlp(sent)
        sub = []
        at = []
        ve = []
        imp_tokens = []
        for token in doc:
            # is this a verb?
            if token.pos_ == "VERB":
                ve.append(token.text)
                if(token.text.lower() not in self.stopwordsList and token.text.lower() not in imp_tokens):
                    imp_tokens.append(token.text.lower())
            # is this the object?
            if token.dep_ in self.OBJECT_DEPS or token.head.dep_ in self.OBJECT_DEPS:
                at.append(token.text)
                if(token.text.lower() not in self.stopwordsList and token.text.lower() not in imp_tokens):
                    imp_tokens.append(token.text.lower())
            # is this the subject?
            if token.dep_ in self.SUBJECT_DEPS or token.head.dep_ in self.SUBJECT_DEPS:
                sub.append(token.text)
                if(token.text.lower() not in self.stopwordsList and token.text.lower() not in imp_tokens):
                    imp_tokens.append(token.text.lower())
        return " ".join(imp_tokens).strip().lower()

    def recreate_database(self):

        # STEP 1: Update personalized_corups if it is not None
        personalized_corpus = self.read_personalized_corpus()

        if personalized_corpus and self.init_database_connector_if_ready():
            try:

                # STEP 2: CREATE CANNED_NGRAM DATABASE IF IT DOES NOT EXIST
                try:
                    self.ngram_db_conn.create_ngram_table(cardinality=1)
                    self.ngram_db_conn.create_ngram_table(cardinality=2)
                    self.ngram_db_conn.create_ngram_table(cardinality=3)
                except Exception as e:
                    self.logger.error(f"exception in creating personalized db : {e}")

                # TODO: CANNED
                sentence_db = SQLiteDatabaseConnector(self.sentences_db)
                sentence_db.connect()

                # CHECK FOR PHRASES TO ADD AND PHRASES TO DELETE FROM THE DATABASES
                sent_db_dict = {}
                res = sentence_db.fetch_all("SELECT * FROM sentences")
                if res:
                    for r in res:
                        sent_db_dict[r[0]]= r[1]
                phrases_toRemove = list(set(sent_db_dict.keys())-set(personalized_corpus))
                phrases_toAdd = list(set(personalized_corpus)-set(sent_db_dict.keys()))
                self.logger.info("PHRASES TO ADD = " + str(phrases_toAdd))
                self.logger.info("PHRASES TO REMOVE = "+ str(phrases_toRemove))

                # Add phrases_toAdd to the database and ngram
                for phrase in phrases_toAdd:
                    # Add to 
                    query = '''INSERT INTO sentences (sentence, count)
                                    VALUES (?,?)'''
                    phraseToInsert = (phrase, 1)
                    sentence_db.execute_query(query, phraseToInsert)

                    ### Add phrase to ngram
                    for curr_card in range(self.cardinality):
                        ngram_map = NgramMap()
                        ngs = self.generate_ngrams(phrase.lower().split(), curr_card)
                        ngram_map = self.getNgramMap(ngs, ngram_map)

                        # for every ngram, get db count, update or insert
                        for ngram, count in ngram_map.items():
                            old_count = self.ngram_db_conn.ngram_count(ngram)
                            if old_count > 0:
                                self.ngram_db_conn.update_ngram(ngram, old_count + count)
                                self.ngram_db_conn.commit()
                            else:
                                self.ngram_db_conn.insert_ngram(ngram, count)
                                self.ngram_db_conn.commit()

                for phrase in phrases_toRemove:
                ##### Remove phrases_toRemove from the database
                    query = 'DELETE FROM sentences WHERE sentence=?'
                    sentence_db.execute_query(query, (phrase,))
                    self.logger.info(f"Phrase {phrase} deleted from sentence_db.")
                    phraseFreq = sent_db_dict[phrase]
                    ### Remove phrase to ngram
                    for curr_card in range(self.cardinality):
                        ngram_map = NgramMap()
                        imp_words = self.extract_svo(phrase)
                        ngs = self.generate_ngrams(imp_words.split(), curr_card)
                        ngram_map = self.getNgramMap(ngs, ngram_map)
                        # for every ngram, get db count, update or insert
                        for ngram, count in ngram_map.items():
                            countToDelete = phraseFreq*count
                            old_count = self.ngram_db_conn.ngram_count(ngram)
                            if old_count > countToDelete:
                                self.ngram_db_conn.update_ngram(ngram, old_count - countToDelete)
                                self.ngram_db_conn.commit()
                            elif old_count == countToDelete:
                                self.ngram_db_conn.remove_ngram(ngram)
                                self.ngram_db_conn.commit()
                            elif old_count < countToDelete:
                                self.logger.info("SmoothedNgramPredictor RecreateDB Delete function: Count in DB < count to Delete")

            except Exception as e:
                self.logger.error(f"= {e}")
            finally:
                sentence_db.close()
