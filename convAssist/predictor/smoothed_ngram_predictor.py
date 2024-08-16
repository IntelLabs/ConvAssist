import os
import collections
import json
import string
from pathlib import Path

from ConvAssist.predictor import Predictor
from ConvAssist.predictor.utilities.predictor_names import PredictorNames
from ConvAssist.utilities.ngram_map import NgramMap
from ConvAssist.utilities.suggestion import Suggestion
from ConvAssist.predictor.utilities.prediction import Prediction
from ConvAssist.utilities.nlp import NLP
from ConvAssist.utilities.databaseutils.sqllite_ngram_dbconnector import SQLiteNgramDatabaseConnector

class SmoothedNgramPredictor(Predictor):
    """
    Calculates prediction from n-gram model in sqlite database. 

    """

    def __init__(
            self,
            config,
            context_tracker,
            predictor_name,
            short_desc=None,
            long_desc=None,
            dbconnection=None,
            logger=None
    ):
        super().__init__(
            config, context_tracker, 
            predictor_name, short_desc, long_desc, logger
        )
        self.db = None
        self.dbconnection = dbconnection
        self.cardinality = None
        self.learn_mode_set = False

        self._database = None
        self._deltas = None
        self._learn_mode = None
        self.config = config
        self.name = predictor_name
        self.context_tracker = context_tracker
        self._read_config()
        
        # load the natural language processing model
        self.nlp = NLP().get_nlp()
        
        # object and subject constants
        self.OBJECT_DEPS = {"dobj","pobj", "dative", "attr", "oprd", "npadvmod", "amod","acomp","advmod"}
        self.SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "agent", "expl"}
        # tags that define wether the word is wh-
        self.WH_WORDS = {"WP", "WP$", "WRB"}
        self.stopwords = []
        stoplist = open(self.stopwordsFile,"r").readlines()
        for s in stoplist:
            self.stopwords.append(s.strip())

        if(self.name == PredictorNames.GeneralWord.value ):
            self.log.info("INSIDE init "+PredictorNames.GeneralWord.value)
            ##### Store the set of most frequent starting words based on an AAC dataset
            ##### These will be displayed during empty context
            if(not os.path.isfile(self.startwords)):
                aac_lines = open(self.aac_dataset,"r").readlines()
                startwords = []
                for line in aac_lines:
                    w = line.lower().split()[0]
                    startwords.append(w)
                counts = collections.Counter(startwords)
                total = sum(counts.values())
                self.precomputed_sentenceStart = {k:v/total for k,v in counts.items()}
                with open(self.startwords, 'w') as fp:
                    json.dump(self.precomputed_sentenceStart, fp)

        if(self.name == PredictorNames.PersonalizedWord.value):
            # try:
            #     self.log.debug("trying to establish connection with "+ self.database)
            #     conn_ngram = self.create_connection(self.database)
            #     self.log.debug("personalized database connection created "+ self.database)

            #     sql_create_1gram_table = """ CREATE TABLE IF NOT EXISTS _1_gram (

            #                                         word TEXT UNIQUE,
            #                                         count INTEGER 
            #                                     ); """

            #     sql_create_2gram_table = """ CREATE TABLE IF NOT EXISTS _2_gram (

            #                                         word_1 TEXT ,
            #                                         word TEXT ,
            #                                         count INTEGER ,
            #                                         UNIQUE(word_1, word)
            #                                     ); """

            #     sql_create_3gram_table = """ CREATE TABLE IF NOT EXISTS _3_gram (
            #                                         word_2 TEXT ,
            #                                         word_1 TEXT ,
            #                                         word TEXT ,
            #                                         count INTEGER ,
            #                                         UNIQUE(word_2, word_1, word)
            #                                     ); """
            #     if conn_ngram is not None:
            #         c = conn_ngram.cursor()
            #         c.execute(sql_create_1gram_table)
            #         c.execute(sql_create_2gram_table)
            #         c.execute(sql_create_3gram_table)
            #     self.log.info("done create queries for personalized db")
            # except Exception as e:
            #     self.log.error(f"exception in creating personalized db : {e}")
            
            try:
                self.db.create_ngram_table(cardinality=1)
                self.db.create_ngram_table(cardinality=2)
                self.db.create_ngram_table(cardinality=3)
            except Exception as e:
                self.log.error(f"exception in creating personalized db : {e}")


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
                if(token.text.lower() not in self.stopwords and token.text.lower() not in imp_tokens):
                    imp_tokens.append(token.text.lower())
            # is this the object?
            if token.dep_ in self.OBJECT_DEPS or token.head.dep_ in self.OBJECT_DEPS:
                at.append(token.text)
                if(token.text.lower() not in self.stopwords and token.text.lower() not in imp_tokens):
                    imp_tokens.append(token.text.lower())
            # is this the subject?
            if token.dep_ in self.SUBJECT_DEPS or token.head.dep_ in self.SUBJECT_DEPS:
                sub.append(token.text)
                if(token.text.lower() not in self.stopwords and token.text.lower() not in imp_tokens):
                    imp_tokens.append(token.text.lower())
        return " ".join(imp_tokens).strip().lower()

    def is_question(self, doc):
        # is the first token a verb?
        if len(doc) > 0 and doc[0].pos_ == "VERB":
            return True, ""
        # go over all words
        for token in doc:
            # is it a wh- word?
            if token.tag_ in self.WH_WORDS:
                return True, token.text.lower()
        return False, ""

    def recreate_canned_db(self, personalized_corpus):
        ##### Check all phrases from the personalized corpus
        ##### if the phrase is found in the cannedSentences DB, continue,
        ##### Else, add it to both ngram and cannedSentences DB

        #### STEP1: CREATE CANNED_NGRAM DATABASE IF IT DOES NOT EXIST
        try:
            sql_create_1gram_table = """ CREATE TABLE IF NOT EXISTS _1_gram (

                                                word TEXT UNIQUE,
                                                count INTEGER 
                                            ); """

            sql_create_2gram_table = """ CREATE TABLE IF NOT EXISTS _2_gram (

                                                word_1 TEXT ,
                                                word TEXT ,
                                                count INTEGER ,
                                                UNIQUE(word_1, word)
                                            ); """

            sql_create_3gram_table = """ CREATE TABLE IF NOT EXISTS _3_gram (
                                                word_2 TEXT ,
                                                word_1 TEXT ,
                                                word TEXT ,
                                                count INTEGER ,
                                                UNIQUE(word_2, word_1, word)
                                            ); """

            self.log.info("executing create queries for canned_ngram db")
            self.db.open_database()
            self.db.execute_sql(sql_create_1gram_table)
            self.db.execute_sql(sql_create_2gram_table)
            self.db.execute_sql(sql_create_3gram_table)
            # self.db.close_database()
            self.log.info("done create queries for canned_ngram db")

            #### STEP1: CHECK IF THE PHRASE EXISTS IN THE CANNED_SENTENCES DATABASE
            sentence_db = SQLiteNgramDatabaseConnector(self.sentences_db)
            sentence_db.connect()
            # c = conn_sent.cursor()

            ###################### CHECK FOR PHRASES TO ADD AND PHRASES TO DELETE FROM THE DATABASES
            sent_db_dict = {}
            res = sentence_db.fetch_all("SELECT * FROM sentences")
            for r in res:
                sent_db_dict[r[0]]= r[1]
            phrases_toRemove = list(set(sent_db_dict.keys())-set(personalized_corpus))
            phrases_toAdd = list(set(personalized_corpus)-set(sent_db_dict.keys()))
            self.log.info("PHRASES TO ADD = " + str(phrases_toAdd))
            self.log.info("PHRASES TO REMOVE = "+ str(phrases_toRemove))


            ##### Add phrases_toAdd to the database and ngram
            for phrase in phrases_toAdd:
                # Add to 
                query = '''INSERT INTO sentences (sentence, count)
                                VALUES (?,?)'''
                phraseToInsert = (phrase, 1)
                # self.insert_into_tables(conn_sent, query,phraseToInsert)
                sentence_db.execute_query(query, phraseToInsert)

                ### Add phrase to ngram
                for curr_card in range(self.cardinality):
                    ngram_map = NgramMap()
                    ngs = self.generate_ngrams(phrase.lower().split(), curr_card)
                    ngram_map = self.getNgramMap(ngs, ngram_map)

                    # for every ngram, get db count, update or insert
                    for ngram, count in ngram_map.items():
                        old_count = self.db.ngram_count(ngram)
                        if old_count > 0:
                            self.db.update_ngram(ngram, old_count + count)
                            self.db.commit()
                        else:
                            self.db.insert_ngram(ngram, count)
                            self.db.commit()

            for phrase in phrases_toRemove:
            ##### Remove phrases_toRemove from the database
                query = 'DELETE FROM sentences WHERE sentence=?'
                # self.insert_into_tables(conn_sent, query,(phrase,))
                sentence_db.execute_query(query, (phrase,))
                self.log.info(f"Phrase {phrase} deleted from sentence_db.")
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
                        old_count = self.db.ngram_count(ngram)
                        if old_count > countToDelete:
                            self.db.update_ngram(ngram, old_count - countToDelete)
                            self.db.commit()
                        elif old_count == countToDelete:
                            self.db.remove_ngram(ngram)
                            self.db.commit()
                        elif old_count < countToDelete:
                            self.log.info("SmoothedNgramPredictor RecreateDB Delete function: Count in DB < count to Delete")

        except Exception as e:
            self.log.error(f"Error , Exception in SmoothedNgramPredictor recreateDB  = {e}")
        finally:
            sentence_db.close()

    def generate_ngrams(self, token, n):
        n = n+1
        # Use the zip function to help us generate n-grams
        # Concatentate the tokens into ngrams and return
        ngrams = zip(*[token[i:] for i in range(n)])
        returnobj = [" ".join(ngram) for ngram in ngrams]
        return returnobj

    def getNgramMap(self, ngs, ngram_map):
        for item in ngs:
            tokens = item.split(" ")
            ngram_list = []
            for token in tokens:
                idx = ngram_map.add_token(token)
                ngram_list.append(idx)
            ngram_map.add(ngram_list)
        return ngram_map

    # Property - deltas
    def deltas():
        doc = "The deltas property."

        def fget(self):
            return self._deltas

        def fset(self, value):
            self._deltas = []
            # make sure that values are floats
            for i, d in enumerate(value):
                self._deltas.append(float(d))
            self.cardinality = len(value)
            self.init_database_connector_if_ready()

        def fdel(self):
            del self._deltas

        return locals()

    deltas = property(**deltas())

    # Property - learn_mode
    def learn_mode():
        doc = "The learn_mode property."

        def fget(self):
            return self._learn_mode

        def fset(self, value):
            self._learn_mode = value
            self.learn_mode_set = True
            self.init_database_connector_if_ready()

        def fdel(self):
            del self._learn_mode

        return locals()

    learn_mode = property(**learn_mode())

    # Property - database
    def database():
        doc = "The database property."

        def fget(self):
            return self._database

        def fset(self, value):
            self._database = value
            self.init_database_connector_if_ready()

        def fdel(self):
            del self._database

        return locals()

    database = property(**database())

    def init_database_connector_if_ready(self):
        if (
                self.database
                and len(self.database) > 0
                and self.cardinality
                and self.cardinality > 0
                and self.learn_mode_set
        ):
            self.db = SQLiteNgramDatabaseConnector(self.database, 
                                              self.cardinality,
                                              self.log
                )
            self.db.connect()

    def predict(self, max_partial_prediction_size, filter):

        tokens = [""] * self.cardinality
        prediction = Prediction()
        try:
            ### For empty context, display the most frequent startwords 
            if(self.context_tracker.token(0)=="" and self.context_tracker.token(1)==""
                and self.context_tracker.token(2)=="" and self.name== PredictorNames.GeneralWord.value):
                f = open(self.startwords)
                self.precomputed_sentenceStart = json.load(f)
                for w, prob in self.precomputed_sentenceStart.items():
                    prediction.add_suggestion(
                            Suggestion(w, prob, self.name)
                        )

            for i in range(self.cardinality):
                # if self.context_tracker.token(i) != "":
                # tokens[self.cardinality - 1 - i] = json.dumps(self.context_tracker.token(i))
                tok = self.context_tracker.token(i)
                tokens[self.cardinality - 1 - i] = tok
            prefix_completion_candidates = []
            for k in reversed(range(self.cardinality)):
                if len(prefix_completion_candidates) >= max_partial_prediction_size:
                    break
                prefix_ngram = tokens[(len(tokens) - k - 1):]
                partial = None
                if not filter:
                    partial = self.db.ngram_like_table(
                        prefix_ngram,
                        max_partial_prediction_size - len(prefix_completion_candidates),
                    )
                else:
                    partial = self.db.ngram_like_table_filtered(
                        prefix_ngram,
                        filter,
                        max_partial_prediction_size - len(prefix_completion_candidates),
                    )

                for p in partial:
                    if len(prefix_completion_candidates) > max_partial_prediction_size:
                        break
                    candidate = p[-2]  # ???
                    if candidate not in prefix_completion_candidates:
                        prefix_completion_candidates.append(candidate)

            # smoothing
            unigram_counts_sum = self.db.unigram_counts_sum()
            for j, candidate in enumerate(prefix_completion_candidates):
                tokens[self.cardinality - 1] = candidate
                probability = 0
                for k in range(self.cardinality):
                    numerator = self._count(tokens, 0, k + 1)

                    denominator = unigram_counts_sum
                    if numerator > 0:
                        denominator = self._count(tokens, -1, k)
                    frequency = 0
                    if denominator > 0:
                        frequency = float(numerator) / denominator
                    probability += self.deltas[k] * frequency
                if probability > 0:
                    if all(char in string.punctuation for char in tokens[self.cardinality - 1]):
                        print(tokens[self.cardinality - 1]+ " contains punctuations ")
                        self.log.debug(tokens[self.cardinality - 1]+ " contains punctuations ")
                    else:
                        prediction.add_suggestion(
                            Suggestion(tokens[self.cardinality - 1], probability, self.name)
                        )
        except Exception as e:
            self.log.debug(f"Exception in SmoothedNgramPredictor predict function: {e}")

        return prediction

    def close_database(self):
        self.db.close()

    def _read_config(self):
        self.deltas = self.config.get(self.name, "deltas").split()
        self.static_resources_path = Path(self.config.get(self.name, "static_resources_path"))
        self.personalized_resources_path = Path(self.config.get(self.name, "personalized_resources_path"))
        self.learn_mode = self.config.get(self.name, "learn")
        self.stopwordsFile = os.path.join(self.static_resources_path, self.config.get(self.name, "stopwords"))
        if(self.name == PredictorNames.CannedWord.value ):
            self.sentences_db = os.path.join(self.personalized_resources_path, self.config.get(self.name, "sentences_db"))
            self.database = os.path.join(self.personalized_resources_path, self.config.get(self.name, "database"))
        if(self.name== PredictorNames.GeneralWord.value ):
            self.aac_dataset = os.path.join(self.static_resources_path, self.config.get(self.name, "aac_dataset"))
            self.log.debug("self.aac_dataset path = "+self.aac_dataset)
            self.database = os.path.join(self.static_resources_path, self.config.get(self.name, "database"))
            self.startwords = os.path.join(self.personalized_resources_path, self.config.get(self.name, "startwords"))
        if(self.name== PredictorNames.PersonalizedWord.value or self.name== PredictorNames.ShortHand.value):
            self.database = os.path.join(self.personalized_resources_path, self.config.get(self.name, "database"))

    def _count(self, tokens, offset, ngram_size):
        result = 0
        if ngram_size > 0:
            ngram = tokens[len(tokens) - ngram_size + offset: len(tokens) + offset]
            result = self.db.ngram_count(ngram)
        else:
            result = self.db.unigram_counts_sum()
        return result

    def learn(self, change_tokens):
        # build up ngram map for all cardinalities
        # i.e. learn all ngrams and counts in memory
        if self.learn_mode == "True":
            try:
                self.log.debug("learning ..."+ str(change_tokens))
                change_tokens = change_tokens.lower().translate(str.maketrans('', '', string.punctuation))
                self.log.debug("after removing punctuations, change_tokens = "+change_tokens)
                if(self.name == PredictorNames.CannedWord.value):
                    change_tokens = self.extract_svo(change_tokens)
                change_tokens = change_tokens.split()
                for curr_card in range(self.cardinality):
                    ngram_map = NgramMap()
                    ngs = self.generate_ngrams(change_tokens, curr_card)
                    # ngram_map = self.getNgramMap(ngs, ngram_map)
                    for item in ngs:
                        tokens = item.split(" ")
                        ngram_list = []
                        for token in tokens:
                            idx = ngram_map.add_token(token)
                            ngram_list.append(idx)
                        ngram_map.add(ngram_list)

                    # write this ngram_map to LM ...
                    # for every ngram, get db count, update or insert
                    for ngram, count in ngram_map.items():
                        old_count = self.db.ngram_count(ngram)
                        if old_count > 0:
                            self.db.update_ngram(ngram, old_count + count)
                            self.db.commit()
                        else:
                            self.db.insert_ngram(ngram, count)
                            self.db.commit()
            except Exception as e:
                self.log.error("Exception in SmoothedNgramPredictor learn function  = "+e)
        pass