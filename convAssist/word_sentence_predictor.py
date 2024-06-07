# Copyright (C) 2023 Intel Corporation


# SPDX-License-Identifier: Apache-2.0


"""
Classes for predictors and to handle suggestions and predictions.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

try:
    import configparser
except ImportError:
    import ConfigParser as configparser
from pathlib import Path
from tqdm import trange
import sys
import torch
import torch.nn.functional as F
import numpy as np
import os
import collections
import string
import hnswlib
import sqlite3
from sqlite3 import Error
from collections import Counter
import re
from enum import Enum
import regex as re
import json
import time
import spacy
import logging
from string import punctuation
import nltk
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk import sent_tokenize
import pandas as pd
import joblib
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")


### convAssist imports
import convAssist.dbconnector
import convAssist.combiner
from convAssist.tokenizer import NgramMap
from convAssist.logger import *

### imports related to sentence completions
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline, set_seed
# from optimum.onnxruntime import  ORTModelForCausalLM
# from optimum.pipelines import pipeline as onnxpipeline


try:
    #base_path = sys._MEIPASS
    #nlp = spacy.load(os.path.join(base_path, "en_core_web_sm", "en_core_web_sm-3.5.0"))
    base_path = sys._MEIPASS
    en_core_path = base_path+"/en_core_web_sm/"
    files = os.listdir(en_core_path)
    for file in files:
        if file.lower().startswith("en_core_web_sm"):
            file_path = os.path.join(en_core_path, file)
            nlp = spacy.load(file_path)

except Exception as e:
    nlp = spacy.load("en_core_web_sm")


MIN_PROBABILITY = 0.0
MAX_PROBABILITY = 1.0
global convAssistLog
convAssistLog = ConvAssistLogger("ConvAssist_Predictor_Log", "", logging.INFO)
convAssistLog.setLogger()


class PredictorNames(Enum):
    """
    Define names of all predictors
    """
    SentenceComp = "SentenceCompletionPredictor"
    CannedWord = "CannedWordPredictor" 
    GeneralWord = "DefaultSmoothedNgramPredictor"
    PersonalizedWord = "PersonalSmoothedNgramPredictor"
    Spell = "SpellCorrectPredictor"
    ShortHand = "ShortHandPredictor"
    CannedPhrases = "CannedPhrasesPredictor"


class SuggestionException(Exception):
    pass


class UnknownCombinerException(Exception):
    pass


class PredictorRegistryException(Exception):
    pass


class Suggestion(object):
    """
    Class for a simple suggestion, consists of a string and a probility for that
    string.

    """

    def __init__(self, word, probability, predictor_name):
        self.word = word
        self._probability = probability
        self.predictor_name = predictor_name

    def __eq__(self, other):
        if self.word == other.word and self.probability == other.probability:
            return True
        return False

    def __lt__(self, other):
        if self.probability < other.probability:
            return True
        if self.probability == other.probability:
            return self.word < other.word
        return False

    def __repr__(self):
        return "Word: {0} - Probability: {1}".format(self.word, self.probability)

    def probability():
        doc = "The probability property."

        def fget(self):
            return self._probability

        def fset(self, value):
            if(value> MAX_PROBABILITY):
                value = 1.0
            if value < MIN_PROBABILITY or value > MAX_PROBABILITY:
                raise SuggestionException("Probability is too high or too low = " + str(value))
            self._probability = value

        def fdel(self):
            del self._probability

        return locals()

    probability = property(**probability())


class Prediction(list):
    """
    Class for predictions from predictors.

    """

    def __init__(self):
        pass

    def __eq__(self, other):
        if self is other:
            return True
        if len(self) != len(other):
            return False
        for i, s in enumerate(other):
            if not s == self[i]:
                return False
        return True

    def suggestion_for_token(self, token):
        for s in self:
            if s.word == token:
                return s

    def add_suggestion(self, suggestion):
        if len(self) == 0:
            self.append(suggestion)
        else:
            i = 0
            while i < len(self) and suggestion < self[i]:
                i += 1

            self.insert(i, suggestion)


class PredictorActivator(object):
    """
    PredictorActivator starts the execution of the active predictors,
    monitors their execution and collects the predictions returned, or
    terminates a predictor's execution if it execedes its maximum
    prediction time.

    The predictions returned by the individual predictors are combined
    into a single prediction by the active Combiner.

    """

    def __init__(self, config, registry, context_tracker):
        self.config = config
        self.registry = registry
        self.context_tracker = context_tracker
        self.predictions = []
        self.word_predictions = []
        self.sent_predictions = []
        self.spell_word_predictions = []
        self.combiner = None
        self.max_partial_prediction_size = int(config.get("Selector", "suggestions"))
        self.predict_time = None
        self._combination_policy = None

    def combination_policy():
        doc = "The combination_policy property."

        def fget(self):
            return self._combination_policy

        def fset(self, value):
            self._combination_policy = value
            if value.lower() == "meritocracy":
                self.combiner = convAssist.combiner.MeritocracyCombiner()
            else:
                raise UnknownCombinerException()

        def fdel(self):
            del self._combination_policy

        return locals()

    combination_policy = property(**combination_policy())

    def predict(self, multiplier=1, prediction_filter=None):
        self.word_predictions[:] = []
        self.sent_predictions[:] = []
        self.spell_word_predictions[:] = []
        sent_nextLetterProbs = []
        sent_result = []
        word_result = []
        word_nextLetterProbs = []
        spell_word_result = []
        spell_word_nextLetterProbs = []
        context = self.context_tracker.token(0)
        for predictor in self.registry:

            if(predictor.name == PredictorNames.SentenceComp.value):                
                sentences = predictor.predict(self.max_partial_prediction_size * multiplier, prediction_filter)
                self.sent_predictions.append(sentences)
                sent_nextLetterProbs, sent_result = self.combiner.combine(self.sent_predictions, context)

            elif(predictor.name == PredictorNames.CannedPhrases.value):
                sentences, words = predictor.predict(self.max_partial_prediction_size * multiplier, prediction_filter)
                sent_result = sentences
                if(words!=[]):
                    for w in words:
                        self.word_predictions.append(w)
            ### If the predictor is spell predictor, use the predictions only if the other predictors return empty lists
            elif(predictor.name == PredictorNames.Spell.value):
                self.spell_word_predictions.append(predictor.predict(
                    self.max_partial_prediction_size * multiplier, prediction_filter)
                )
                spell_word_nextLetterProbs, spell_word_result = self.combiner.combine(self.spell_word_predictions, context)
            else:
                self.word_predictions.append(predictor.predict(
                    self.max_partial_prediction_size * multiplier, prediction_filter)
                )
                word_nextLetterProbs, word_result = self.combiner.combine(self.word_predictions, context)

        if(word_result==[]):
            word_result = spell_word_result
            word_nextLetterProbs = spell_word_nextLetterProbs
        result = (word_nextLetterProbs, word_result, sent_nextLetterProbs, sent_result)
        return result

    def recreate_canned_phrasesDB(self):
        for predictor in self.registry:
            if(predictor.name == PredictorNames.CannedPhrases.value or predictor.name == PredictorNames.CannedWord.value):
                personalized_resources_path = Path(self.config.get(predictor.name, "personalized_resources_path"))
                personalized_cannedphrases = os.path.join(personalized_resources_path, self.config.get(predictor.name, "personalized_cannedphrases"))
                pers_cannedphrasesLines = open(personalized_cannedphrases, "r").readlines()
                pers_cannedphrasesLines = [s.strip() for s in pers_cannedphrasesLines]

                predictor.recreate_canned_db(pers_cannedphrasesLines)

    def update_params(self, test_gen_sentence_pred,retrieve_from_AAC):
        convAssistLog.Log("INSIDE PREDICTOR ACTIVATOR update_params FUNCTION")
        for predictor in self.registry:
            predictor.load_model(test_gen_sentence_pred,retrieve_from_AAC)
    
    def read_updated_toxicWords(self):
        convAssistLog.Log("READING UPDATED PERSONALIZED TOXIC WORDS")
        for predictor in self.registry:
            predictor.read_personalized_toxic_words()

    def learn_text(self, text):
        for predictor in self.registry:
            predictor.learn(text)

    def set_log(self,filename, pathLoc, level):
        global convAssistLog
        if convAssistLog.IsLogInitialized():
            convAssistLog.Close()
        convAssistLog = None
        convAssistLog = ConvAssistLogger(filename, pathLoc, level)
        convAssistLog.setLogger()


class PredictorRegistry(list):  # pressagio.observer.Observer,
    """
    Manages instantiation and iteration through predictors and aids in
    generating predictions and learning.

    PredictorRegitry class holds the active predictors and provides the
    interface required to obtain an iterator to the predictors.

    The standard use case is: Predictor obtains an iterator from
    PredictorRegistry and invokes the predict() or learn() method on each
    Predictor pointed to by the iterator.

    Predictor registry should eventually just be a simple wrapper around
    plump.

    """

    def __init__(self, config, dbconnection=None):
        self.config = config
        self.dbconnection = dbconnection
        self._context_tracker = None
        self.set_predictors()

    def context_tracker():
        doc = "The context_tracker property."

        def fget(self):
            return self._context_tracker

        def fset(self, value):
            if self._context_tracker is not value:
                self._context_tracker = value
                self[:] = []
                self.set_predictors()

        def fdel(self):
            del self._context_tracker

        return locals()

    context_tracker = property(**context_tracker())

    def set_predictors(self):
        if self.context_tracker:
            self[:] = []
            for predictor in self.config.get("PredictorRegistry", "predictors").split():
                self.add_predictor(predictor)

    def add_predictor(self, predictor_name):
        predictor = None
        if (
                self.config.get(predictor_name, "predictor_class")
                == "SmoothedNgramPredictor"
        ):
            predictor = SmoothedNgramPredictor(
                self.config,
                self.context_tracker,
                predictor_name,
                dbconnection=self.dbconnection,
            )
        if (
                self.config.get(predictor_name, "predictor_class")
                == "SpellCorrectPredictor"
        ):
            predictor = SpellCorrectPredictor(
                self.config,
                self.context_tracker,
                predictor_name,
                dbconnection=self.dbconnection,
            )

        if (
                self.config.get(predictor_name, "predictor_class")
                == "SentenceCompletionPredictor"
        ):
            predictor = SentenceCompletionPredictor(self.config, self.context_tracker, predictor_name, "gpt2",
                                           "gpt-2 model predcitions", self.dbconnection)


        if (
                self.config.get(predictor_name, "predictor_class")
                == "CannedPhrasesPredictor"
        ):
            predictor = CannedPhrasesPredictor(self.config, self.context_tracker, predictor_name, "gpt2",
                                           "gpt-2 model predcitions", self.dbconnection)




        if predictor:
            self.append(predictor)
    
    def model_status(self):
        model_status = 999
        for each in self:
            if(str(each).find(PredictorNames.SentenceComp.value)!=-1):
                status = each.is_model_loaded()
                if(status):
                    convAssistLog.Log("SentenceCompletionPredictor model loaded")
                    model_status = 1
                else:
                    model_status = 0
            if(str(each).find(PredictorNames.CannedPhrases.value)!=-1):
                status = each.is_model_loaded()
                if(status):
                    model_status = 1
                    convAssistLog.Log("CannedPhrasesPredictor model loaded")
                else:
                    model_status = 0
        return model_status

    def close_database(self):
        for predictor in self:
            predictor.close_database()


class Predictor(object):
    """
    Base class for predictors.

    """

    def __init__(
            self, config, context_tracker, predictor_name, short_desc=None, long_desc=None
    ):
        self.short_description = short_desc
        self.long_description = long_desc
        self.context_tracker = context_tracker
        self.name = predictor_name
        self.config = config

    def token_satifies_filter(token, prefix, token_filter):
        if token_filter:
            for char in token_filter:
                candidate = prefix + char
                if token.startswith(candidate):
                    return True
        return False


class SpellCorrectPredictor(Predictor):  # , pressagio.observer.Observer
    """Spelling Corrector in Python 3; see http://norvig.com/spell-correct.html

    Copyright (c) 2007-2016 Peter Norvig
    MIT license: www.opensource.org/licenses/mit-license.php

    """

    def __init__(
            self,
            config,
            context_tracker,
            predictor_name,
            short_desc=None,
            long_desc=None,
            dbconnection=None,
    ):
        Predictor.__init__(
            self, config, context_tracker, predictor_name, short_desc, long_desc
        )
        self.db = None
        self.dbconnection = dbconnection
        self.cardinality = None
        self.learn_mode_set = False

        self.dbclass = None
        self.dbuser = None
        self.dbpass = None
        self.dbhost = None
        self.dbport = None

        self._database = None
        self._deltas = None
        self._learn_mode = None
        self.config = config
        self.name = predictor_name
        self.context_tracker = context_tracker
        self._read_config()

        self.WORDS = Counter(self.words(open(self.spellingDatabase).read()))

    def words(self, text): return re.findall(r'\w+', text.lower())

    def P(self, word): 
        """ Probability of `word`."""
        N=sum(self.WORDS.values())
        return self.WORDS[word] / N

    def correction(self, word):
        """Most probable spelling correction for word."""
        return max(self.candidates(word), key=self.P)

    def candidates(self, word):
        """Generate possible spelling corrections for word."""
        return (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word])

    def known(self, words):
        """The subset of `words` that appear in the dictionary of WORDS."""
        return set(w for w in words if w in self.WORDS)

    def edits1(self, word):
        """All edits that are one edit away from `word`."""
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        """All edits that are two edits away from `word`."""
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

    def predict(self, max_partial_prediction_size, filter):
        token = self.context_tracker.token(0)
        prediction = Prediction()
        prefix_completion_candidates = self.candidates(token)

        for j, candidate in enumerate(prefix_completion_candidates):
            probability = self.P(candidate)
            if probability > 0.0001:
                prediction.add_suggestion(
                    Suggestion(candidate, probability, self.name)
                )
        return prediction
    
    def learn(self, text):
        pass

    def _read_config(self):
        self.static_resources_path = Path(self.config.get(self.name, "static_resources_path"))
        self.spellingDatabase = os.path.join(self.static_resources_path, self.config.get(self.name, "spellingDatabase"))


class SmoothedNgramPredictor(Predictor):  # , pressagio.observer.Observer
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
    ):
        Predictor.__init__(
            self, config, context_tracker, predictor_name, short_desc, long_desc
        )
        self.db = None
        self.dbconnection = dbconnection
        self.cardinality = None
        self.learn_mode_set = False

        self.dbclass = None
        self.dbuser = None
        self.dbpass = None
        self.dbhost = None
        self.dbport = None

        self._database = None
        self._deltas = None
        self._learn_mode = None
        self.config = config
        self.name = predictor_name
        self.context_tracker = context_tracker
        self._read_config()
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
            convAssistLog.Log("INSIDE init "+PredictorNames.GeneralWord.value)
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
            convAssistLog.Log("INSIDE init "+PredictorNames.PersonalizedWord.value)
            try:
                convAssistLog.Log("trying to establish connection with "+ self.database)
                conn_ngram = self.create_connection(self.database)
                convAssistLog.Log("personalized database connection created "+ self.database)
                
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
                if conn_ngram is not None:
                    c = conn_ngram.cursor()
                    c.execute(sql_create_1gram_table)
                    c.execute(sql_create_2gram_table)
                    c.execute(sql_create_3gram_table)
                convAssistLog.Log("done create queries for personalized db")
            except Error as e:
                convAssistLog.Log("exception in creating personalized db : "+e)

    def create_connection(self, db_file):
        """ create a database connection to the SQLite database
            specified by db_file
        :param db_file: database file
        :return: Connection object or None
        """
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            return conn
        except Error as e:
            print("error  = ",e)

        return conn

    def insert_into_tables(self, conn, sql, task):

        cur = conn.cursor()
        cur.execute(sql, task)
        conn.commit()

        return cur.lastrowid

    def extract_svo(self, sent):
        doc = nlp(sent)
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

            convAssistLog.Log("executing create queries for canned_ngram db")
            self.db.open_database()
            self.db.execute_sql(sql_create_1gram_table)
            self.db.execute_sql(sql_create_2gram_table)
            self.db.execute_sql(sql_create_3gram_table)
            # self.db.close_database()
            print("done create queries for canned_ngram db")

            #### STEP1: CHECK IF THE PHRASE EXISTS IN THE CANNED_SENTENCES DATABASE
            conn_sent = sqlite3.connect(self.sentences_db) 
            c = conn_sent.cursor()

            ###################### CHECK FOR PHRASES TO ADD AND PHRASES TO DELETE FROM THE DATABASES
            sent_db_dict = {}
            c.execute("SELECT * FROM sentences ")
            res = c.fetchall()
            for r in res:
                sent_db_dict[r[0]]= r[1]
            phrases_toRemove = list(set(sent_db_dict.keys())-set(personalized_corpus))
            phrases_toAdd = list(set(personalized_corpus)-set(sent_db_dict.keys()))
            convAssistLog.Log("PHRASES TO ADD = " + str(phrases_toAdd))
            convAssistLog.Log("PHRASES TO REMOVE = "+ str(phrases_toRemove))

            
            ##### Add phrases_toAdd to the database and ngram
            for phrase in phrases_toAdd:
                query = '''INSERT INTO sentences (sentence, count)
                                VALUES (?,?)'''
                phraseToInsert = (phrase, 1)
                self.insert_into_tables(conn_sent, query,phraseToInsert)

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
                self.insert_into_tables(conn_sent, query,(phrase,))
                convAssistLog.Log("Phrase "+ phrase+ " deleted from sentence_db !!!!")
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
                            convAssistLog.Log("SmoothedNgramPredictor RecreateDB Delete function: Count in DB < count to Delete")
                            print("SmoothedNgramPredictor RecreateDB Delete function: Count in DB < count to Delete")

        except Error as e:
            convAssistLog.Log("Error , Exception in SmoothedNgramPredictor recreateDB  = "+e)

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

    def database():
        print("inside database defn")
        doc = "The database property."

        def fget(self):
            return self._database

        def fset(self, value):
            self._database = value

            self.dbclass = self.config.get("Database", "class")
            if self.dbclass == "PostgresDatabaseConnector":
                self.dbuser = self.config.get("Database", "user")
                self.dbpass = self.config.get("Database", "password")
                self.dbhost = self.config.get("Database", "host")
                self.dbport = self.config.get("Database", "port")
                self.dblowercase = self.config.getboolean("Database", "lowercase_mode")
                self.dbnormalize = self.config.getboolean("Database", "normalize_mode")

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
            if self.dbclass == "SqliteDatabaseConnector":
                self.db = convAssist.dbconnector.SqliteDatabaseConnector(
                    self.database, self.cardinality
                )  # , self.learn_mode
            elif self.dbclass == "PostgresDatabaseConnector":
                self.db = convAssist.dbconnector.PostgresDatabaseConnector(
                    self.database,
                    self.cardinality,
                    self.dbhost,
                    self.dbport,
                    self.dbuser,
                    self.dbpass,
                    self.dbconnection,
                )
                self.db.lowercase = self.dblowercase
                self.db.normalize = self.dbnormalize
                self.db.open_database()

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
                        convAssistLog.Log(tokens[self.cardinality - 1]+ " contains punctuations ")
                    else:
                        prediction.add_suggestion(
                            Suggestion(tokens[self.cardinality - 1], probability, self.name)
                        )
        except Error as e:
            convAssistLog.Log("Exception in SmoothedNgramPredictor predict function  = "+e)

        return prediction

    def close_database(self):
        self.db.close_database()

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
            convAssistLog.Log("self.aac_dataset path = "+self.aac_dataset)
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
                convAssistLog.Log("learning ..."+ str(change_tokens))
                change_tokens = change_tokens.lower().translate(str.maketrans('', '', string.punctuation))
                convAssistLog.Log("after removing punctuations, change_tokens = "+change_tokens)
                print("after removing punctuations, change_tokens = ", change_tokens)
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
            except Error as e:
                convAssistLog.Log("Exception in SmoothedNgramPredictor learn function  = "+e)


        pass


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)

        else:
            # print("args[2]", args[1]) #hack to pass the context_tracker that contains the updated context
            cls._instances[cls].context_tracker = args[1]
        return cls._instances[cls]


class SentenceCompletionPredictor(Predictor, metaclass=Singleton):  # , pressagio.observer.Observer
    """
    Calculates prediction from n-gram model using gpt-2.
    """

    def set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(seed)

    def __init__(self, config, context_tracker, predictor_name, short_desc=None, long_desc=None, dbconnection=None):

        Predictor.__init__(
            self, config, context_tracker, predictor_name, short_desc, long_desc
        )
        self.db = None
        self.dbconnection = dbconnection
        self.cardinality = 3
        self.learn_mode_set = False

        self.dbclass = None
        self.dbuser = None
        self.dbpass = None
        self.dbhost = None
        self.dbport = None

        self._database = None
        self._deltas = None
        self._learn_mode = None
        self.config = config
        self.name = predictor_name
        self.context_tracker = context_tracker
        self._read_config()
        self.MODEL_LOADED = False
        self.corpus_sentences=[]
        ################ check if saved torch model exists  
        self.load_model(self.test_generalSentencePrediction, self.retrieve)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()
        self.stemmer = PorterStemmer()
        ####### CREATE INDEX TO QUERY DATABASE
        self.embedder = SentenceTransformer(self.sentence_transformer_model)
        self.embedding_size = 384    #Size of embeddings
        self.top_k_hits = 2       #Output k hits
        self.n_clusters = 350
        #We use Inner Product (dot-product) as Index. 
        #We will normalize our vectors to unit length, then is Inner Product equal to cosine similarity
        self.index = hnswlib.Index(space = 'cosine', dim = self.embedding_size)

        self.corpus_sentences = open(self.retrieve_database).readlines()
        self.corpus_sentences = [s.strip() for s in self.corpus_sentences]

        self.blacklist_words = open(self.blacklist_file).readlines()
        self.blacklist_words = [s.strip() for s in self.blacklist_words]

        self.personalized_allowed_toxicwords = self.read_personalized_toxic_words()


        self.OBJECT_DEPS = {"dobj","pobj", "dative", "attr", "oprd", "npadvmod", "amod","acomp","advmod"}
        self.SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "agent", "expl"}
        # tags that define wether the word is wh-
        self.WH_WORDS = {"WP", "WP$", "WRB"}
        self.stopwords = []
        stoplist = open(self.stopwordsFile,"r").readlines()
        for s in stoplist:
            self.stopwords.append(s.strip())

        if(not os.path.isfile(self.embedding_cache_path)):
            self.corpus_embeddings = self.embedder.encode(self.corpus_sentences, show_progress_bar=True, convert_to_numpy=True)
            joblib.dump({'sentences': self.corpus_sentences, 'embeddings': self.corpus_embeddings}, self.embedding_cache_path)
            # np.save(self.embedding_cache_path,{'sentences': self.corpus_sentences, 'embeddings': self.corpus_embeddings})

        else:
            # cache_data = np.load(self.embedding_cache_path)
            cache_data = joblib.load(self.embedding_cache_path)
            self.corpus_sentences = cache_data['sentences']
            self.corpus_embeddings = cache_data['embeddings']



        # if not os.path.exists(self.embedding_cache_path):
        #     convAssistLog.Log(" embeddings do not exist, creating")
        #     self.corpus_embeddings = self.embedder.encode(self.corpus_sentences, show_progress_bar=True, convert_to_numpy=True)
        #     with open(self.embedding_cache_path, "wb") as fOut:
        #         pickle.dump({'sentences': self.corpus_sentences, 'embeddings': self.corpus_embeddings}, fOut)
        # else:
        #     convAssistLog.Log(" found "+self.embedding_cache_path)
        #     with open(self.embedding_cache_path, "rb") as fIn:
        #         cache_data = pickle.load(fIn)
        #         self.corpus_sentences = cache_data['sentences']
        #         self.corpus_embeddings = cache_data['embeddings']

        ###### LOAD INDEX IF EXISTS, ELSE CREATE INDEX 
        if os.path.exists(self.index_path):
            convAssistLog.Log("Loading index...")
            self.index.load_index(self.index_path)

        else:
            ## creating the embeddings pkl file
            convAssistLog.Log(" index does not exist, creating index")



            ### Create the HNSWLIB index
            convAssistLog.Log("Start creating HNSWLIB index")
            self.index.init_index(max_elements = 20000, ef_construction = 400, M = 64)

            # Then we train the index to find a suitable clustering
            self.index.add_items(self.corpus_embeddings, list(range(len(self.corpus_embeddings))))

            convAssistLog.Log("Saving index to:"+ self.index_path)
            self.index.save_index(self.index_path)
        # Controlling the recall by setting ef:
        self.index.set_ef(50)  # ef should always be > top_k_hits

        if(not os.path.isfile(self.sent_database)) :
            convAssistLog.Log(self.sent_database+" not found, creating it")
            self.createSentDB(self.sent_database)

        # if(not os.path.isfile(self.startsents)):
        #     convAssistLog.Log(self.startsents+" not found, creating it")
        #     self.writeTopSent(self.startsents)
    '''
    def writeTopSent(self, startsentFile):
        lines = open(self.retrieve_database, "r").readlines()
        retrieved = []
        totalsent= len(lines)
        for each in lines:
            each = re.split('[.\n?!]',each)[0]
            retrieved.append(each)
        retrieved_set = set(retrieved)
        convAssistLog.Log("len(retrieved_set = "+str(len(retrieved_set))+ "len(retrieved) = "+ str(len(retrieved)))
        probs = {}
        for s in retrieved_set:
            probs[s] = float(retrieved.count(s))/totalsent

        #### RETRIVE TOP SENTENCES FROM PERSONALIZED SENTENCE DATABASE
        try:
            pers_results = {}
            conn = sqlite3.connect(self.sent_database) 
            c = conn.cursor()
            count = 0
            #### CHECK IF SENTENCE EXISITS IN THE DATABASE
            c.execute("SELECT * FROM sentences")
            res = c.fetchall()
            convAssistLog.Log("Pringing sentence_db results, SentenceCompletionPred, writetopSent function= "+ str(res))
            for r in res:
                pers_results[r[0]] = r[1]
            total_sent = sum(pers_results.values())
            for k,v in pers_results.items():
                probs[k] = float(v/total_sent)

            sorted_x = collections.OrderedDict(sorted(probs.items(), key=lambda kv: kv[1], reverse=True))
            import itertools
            out = dict(itertools.islice(sorted_x.items(), 5))
            with open(startsentFile, 'w') as fp:
                json.dump(out, fp)
        except:
            convAssistLog.Log("Exception in SentenceCompletionPredictor, writeTopSent function  = "+e)
    '''

    def read_personalized_toxic_words(self):
        if not os.path.exists(self.personalized_allowed_toxicwords_file):
            f = open(self.personalized_allowed_toxicwords_file, "w")
            f.close()
        self.personalized_allowed_toxicwords = open(self.personalized_allowed_toxicwords_file, "r").readlines()
        self.personalized_allowed_toxicwords = [s.strip() for s in self.personalized_allowed_toxicwords]
        convAssistLog.Log("UPDATED TOXIC WORDS = "+ str(self.personalized_allowed_toxicwords))
        return self.personalized_allowed_toxicwords

    def extract_svo(self, sent):
        doc = nlp(sent)
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
        return imp_tokens


    def createSentDB(self, dbname):
        convAssistLog.Log("IN createSentDB")
        try:
            convAssistLog.Log("creating sentence_db = "+ dbname)
            conn = sqlite3.connect(dbname) 
            c = conn.cursor()
            c.execute('''
                    CREATE TABLE IF NOT EXISTS sentences
                    (sentence TEXT UNIQUE, count INTEGER)
                    ''')        
            conn.commit()
        except Error as e:
            convAssistLog.Log("Exception in SentenceCompletionPredictor, createSentDB  = "+e)

    def load_model(self, test_generalSentencePrediction, retrieve): 
        self.test_generalSentencePrediction = test_generalSentencePrediction
        self.retrieve = retrieve
        convAssistLog.Log("INSIDE SentenceCompletionPredictor LOAD MODEL:"+ str(os.path.exists(self.modelname)))
        #### if we are only testing the models
        if(self.test_generalSentencePrediction=="True"):
            if(self.use_onnx_model=="True" and os.path.exists(self.onnx_path)):
                convAssistLog.Log("No support for ONNX model ")
                # convAssistLog.Log("Loading onnx model from "+self.onnx_path)
                # model = ORTModelForCausalLM.from_pretrained(self.onnx_path, file_name="decoder_model.onnx")
                # tokenizer = AutoTokenizer.from_pretrained(self.onnx_path)
                # self.generator = onnxpipeline("text-generation", model=model, tokenizer=tokenizer)
                self.MODEL_LOADED = True

            elif(self.use_onnx_model=="False" and os.path.exists(self.modelname)):
                convAssistLog.Log("Loading gpt2 model from "+str(self.modelname))
                self.generator = pipeline('text-generation', model=self.modelname, tokenizer=self.tokenizer)
                self.MODEL_LOADED = True

        else:
            if(self.retrieve=="False"):
                convAssistLog.Log("RETRIEVE IS FALSE, loading model = "+self.modelname)
                if(self.use_onnx_model=="True" and os.path.exists(self.onnx_path)):
                    convAssistLog.Log("No support for ONNX model ")
                    # model = ORTModelForCausalLM.from_pretrained(self.onnx_path, file_name="decoder_model_quantized.onnx")
                    # tokenizer = AutoTokenizer.from_pretrained(self.onnx_path)
                    # self.generator = onnxpipeline("text-generation", model=model, tokenizer=tokenizer)
                    self.MODEL_LOADED = True
                elif(self.use_onnx_model=="False" and os.path.exists(self.modelname)):
                    convAssistLog.Log("Loading gpt2 model from "+self.modelname)
                    self.generator = pipeline('text-generation', model=self.modelname, tokenizer=self.tokenizer)
                    self.MODEL_LOADED = True
            elif(self.retrieve=="True"):
                self.MODEL_LOADED = True

        convAssistLog.Log("self.MODEL_LOADED = "+ str(self.MODEL_LOADED))
        
    def is_model_loaded(self):
        return self.MODEL_LOADED

    def ngram_to_string(self, ngram):
        "|".join(ngram)

    def filter_text(self, text):
        res = False
        words = []
        toxicwordsList = list(set(self.blacklist_words) - set(self.personalized_allowed_toxicwords))

        if(any(x in text.lower().split() for x in toxicwordsList)):
            words = list(set(text.lower().split()) & set(toxicwordsList))
            # print("blacklisted word is present!!", set(text.split()) & set(self.blacklist_words))
            res = True
        return (res, words)

    def textInCorpus(self, text):
        query_embedding = self.embedder.encode(text)
        
        #We use hnswlib knn_query method to find the top_k_hits
        corpus_ids, distances = self.index.knn_query(query_embedding, k=self.top_k_hits)
        hits = [{'corpus_id': id, 'score': 1-score} for id, score in zip(corpus_ids[0], distances[0])]
        hits = sorted(hits, key=lambda x: x['score'], reverse=True)
        # if(hits[0]['score']>0.6):
        #     print(hits[0]['score'], self.corpus_sentences[hits[0]['corpus_id']])
        #     return hits[0]['score']
        print(hits[0]['score'], hits[0]['corpus_id'], len(self.corpus_sentences), self.corpus_embeddings.shape)
        print("text = ", text, " score = ", hits[0]['score'], " sentence = ", self.corpus_sentences[hits[0]['corpus_id']])
        return hits[0]['score']
        
    def retrieve_fromDataset(self, context):
        pred = Prediction()
        probs = {}

        lines = open(self.retrieve_database, "r").readlines()
        retrieved = []
        totalsent= len(lines)
        for each in lines:
            # if(each.lower().find(context.lower())!=-1):
            if(each.lower().startswith(context.lower())):
                each = re.split('[.\n?!]',each)[0]
                retrieved.append(each)
        retrieved_set = set(retrieved)
        convAssistLog.Log("len(retrieved_set = "+str(len(retrieved_set))+ "len(retrieved) = "+ str(len(retrieved)))
        for s in retrieved_set:
            probs[s] = float(retrieved.count(s))/totalsent
        try:
            pers_results = {}
            totalsentences = 0
            conn = sqlite3.connect(self.sent_database) 
            c = conn.cursor()
            count = 0
            #### CHECK IF SENTENCE EXISITS IN THE DATABASE
            c.execute("SELECT * FROM sentences")
            res = c.fetchall()
            for r in res:
                sent = r[0]
                totalsentences = totalsentences+r[1]
                if(sent.lower().startswith(context.lower())):
                    sent = re.split('[.\n?!]',sent)[0]
                    if(sent in pers_results):
                        pers_results[sent] = pers_results[sent]+r[1]
                    else:
                        pers_results[sent] = r[1]
            for k,v in pers_results.items():
                probs[k] = float(v/totalsentences)


            sorted_x = collections.OrderedDict(sorted(probs.items(), key=lambda kv: kv[1], reverse=True))
            count = 0
            addedcompletions = []
            for k,v in sorted_x.items():
                if(count > 5):
                    break
                k = k[len(context):]
                if (not k in addedcompletions):
                    pred.add_suggestion(Suggestion(k, v, self.name))
                    addedcompletions.append(k)
                    count = count+1
        except Error as e:
            convAssistLog.Log("Exception in SentenceCompletionPredictor, retrieveFromDataset function  = "+e)

        return pred

    def checkRepetition(self, text):
        tokens = nltk.word_tokenize(text)

        #Create bigrams and check if there are any repeititions. 
        bgs = nltk.bigrams(tokens)
        #compute frequency distribution for all the bigrams in the text
        fdist = nltk.FreqDist(bgs)
        fdist = {k:v for (k,v) in fdist.items() if v >= 2}
        if (fdist!={}):
            return True

        #Create trigrams and check if there are any repeititions. 
        bgs = nltk.trigrams(tokens)
        #compute frequency distribution for all the trigrams in the text
        fdist = nltk.FreqDist(bgs)
        fdist = {k:v for (k,v) in fdist.items() if v >= 2}
        if (fdist!={}):
            return True        
        return False

    def generate(self, context, num_gen, predi):
        try:
            start = time.perf_counter()
            out = self.generator(context, do_sample=False, max_new_tokens=20, num_return_sequences=10, num_beams = 10, num_beam_groups=10, diversity_penalty=1.5, repetition_penalty = 1.1) 
            
            inputContext = context
            allsent = []
            probability = 1/len(out)
            counts= {}
            totalsent = 0
            if(num_gen<5):
                num_gen = 5
            num_gen = 10
            inputContext = inputContext.replace("<bos> ","")
            contextList = sent_tokenize(inputContext)
            num_context_sent = len(contextList)
            for o in out:
                print(o["generated_text"])
                gentext = o["generated_text"]
                newgen = re.split(r'<bos> |<eos> |bos|eos|<bos>|<eos>|<|>|\[|\]|\d',gentext)
                # print("Full generated Text = "+newgen[1])
                gen_text_sent = sent_tokenize(newgen[1])
                currSentence = gen_text_sent[num_context_sent-1]
                
                ### check for repetitive sentences
                if (self.checkRepetition(currSentence)):
                    convAssistLog.Log("REPETITION!!!!!!! in the sentence:  "+currSentence)
                    continue;


                # print("cursetnence = ", currSentence)
                reminderText = currSentence[len(contextList[-1]):]
                reminderTextForFilter = re.sub(r'[?,!.\n]', '', reminderText.strip())
                # print("remindertext - sending to filter = ", reminderTextForFilter)
                if(self.filter_text(reminderTextForFilter)[0]!=True):
                    reminderText = re.sub(r'[?!.\n]', '', reminderText.strip())
                    score = self.textInCorpus(currSentence.strip())

                    ################ TODO: DO WE THRESHOLD SCORES?
                    ########### TODO: DETOXIFY
                    # convAssistLog.Log("reminderTExt = "+reminderText+ " currentSentence = "+currSentence+" , score = "+str(score))
                    if reminderText!='':
                        if(currSentence not in allsent):
                            imp_tokens = self.extract_svo(currSentence)
                            imp_tokens_reminder = []
                            #### get important tokens only of the generated completion
                            for imp in imp_tokens:
                                if imp in word_tokenize(reminderText):
                                    imp_tokens_reminder.append(imp)
                            present = False
                            for a in allsent:
                                for it in imp_tokens_reminder:
                                    if(self.stemmer.stem(it) in [self.stemmer.stem(w) for w in word_tokenize(a[len(contextList[-1]):])]):
                                        present = True
                                        break
                            if(present==False):
                                allsent.append(currSentence)
                                counts[reminderText] = 1*score
                                totalsent = totalsent + 1 
                        else:
                            counts[reminderText] = counts[reminderText]+1*score
                            totalsent = totalsent + 1 

            # toxic_filtered_sent = self.detoxify(allsent)
            # print(toxic_filtered_sent)
            for k, v in counts.items():
                counts[k] = float(v)/totalsent

            sorted_x = collections.OrderedDict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))
            count = 0
            for k,v in sorted_x.items():
                
                if(count==num_gen):
                    break
                convAssistLog.Log("sentence = "+ k+ " score = "+ str(v))
                predi.add_suggestion(Suggestion(k, v, self.name))
                count = count+1

            convAssistLog.Log("latency in generation = "+ str(time.perf_counter()-start))
        except Error as e:
            convAssistLog.Log("Exception in SentenceCompletionPredictor, generate function  = "+e)

        return predi

    def predict(self, max_partial_prediction_size, filter):
        tokens = [""] * self.cardinality
        prediction = Prediction()
        context = self.context_tracker.past_stream().lstrip()
        if(context == "" or context==" "):
            convAssistLog.Log("context is empty, loading top sentences from "+self.startsents)
            if(not os.path.isfile(self.startsents)):
                convAssistLog.Log(self.startsents+" not found!!!")
                # self.writeTopSent(self.startsents)
            ##### retrieve top5 from startsentFile 
            data = open(self.startsents,"r").readlines()
            for k in data:
                prediction.add_suggestion(Suggestion(k.strip(), float(1/len(data)), self.name))
            return prediction
        start = time.perf_counter()
        convAssistLog.Log("context inside predictor predict = "+ context)
        #### If we are testing generation models
        if (self.test_generalSentencePrediction == "True"):
            if(self.use_onnx_model=="True" or str(self.modelname).find("DialoGPT")!=-1):
                prediction = self.generate(context.strip(),5,prediction)
            else:
                prediction = self.generate("<bos> "+context.strip(),5, prediction)

        
        #### if we want to Only retrieve from AAC dataset
        elif(self.retrieve=="True"):
            convAssistLog.Log("retireve is True - retrieving from database")
            prediction = self.retrieve_fromDataset(context)

        #### Hybrid retrieve mode  elif(self.retrieve=="hybrid"):
        elif(self.retrieve=="False"):
            convAssistLog.Log("Hybrid retrieval - AAC dataset + model generation")
            prediction = self.retrieve_fromDataset(context)
            convAssistLog.Log("retrieved "+ str(len(prediction))+ " sentences = "+str(prediction))
            ##### ONLY IF THE GENERATION MODEL IS LOADED, GENERATE MODEL BASED PREDICTIONS
            if(len(prediction)<5 and self.MODEL_LOADED):
                convAssistLog.Log("generating "+ str(5-len(prediction))+ " more predictions")
                if(self.use_onnx_model=="True" or str(self.modelname).find("DialoGPT")!=-1):
                    prediction = self.generate(context.strip(),5-len(prediction), prediction)
                else:
                    prediction = self.generate("<bos> "+context.strip(),5-len(prediction), prediction)
        latency = time.perf_counter() - start 
        convAssistLog.Log("latency = "+str(latency))
        print("latency = ", latency)
        convAssistLog.Log("prediction = "+ str(prediction))
        return prediction

    def close_database(self):
        self.db.close_database()

    def learn(self, change_tokens):
        #### For the sentence completion predictor, learning adds the sentence to the database
        if self.learn_mode == "True":
            change_tokens = change_tokens.strip()
            convAssistLog.Log("learning, "+str(change_tokens))
            #### add to sentence database
            try:
                conn = sqlite3.connect(self.sent_database) 
                c = conn.cursor()
                count = 0
                #### CHECK IF SENTENCE EXISITS IN THE DATABASE
                c.execute("SELECT count FROM sentences WHERE sentence = ?", (change_tokens,))
                res = c.fetchall()
                if len(res) > 0:
                    if len(res[0]) > 0:
                        count = int(res[0][0])

                ### IF SENTENCE DOES NOT EXIST, ADD INTO DATABASE WITH COUNT = 1
                if count==0:
                    convAssistLog.Log("count is 0, inserting into database")
                    c.execute('''
                    INSERT INTO sentences (sentence, count)
                    VALUES (?,?)''', (change_tokens, 1))

                    ### update retrieval index: 
                    # self.index.load_index(self.index_path)
                    print("shape before: ", self.corpus_embeddings.shape, "len*self.corpus_sentences =  ", len(self.corpus_sentences))
                    
                    convAssistLog.Log("sentence  "+ change_tokens+ " not present, adding to embeddings and creating new index")
                    print("sentence  "+ change_tokens+ " not present, adding to embeddings and creating new index")
                    phrase_emb = self.embedder.encode(change_tokens.strip())
                    phrase_id = len(self.corpus_embeddings)
                    self.corpus_embeddings = np.vstack((self.corpus_embeddings, phrase_emb))
                    self.corpus_sentences.append(change_tokens.strip())
                    # np.save(self.embedding_cache_path,{'sentences': self.corpus_sentences, 'embeddings': self.corpus_embeddings})
                    joblib.dump({'sentences': self.corpus_sentences, 'embeddings': self.corpus_embeddings}, self.embedding_cache_path)
                    # with open(self.embedding_cache_path, "wb") as fOut:
                    #     pickle.dump({'sentences': self.corpus_sentences, 'embeddings': self.corpus_embeddings}, fOut)
                    
                    # Then we train the index to find a suitable clustering
                    print("phrase_emb.shape = ", phrase_emb.shape, " id= ", len(self.corpus_embeddings))
                    self.index.add_items(phrase_emb, phrase_id)

                    convAssistLog.Log("Saving index to:"+ self.index_path)
                    self.index.save_index(self.index_path)
                    print("shape after: ", self.corpus_embeddings.shape, "len*self.corpus_sentences =  ", len(self.corpus_sentences))


                    #### DEALING WITH PERSONALIZED, ALLOWED TOXIC WORDS
                    #### if sentence to be learnt contains a toxic word, add the toxic word to the "allowed" word list
                    res, words = self.filter_text(change_tokens)
                    if(res==True):
                        for tox in words:
                            convAssistLog.Log("toxic words to be added to personalized db: "+tox)
                            if(tox not in self.personalized_allowed_toxicwords):
                                self.personalized_allowed_toxicwords.append(tox)
                                fout = open(self.personalized_allowed_toxicwords_file, "w")
                                for tox_word in self.personalized_allowed_toxicwords:
                                    fout.write(tox_word+"\n")
                                fout.close()

                ### ELSE, IF SENTENCE EXIST, ADD INTO DATABASE WITH UPDATED COUNT
                else:
                    convAssistLog.Log("sentence exists, updating count")
                    c.execute('''
                    UPDATE sentences SET count = ? where sentence = ?''', (count+1, change_tokens))
                conn.commit()
            except Error as e:
                convAssistLog.Log("Exception in SentenceCompletionPredictor learn  = "+str(e))

    def _read_config(self):
        self.static_resources_path = Path(self.config.get(self.name, "static_resources_path"))
        self.personalized_resources_path = Path(self.config.get(self.name, "personalized_resources_path"))
        self.learn_mode = self.config.get(self.name, "learn")
        self.retrieve = self.config.get(self.name, "retrieveAAC")
        self.use_onnx_model = self.config.get(self.name,"use_onnx_model")
        self.test_generalSentencePrediction = self.config.get(self.name,"test_generalSentencePrediction")
        #### define static databases, models

        self.modelname = os.path.join(self.static_resources_path, self.config.get(self.name, "modelname"))
        self.tokenizer = os.path.join(self.static_resources_path, self.config.get(self.name, "tokenizer"))
        self.retrieve_database = os.path.join(self.static_resources_path, self.config.get(self.name, "retrieve_database"))
        self.onnx_path = os.path.join(self.static_resources_path, self.config.get(self.name,"onnx_path"))                             
        self.sentence_transformer_model = os.path.join(self.static_resources_path,self.config.get(self.name,"sentence_transformer_model"))
        self.blacklist_file = os.path.join(self.static_resources_path,self.config.get(self.name,"blacklist_file"))
        self.stopwordsFile = os.path.join(self.static_resources_path,self.config.get(self.name,"stopwords"))
        
        #### define personalized databases

        self.sent_database = os.path.join(self.personalized_resources_path, self.config.get(self.name, "sent_database"))
        self.startsents = os.path.join(self.personalized_resources_path, self.config.get(self.name,"startsents"))
        self.embedding_cache_path = os.path.join(self.personalized_resources_path,self.config.get(self.name,"embedding_cache_path"))
        self.index_path = os.path.join(self.personalized_resources_path,self.config.get(self.name, "index_path"))
        self.personalized_allowed_toxicwords_file = os.path.join(self.personalized_resources_path,self.config.get(self.name, "personalized_allowed_toxicwords_file"))

        convAssistLog.Log("SENTENCE MODE CONFIGURATIONS")
        convAssistLog.Log("using Onnx model for inference = "+self.use_onnx_model)
        convAssistLog.Log("test_generalSentencePrediction = "+self.test_generalSentencePrediction)
        convAssistLog.Log("model = "+ str(self.modelname)+ "tokenizer = "+ str(self.tokenizer))

class CannedPhrasesPredictor(Predictor, metaclass=Singleton):  # , pressagio.observer.Observer
    """
    Searches the canned phrase database for matching next words and sentences
    """
    def __init__(self, config, context_tracker, predictor_name, short_desc=None, long_desc=None, dbconnection=None):
        Predictor.__init__(
            self, config, context_tracker, predictor_name, short_desc, long_desc
        )
        self.db = None
        self.dbconnection = dbconnection
        self.cardinality = 3
        self.learn_mode_set = False
        self.dbclass = None
        self.dbuser = None
        self.dbpass = None
        self.dbhost = None
        self.dbport = None
        self.MODEL_LOADED = False
        self._database = None
        self._deltas = None
        self._learn_mode = None
        self.config = config
        self.name = predictor_name
        self.context_tracker = context_tracker
        self._read_config()
        self.seed = 42
        self.cannedPhrases_counts={}
        self.stemmer = PorterStemmer()
        self.embedder = SentenceTransformer(self.sbertmodel)
        self.pers_cannedphrasesLines = open(self.personalized_cannedphrases, "r").readlines()
        self.pers_cannedphrasesLines = [s.strip() for s in self.pers_cannedphrasesLines]
            
        convAssistLog.Log("Logging inside canned phrases init!!!")
        if(not os.path.isfile(self.sentences_db)):
            self.createSentDB(self.sentences_db)
        

        if(not os.path.isfile(self.embedding_cache_path)):
            self.corpus_embeddings = self.embedder.encode(self.pers_cannedphrasesLines, show_progress_bar=True, convert_to_numpy=True)
            # np.save(self.embedding_cache_path,{'sentences': self.pers_cannedphrasesLines, 'embeddings': self.corpus_embeddings})
            joblib.dump({'sentences': self.pers_cannedphrasesLines, 'embeddings': self.corpus_embeddings},self.embedding_cache_path)

        else:
            # cache_data = np.load(self.embedding_cache_path)

            cache_data = joblib.load(self.embedding_cache_path)
            self.corpus_sentences = cache_data['sentences']
            self.corpus_embeddings = cache_data['embeddings']



        # #### IF CANNED PHRASES EMBEDDING NOT FOUND, 
        # if(not os.path.isfile(self.embedding_cache_path)):
        #     self.corpus_embeddings = self.embedder.encode(self.pers_cannedphrasesLines, show_progress_bar=True, convert_to_numpy=True)
        #     with open(self.embedding_cache_path, "wb") as fOut:
        #         pickle.dump({'sentences': self.pers_cannedphrasesLines, 'embeddings': self.corpus_embeddings}, fOut)

        # else:
        #     with open(self.embedding_cache_path, "rb") as fIn:
        #         cache_data = pickle.load(fIn)
        #         self.corpus_embeddings = cache_data['embeddings']
        #         self.corpus_sentences = cache_data['sentences']

        self.n_clusters = 20     ### clusters for hnswlib index
        self.embedding_size = self.corpus_embeddings.shape[1]
        self.index = hnswlib.Index(space = 'cosine', dim = self.embedding_size)

        ####### CHECK IF INDEX IS PRESENT
        if os.path.exists(self.index_path):
            convAssistLog.Log("Loading index at ..."+ self.index_path)
            self.index.load_index(self.index_path)
        else:
            ### Create the HNSWLIB index
            convAssistLog.Log("Start creating HNSWLIB index")
            self.index.init_index(max_elements = 10000, ef_construction = 400, M = 64)
            self.index = self.create_index(self.index)
        self.index.set_ef(50)

        self.MODEL_LOADED = True

    def create_index(self, ind):
        ind.add_items(self.corpus_embeddings, list(range(len(self.corpus_embeddings))))
        convAssistLog.Log("Saving index to:"+ self.index_path)
        ind.save_index(self.index_path)
        return ind
        
    def is_model_loaded(self):
        return self.MODEL_LOADED

    def insert_into_tables(self, conn, sql, task):
        cur = conn.cursor()
        cur.execute(sql, task)
        conn.commit()

        return cur.lastrowid

    def recreate_canned_db(self, personalized_corpus):
        convAssistLog.Log("inside CannedPhrasesPredictor recreate_canned_db")
        
        self.corpus_sentences = []
        self.pers_cannedphrasesLines = open(self.personalized_cannedphrases, "r").readlines()
        self.pers_cannedphrasesLines = [s.strip() for s in self.pers_cannedphrasesLines]

        try:
            #### RETRIEVE ALL SENTENCES FROM THE DATABASE
            conn_sent = sqlite3.connect(self.sentences_db) 
            c = conn_sent.cursor()
            c.execute("SELECT * FROM sentences")
            res_all = c.fetchall()
            convAssistLog.Log("len(sent_db) = "+ str(len(res_all))+ " len(self.pers_cannedphrases) = "+ str(len(self.pers_cannedphrasesLines)))
            for r in res_all:
                self.cannedPhrases_counts[r[0]] = r[1]
            
            if(os.path.isfile(self.embedding_cache_path)):
                # cache_data = np.load(self.embedding_cache_path)
                cache_data = joblib.load(self.embedding_cache_path)
                self.corpus_sentences = cache_data['sentences']
                self.corpus_embeddings = cache_data['embeddings']


            # if(os.path.isfile(self.embedding_cache_path)):
            #     with open(self.embedding_cache_path, "rb") as fIn:
            #         cache_data = pickle.load(fIn)
            #         self.corpus_embeddings = cache_data['embeddings']
            #         self.corpus_sentences = cache_data['sentences']
            else:
                convAssistLog.Log("In Recreate_DB of cannedPhrasesPredictor, EMBEDDINGS FILE DOES NOT EXIST!!! ")
        

            ###### check if cannedPhrases file has been modified!!! 
            if(set(self.corpus_sentences) != set(self.pers_cannedphrasesLines) ):
                convAssistLog.Log("Canned Phrases has been modified externally.. Recreating embeddings and indices")
                phrasesToAdd = set(self.pers_cannedphrasesLines) - set(self.corpus_sentences)
                phrasesToRemove = set(self.corpus_sentences) - set(self.pers_cannedphrasesLines)
                print("phrases to add = ", str(phrasesToAdd))
                print("phrases to remove = ", str(phrasesToRemove))
                convAssistLog.Log("phrases to add Recreate_DB of cannedPhrasesPredictor = "+ str(phrasesToAdd))
                convAssistLog.Log("phrases to phrasesToRemove Recreate_DB of cannedPhrasesPredictor= "+ str(phrasesToRemove))
                #### update embeddings, 
                self.corpus_embeddings = self.embedder.encode(self.pers_cannedphrasesLines, show_progress_bar=True, convert_to_numpy=True)
                # np.save(self.embedding_cache_path,{'sentences': self.pers_cannedphrasesLines, 'embeddings': self.corpus_embeddings})
                joblib.dump({'sentences': self.pers_cannedphrasesLines, 'embeddings': self.corpus_embeddings}, self.embedding_cache_path)
                # with open(self.embedding_cache_path, "wb") as fOut:
                #     pickle.dump({'sentences': self.pers_cannedphrasesLines, 'embeddings': self.corpus_embeddings}, fOut)

                #### update index:
                self.index = self.create_index(self.index)
            else:
                convAssistLog.Log("Recreate_DB of cannedPhrasesPredictor: NO modifications to cannedPhrases= ")

        except Error as e:
            convAssistLog.Log("Exception in CannedPhrasePredictor recreateDB  = "+e)

    def createSentDB(self, dbname):
        convAssistLog.Log("IN createSentDB")
        try:
            convAssistLog.Log("creating db = "+ dbname)
            conn = sqlite3.connect(dbname) 
            c = conn.cursor()
            c.execute('''
                    CREATE TABLE IF NOT EXISTS sentences
                    (sentence TEXT UNIQUE, count INTEGER)
                    ''')        
            conn.commit()
        except Error as e:
            convAssistLog.Log("Exception in createSentDB  = "+str(e))

    def find_semantic_matches(self,context, sent_prediction, cannedph):
        try:
            direct_matchedSentences = [s.word for s in sent_prediction]
            question_embedding = self.embedder.encode(context)
            #We use hnswlib knn_query method to find the top_k_hits
            corpus_ids, distances = self.index.knn_query(question_embedding, k=5)
            # We extract corpus ids and scores for the first query
            hits = [{'corpus_id': id, 'score': 1-score} for id, score in zip(corpus_ids[0], distances[0])]
            hits = sorted(hits, key=lambda x: x['score'], reverse=True)
            for i in range(0, len(hits)):
                ret_sent = self.pers_cannedphrasesLines[hits[i]['corpus_id']]
                if ret_sent.strip() not in direct_matchedSentences:
                    sent_prediction.add_suggestion(Suggestion(ret_sent.strip(), hits[i]["score"], self.name))
        except Error as e:
            convAssistLog.Log("Exception in CannedPhrasePredictor find_semantic_matches  = "+e)
        return sent_prediction

    def find_direct_matches(self,context, lines, sent_prediction, cannedph):
        try:
            total_sent = sum(cannedph.values())
            context_StemmedWords = [self.stemmer.stem(w) for w in context.split()]
            num_contextWords = len(context_StemmedWords)
            rows = []
            for k,v in cannedph.items():
                matchfound = 0
                sentence_StemmedWords = [self.stemmer.stem(w) for w in word_tokenize(k)]
                for c in context_StemmedWords:
                    if c in sentence_StemmedWords:
                        matchfound = matchfound+1
                new_row = {'sentence':k, 'matches':matchfound, 'probability':float(cannedph[k]/total_sent)}
                rows.append(new_row)
            scores = pd.DataFrame.from_records(rows)
            sorted_df = scores.sort_values(by = ['matches', 'probability'], ascending = [False, False])
            for index, row in sorted_df.iterrows():
                if(row["matches"]>0):
                    sent_prediction.add_suggestion(Suggestion(row['sentence'], row["matches"]+row["probability"], self.name))
        except Error as e:
            convAssistLog.Log("Exception in CannedPhrasePredictor find_direct_matches  = "+e)
        return sent_prediction

    def getTop5InitialPhrases(self, cannedph, sent_prediction):
        total_sent = sum(cannedph.values())
        probs = {}
        for k,v in cannedph.items():
            probs[k] = float(v/total_sent)

        sorted_x = collections.OrderedDict(sorted(probs.items(), key=lambda kv: kv[1], reverse=True))
        count = 0
        for k,v in sorted_x.items():
            if(count==5):
                break
            sent_prediction.add_suggestion(Suggestion(k, v, self.name))
            count = count+1
        return sent_prediction

    def predict(self, max_partial_prediction_size, filter):
        
        tokens = [""] * self.cardinality
        sent_prediction = Prediction()
        word_prediction = Prediction()
        try:
            context = self.context_tracker.past_stream().strip()
    
            if(context==""):
                ##### GET 5 MOST FREQUENT SENTENCES 
                sent_prediction = self.getTop5InitialPhrases(self.cannedPhrases_counts, sent_prediction)
                return sent_prediction, word_prediction

            ###### get matching sentences 
            ###### First get direct matches based on both databases: 

            sent_prediction = self.find_direct_matches(context, self.pers_cannedphrasesLines, sent_prediction, self.cannedPhrases_counts)

            ###### Get semantic matches based on both databases: 
            sent_prediction = self.find_semantic_matches(context, sent_prediction, self.cannedPhrases_counts)
            convAssistLog.Log("sent_prediction = "+str(sent_prediction))

        except Error as e:
            convAssistLog.Log("Exception in cannedPhrases Predict = "+e)
        return sent_prediction, word_prediction
            
    def close_database(self):
        self.db.close_database()
    
    def learn(self, change_tokens):
        #### For the cannedPhrase predictor, learning adds the sentence to the PSMCannedPhrases 
        if self.learn_mode == "True":
            convAssistLog.Log("learning ..."+change_tokens)
            try:

                #### ADD THE NEW PHRASE TO THE EMBEDDINGS, AND RECREATE THE INDEX. 
                if(change_tokens not in self.corpus_sentences):
                    convAssistLog.Log("phrase "+ change_tokens+ " not present, adding to embeddings and creating new index")
                    phrase_emb = self.embedder.encode(change_tokens.strip())
                    self.corpus_embeddings = np.vstack((self.corpus_embeddings, phrase_emb))
                    self.corpus_sentences.append(change_tokens.strip())
                    # np.save(self.embedding_cache_path,{'sentences': self.corpus_sentences, 'embeddings': self.corpus_embeddings})
                    joblib.dump({'sentences': self.corpus_sentences, 'embeddings': self.corpus_embeddings}, self.embedding_cache_path)
                    # with open(self.embedding_cache_path, "wb") as fOut:
                    #     pickle.dump({'sentences': self.corpus_sentences, 'embeddings': self.corpus_embeddings}, fOut)
                    self.index = self.create_index(self.index)

                conn = sqlite3.connect(self.sentences_db) 
                c = conn.cursor()
                count = 0
                #### CHECK IF SENTENCE EXISITS IN THE DATABASE
                c.execute("SELECT count FROM sentences WHERE sentence = ?", (change_tokens,))
                res = c.fetchall()
                if len(res) > 0:
                    if len(res[0]) > 0:
                        count = int(res[0][0])

                ### IF SENTENCE DOES NOT EXIST, ADD INTO DATABASE WITH COUNT = 1
                if count==0:
                    self.pers_cannedphrasesLines.append(change_tokens)
                    fout = open(self.personalized_cannedphrases, "w")
                    for l in self.pers_cannedphrasesLines:
                        fout.write(l+"\n")
                    fout.close()
                    c.execute('''
                    INSERT INTO sentences (sentence, count)
                    VALUES (?,?)''', (change_tokens, 1))
                    
                    self.cannedPhrases_counts[change_tokens] = 1 
                ### ELSE, IF SENTENCE EXIST, ADD INTO DATABASE WITH UPDATED COUNT
                else:
                    c.execute('''
                    UPDATE sentences SET count = ? where sentence = ?''', (count+1, change_tokens))
                    
                    self.cannedPhrases_counts[change_tokens] = count +1 
                conn.commit()
            except Error as e:
                convAssistLog.Log("Exception in LEARN CANNED PHRASES SENTENCES  = "+e)

    def _read_config(self):
        self.static_resources_path = self.config.get(self.name, "static_resources_path")
        self.personalized_resources_path = self.config.get(self.name, "personalized_resources_path")
        self.learn_mode = self.config.get(self.name, "learn")
        self.personalized_cannedphrases = os.path.join(self.personalized_resources_path, self.config.get(self.name, "personalized_cannedphrases"))
        self.sentences_db  = os.path.join(self.personalized_resources_path, self.config.get(self.name, "sentences_db"))
        self.embedding_cache_path = os.path.join(self.personalized_resources_path, self.config.get(self.name, "embedding_cache_path"))
        self.index_path = os.path.join(self.personalized_resources_path, self.config.get(self.name, "index_path"))
        self.sbertmodel = os.path.join(self.static_resources_path, self.config.get(self.name, "sbertmodel"))
