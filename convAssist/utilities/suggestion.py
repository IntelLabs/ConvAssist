from ConvAssist.predictor.utilities.prediction import MAX_PROBABILITY, MIN_PROBABILITY

class SuggestionException(Exception):
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