
from convassist.predictor.utilities.nlp import NLP

class SVOUtil:
        
    OBJECT_DEPS = {
            "dobj",
            "pobj",
            "dative",
            "attr",
            "oprd",
            "npadvmod",
            "amod",
            "acomp",
            "advmod",
    }
    
    SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "agent", "expl"}

    def __init__(self, stopwordsFile):
        self.nlp = NLP().get_nlp()
        
        self.stopwords = []
        with open(stopwordsFile) as f:
            self.stopwords = f.read().splitlines()
            self.stopwords = [word.strip() for word in self.stopwords]


    def extract_svo(self, sent) -> list[str]:
        doc = self.nlp(sent)
        sub = []
        at = []
        ve = []
        imp_tokens = []
        for token in doc:
            # is this a verb?
            if token.pos_ == "VERB":
                ve.append(token.text)
                if (
                    token.text.lower() not in self.stopwords
                    and token.text.lower() not in imp_tokens
                ):
                    imp_tokens.append(token.text.lower())
            # is this the object?
            if token.dep_ in self.OBJECT_DEPS or token.head.dep_ in self.OBJECT_DEPS:
                at.append(token.text)
                if (
                    token.text.lower() not in self.stopwords
                    and token.text.lower() not in imp_tokens
                ):
                    imp_tokens.append(token.text.lower())
            # is this the subject?
            if token.dep_ in self.SUBJECT_DEPS or token.head.dep_ in self.SUBJECT_DEPS:
                sub.append(token.text)
                if (
                    token.text.lower() not in self.stopwords
                    and token.text.lower() not in imp_tokens
                ):
                    imp_tokens.append(token.text.lower())
        return imp_tokens
