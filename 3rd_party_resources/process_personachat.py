import nltk
from nltk.tokenize import sent_tokenize
import string

def parse_dialogs(text):
    ##Parse dialogs from persona chat, split into sentences using NLTK,
    ## and clean punctuation  (keep apostrophes)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    def clean_sentence(sentence):
        # Create a translation table that removes punctuation except apostrophes
        punct_to_remove = string.punctuation.replace("'", "")
        trans_table = str.maketrans("", "", punct_to_remove)
        cleaned = sentence.translate(trans_table)
        cleaned = " ".join(cleaned.split())
        cleaned = cleaned.replace(" ' ", " ")
        
        return cleaned
    
    lines = text.split('\n')
    all_sentences = []
    
    for line in lines:
        if not line.strip():
            continue
            
        parts = line.split('\t')
        if len(parts) < 2:
            continue
        
        # Get the parts before the double tabs (in persona chat, after double tabs are the candidate responses)
        # Remove the turn number from the first part
        first_part = ' '.join(parts[0].split(' ')[1:])
        second_part = parts[1]
        
        # Use NLTK to split into sentences 
        all_sentences.extend(clean_sentence(sent) for sent in sent_tokenize(first_part))
        all_sentences.extend(clean_sentence(sent) for sent in sent_tokenize(second_part))
    
    return all_sentences

fout = open("processed_personachat.txt", "w")
def parse_personachat(text):
    sentences = parse_dialogs(text)
    for sentence in sentences:
        if(len(sentence.split(" ")) >2 ):
        # if(sentence!="" or sentence!="\n"):
            fout.write(sentence+"\n")



lines = open("train_none_original_personachat.txt", "r").readlines()
for each in lines:
    parse_personachat(each)

fout.close()