import re

def fix_encoding(text):
    replacements = {
        "â€™": "'",    
        'â€"': "-",   
        "â€œ": '"',    
        "â€": '"',     
        "â€˜": "'",
    }
    
    for bad, good in replacements.items():
        print(bad,good)
        text = text.replace(bad, good)
    
    # Remove corrupted emoji patterns using regex
    text = re.sub(r'ð[Ÿÿ][™˜][‚‰ƒ†›žŸ\s€<0x90>][?:]?|ï¼šï¼‰', '', text)
    # text = re.sub(r'ðÿ[™˜][‚ƒ†›žŸ\s]', '', text)
    return text

def clean_sentence(sentence):
    # Skip sentences containing redacted terms
    if '<' in sentence and '>' in sentence:
        return None
        
    # Remove punctuation except apostrophes
    punctuation = '.,!?;:"""()-'
    for p in punctuation:
        sentence = sentence.replace(p, '')
    
    # Remove extra spaces and convert to lowercase
    return ' '.join(sentence.split()).lower()

def parse_conversation(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    clean_sentences = []
    
    for line in lines:
        if line.startswith('Human'):
            _, message = line.split(':', 1)
            message = fix_encoding(message.strip())
            clean_sentence_result = clean_sentence(message)
            if clean_sentence_result:  # Only add non-empty and non-redacted sentences
                clean_sentences.append(clean_sentence_result)
    
    return clean_sentences


text = open("orig_Human_Conversation_dataset.txt", "r").read()
sentences = parse_conversation(text)
print(sentences[1:5])
fout = open("processed_Human_Conversation_dataset.txt", "w")
for each in sentences:
    fout.write(each+"\n")
fout.close()