# ConvAssist Interface for ACAT
## Predictive Sentence Completion and Response Generation for Assistive Context-Aware Toolkit

ConvAssist enables different modes of interactions for users through the ACAT interface. The three modes are 
1) Sentence Mode 
2) Canned Phrases Mode 
3) Short-Hand Mode 

In the Sentence Mode, users can use the interface to type any sentence they would like to. As they type a letter, word completions appear along with their probabilities. After a word is chosen or typed, sentence completions also appear with 5 possible completions of the context typed so far. The predictions come from an AI model finetuned for the AAC use case. Please find the associated [model card here](https://github.com/IntelLabs/ConvAssist/blob/main/ACAT_ConvAssist_Interface/resources/static_resources/aac_gpt2/GPT2ForAAC_model_card.md). 

The Canned Phrases Mode can be used to store and search for frequently used phrases for easy and quick access. 

Short-Hand Mode can be used by users for quick communication - in situations where the user needs to communicate with their caretaker with minimum number of words (without the need for grammatical sentences). 