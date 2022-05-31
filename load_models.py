from transformers import RobertaTokenizer, RobertaModel, RobertaForMaskedLM
import torch
import numpy as np

'''
Raw model is used for getting the word embeddings (needed because of altering of the masked embedding). 
LM model is used for substitute prediction. 
'''
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
lm_model = RobertaForMaskedLM.from_pretrained('roberta-base')
raw_model = RobertaModel.from_pretrained('roberta-base', output_hidden_states=True, output_attentions=True)

def load_transformers():
    return tokenizer, lm_model, raw_model

data = '''Redacted (NDA)'''

def load_data():
    return data