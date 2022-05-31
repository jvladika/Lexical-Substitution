import torch
import string
import nltk
import time
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer

from filter import filter_words
from scores import calc_scores
from load_models import load_transformers

tokenizer, lm_model, raw_model = load_transformers()

'''
Second approach to lexical substitution, concatenating the original sentence to the masked sentence and passing it like that to the transformer 
to predict the target word. Example input: "Best offers of the season! [SEP] Best <mask> of the season!"

Implemented from scratch from:
Qiang et al. (2019) [https://arxiv.org/abs/1907.06226]
'''

def lexsub_concatenation(sentence, target):
    start = time.time()

    #Removes the unnecessary punctuation from the input sentence.
    sentence = sentence.replace('-', ' ')
    table = str.maketrans(dict.fromkeys(string.punctuation)) 

    split_sent = nltk.word_tokenize(sentence)
    split_sent = list(map(lambda wrd : wrd.translate(table) if wrd not in string.punctuation else wrd, split_sent))
    original_sent = ' '.join(split_sent)

    #Masks the target word in the original sentence.
    masked_sent = ' '.join(split_sent)
    masked_sent = masked_sent.replace(target, tokenizer.mask_token, 1)

    #Get the input token IDs of the input consisting of: the original sentence + separator + the masked sentence.
    input_ids = tokenizer.encode(" "+original_sent, " "+masked_sent, add_special_tokens=True)
    masked_position = input_ids.index(tokenizer.mask_token_id)

    original_output = raw_model(torch.tensor(input_ids).reshape(1, len(input_ids)))

    #Get the predictions of the Masked LM transformer.
    with torch.no_grad():
            output = lm_model(torch.tensor(input_ids).reshape(1, len(input_ids)))
    
    logits = output[0].squeeze()

    #Get top guesses: their token IDs, scores, and words.
    mask_logits = logits[masked_position].squeeze()
    top_tokens = torch.topk(mask_logits, k=20, dim=0)[1]
    scores = torch.softmax(mask_logits, dim=0)[top_tokens].tolist()
    words = [tokenizer.decode(i.item()).strip() for i in top_tokens]
    
    words, scores, top_tokens = filter_words(target, words, scores, top_tokens)
    assert len(words) == len(scores)

    if len(words) == 0: 
        return

    print("GUESSES: ", words)

    #Calculate proposal scores, substitute validation scores, and final scores
    original_score = torch.softmax(mask_logits, dim=0)[masked_position]
    sentences = list()

    for i in range(len(words)):
        subst_word = top_tokens[i]
        input_ids[masked_position] = int(subst_word)
        sentences.append(list(input_ids))

    #print([tokenizer.decode(s) for s in sentences])
    torch_sentences = torch.tensor(sentences)

    finals, props, subval = calc_scores(scores, torch_sentences, original_output, original_score, masked_position)
    finals = map(lambda f : float(f), finals)
    props = map(lambda f : float(f), props)
    subval = map(lambda f : float(f), subval)

    if target in words:
        words = [w for w in words if w not in [target, target.capitalize(), target.upper()]] 

    zipped = dict(zip(words, finals))
    lemmatizer = WordNetLemmatizer()

    ###Remove plurals, wrong verb tenses, duplicate forms, etc.############
    original_pos = nltk.pos_tag(nltk.word_tokenize(original_sent))
    target_index = split_sent.index(target)
    assert original_pos[target_index][0] == target
    original_tag = original_pos[target_index][1]

    for i in range(len(words)):
        cand = words[i]
        if cand not in zipped:
            continue
        
        sent = original_sent
        masked_sent = sent.replace(target, cand, 1)

        new_pos = nltk.pos_tag(nltk.word_tokenize(masked_sent))
        new_tag = new_pos[target_index][1]

        #If the word appears in both singular and plural in the candidate list, remove one of them.
        if new_tag.startswith('N') and not new_tag.endswith('S'):
            if (cand+'s' in words or cand+'es' in words) in words and original_tag.endswith('S'):
                del zipped[cand]
                continue        
        elif new_tag.startswith('N') and new_tag.endswith('S'):
            if (cand[:-1] in words or cand[:-2] in words) and not original_tag.endswith('S'):
                del zipped[cand]
                continue
        
        #If multiple forms of the original word appear in the candidate list, remove them (e.g. begin, begins, began, begun...)
        wntags = ['a', 'r', 'n', 'v']
        for tag in wntags:
           if lemmatizer.lemmatize(cand, tag) == lemmatizer.lemmatize(target, tag):
                del zipped[cand]
                break
    #################        

    #Print sorted candidate words.
    zipped = dict(zipped)
    finish = list(sorted(zipped.items(), key=lambda item: item[1], reverse=True))[:15]
    print("CANDIDATES:", ["({0}: {1:0.8f})".format(k, v) for k,v in finish])

    #Print any relations between candidates and the original word.
    words = zipped.keys()
    nyms = get_nyms(target)
    nym_output = list()
    for cand in words:
        for k, v in nyms.items():
            if lemmatizer.lemmatize(cand.lower()) in v:
                nym_output.append((cand, k[:-1]))

    if nym_output:
        print("NYMS: ", ["({0}, {1})".format(pair[0], pair[1]) for pair in nym_output])
    
    print("Elapsed time: ", time.time() - start, "\n")
