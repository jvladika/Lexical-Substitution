import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import torch
import string

from nyms import get_nyms

'''
WordNet is a lexical database of semantic relations between words in more than 200 languages. WordNet links words into semantic relations including synonyms, 
hyponyms, and meronyms. The synonyms are grouped into synsets with short definitions and usage examples. WordNet can thus be seen as a combination and extension 
of a dictionary and thesaurus. While it is accessible to human users via a web browser, its primary use is in automatic text analysis and AI applications.
'''
lemmatizer = WordNetLemmatizer()

#Converts a part-of-speech tag returned by NLTK to a POS tag from WordNet
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return ''

#Function for clearing up duplicate words (capitalized, upper-case, etc.), stop words, and antonyms from the list of candidates.
def filter_words(target, words, scr, tkn):
    dels = list()
    toks = tkn.tolist()
    nyms = get_nyms(target)
    lemmatizer = WordNetLemmatizer()

    for w in words:
        if w.lower() in words and w.capitalize() in words:
            dels.append(w.capitalize())
        if w.lower() in words and w.upper() in words:
            dels.append(w.upper())
        if w in nltk.corpus.stopwords.words('english') or w in string.punctuation:
            dels.append(w)
        if lemmatizer.lemmatize(w.lower()) in nyms['antonyms']:
            dels.append(w)

    dels = list(set(dels))
    for d in dels:
        del scr[words.index(d)]
        del toks[words.index(d)]
        words.remove(d)

    return words, scr, torch.tensor(toks)