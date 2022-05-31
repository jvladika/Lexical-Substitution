import nltk
import string
import time
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import torch

'''
WordNet is a lexical database of semantic relations between words in more than 200 languages. WordNet links words into semantic relations including synonyms, 
hyponyms, and meronyms. The synonyms are grouped into synsets with short definitions and usage examples. WordNet can thus be seen as a combination and extension 
of a dictionary and thesaurus. While it is accessible to human users via a web browser, its primary use is in automatic text analysis and AI applications.
'''
lemmatizer = WordNetLemmatizer()

'''
Gets antonyms of a word in all its meanings from the WordNet knowledge base.
* antonyms = words with an opposite meaning of the target word (day-night)
'''
def get_antonyms(word):
    ants = list()

    #Get antonyms from WordNet for this word and any of its synonyms.
    for ss in wn.synsets(word):
        ants.extend([lm.antonyms()[0].name() for lm in ss.lemmas() if lm.antonyms()]) 

    #Get snyonyms of antonyms found in the previous step, thus expanding the list even more.
    syns = list()
    for word in ants:
        for ss in wn.synsets(word):
            syns.extend([lm.name() for lm in ss.lemmas()])

    return sorted(list(set(syns)))

'''
Gets pertainyms of the target word from the WordNet knowledge base.
* pertainyms = words pertaining to the target word (industrial -> pertainym is "industry")
'''
def get_pertainyms(word):
    perts = list()
    for ss in wn.synsets(word):
        perts.extend([lm.pertainyms()[0].name() for lm in ss.lemmas() if lm.pertainyms()]) 
    return sorted(list(set(perts)))

'''
Gets derivationally related forms (e.g. begin -> 'beginner', 'beginning')
'''
def get_related_forms(word):
    forms = list()
    for ss in wn.synsets(word):
        forms.extend([lm.derivationally_related_forms()[0].name() for lm in ss.lemmas() if lm.derivationally_related_forms()]) 
    return sorted(list(set(forms)))

'''
Gets antonyms, hypernyms, hyponyms, holonyms, meronyms, pertainyms, and derivationally related forms of a target word from WordNet.
* hypernym = a word whose meaning includes a group of other words ("animal" is a hypernym of "dog")
* hyponym = a word whose meaning is included in the meaning of another word ("bulldog" is a hyponym of "dog")
* a meronym denotes a part and a holonym denotes a whole: "week" is a holonym of "weekend", "eye" is a meronym of "face", and vice-versa
'''
def get_nyms(word, depth=-1):
    nym_list = ['antonyms', 'hypernyms', 'hyponyms', 'holonyms', 'meronyms', 
                'pertainyms', 'derivationally_related_forms']
    results = list()
    word = lemmatizer.lemmatize(word)

    def query_wordnet(getter):
        res = list()
        for ss in wn.synsets(word):
            res_list = [item.lemmas() for item in ss.closure(getter, depth=depth)]
            res_list = [item.name() for sublist in res_list for item in sublist]
            res.extend(res_list)
        return res

    for nym in nym_list:
        if nym=='antonyms':
            results.append(get_antonyms(word))

        elif nym in ['hypernyms', 'hyponyms']:
            getter = eval("lambda s : s."+nym+"()") 
            results.append(query_wordnet(getter))

        elif nym in ['holonyms', 'meronyms']:
            res = list()
            #Three different types of holonyms and meronyms as defined in WordNet
            for prefix in ['part_', 'member_', 'substance_']:
                getter = eval("lambda s : s."+prefix+nym+"()")
                res.extend(query_wordnet(getter))
            results.append(res)

        elif nym=='pertainyms':
            results.append(get_pertainyms(word))

        else:
            results.append(get_related_forms(word))

    results = map(set, results)
    nyms = dict(zip(nym_list, results))
    return nyms

