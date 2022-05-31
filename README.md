# Lexical-Substitution
Implementation of a couple of Lexical Substitution approaches using BERT and RoBERTa transformers. 

_**Lexical Substitution** is the task of selecting a word that can replace a target word in a context (a sentence, paragraph, document, etc.) without changing the meaning of the context. If you think about it, this is the main application of a thesaurus: using synonyms to replace words._

_The problem is, however, that a thesaurus is not necessary for substituting words. Consider the words "questions" and "answers". Would you find them as synonyms in a thesaurus? In the sentence "I got 10 answers right", the word "answers" can be substituted with "questions". Thesauri are not sufficient either because two words which are commonly considered synonyms might not be substitutable in certain contexts even if they are of the same sense, such as "error" and "mistake" in the sentence "An error message popped up."_
(further read: https://geekyisawesome.blogspot.com/2014/10/thoughts-on-lexical-substitution.html)

This project contains implementations of two approaches to lexical substitution: (1) using concatenation of the original sentence with the masked sentence from [Lexical Simplification with Pretrained Encoders](https://arxiv.org/abs/1907.06226) by Qiang et al. (2019); (2) using dropout of random weights in transformer's word embedding for the target word from [BERT-based Lexical Substitution](https://aclanthology.org/P19-1328/) by Zhou et al. (2019). Final substitution candidates are ranked using measures and scores from [Always Keep your Target in Mind: Studying Semantics and Improving Performance of Neural Lexical Substitution](https://aclanthology.org/2020.coling-main.107.pdf) by Arefyev et al. (2020). 

Afterwards, those candidate words that have according to [WordNet](https://wordnet.princeton.edu) the relation to the target word of being its antonyms, hypernyms, hyponyms, pertainyms, holonyms, meronyms, or derivationally related forms, are removed from the candidate list and the final clean list is printed out. Final sentences containing the substituted word are evaluated by similarity to the original sentence as measured by SentenceBERT (Reimers and Gurevych, 2018) in order to assess the quality of the substitution.


