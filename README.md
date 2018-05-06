Tools for computing distributed representtion of words
------------------------------------------------------

We've adapted the Skip-gram model of word2vec from https://code.google.com/p/word2vec/ to be able to take in tokens of the form word:lemma rather than just word, and learn embeddings for both. Since a word type may correspond to multiple lemma types, we compute a word's overall vector by the word vector concatenated with a weighted average of the lemma vectors.

Example usage:

make

word2vec_morph -train input.txt -output output.txt -output-lemmas lemmas.txt -output-num-lemmas num_lemmas.txt -cbow 0 -word-size 150 -lemma-size 150 -window 5 -negative 5 -hs 0 -sample 1e-4 -threads 12 -binary 0 -iter 5
# word2vec_morph
