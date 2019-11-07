# LSTM CODE: Sentiment Analysis

## Install library

* pip : `pip install -r requirements.txt`
* pip3: `pip3 install -r requirements.txt`

## Download dictionary

* [Glove 50 dimension, 40000 words of dictionary file](http://akii.tk/QVNCUL)


## DataSet

We have a dataset (X, Y) from amazon imdb where:

X contains 1000 sentences (strings) consist of 500 negative samples and 500 positive samples
Y contains a integer label between 0 and 1 corresponding to each sentence


## Embeddings
Glove 50 dimension, 40000 words of dictionary file is used for word embeddings. It should be downloaded from [http://akii.tk/QVNCUL](http://akii.tk/QVNCUL) (file size = ~168MB))

* `word_to_index`: dictionary mapping from words to their indices in the vocabulary (400,001 words, with the valid indices ranging from 0 to 400,000)
* `index_to_word`: dictionary mapping from indices to their corresponding words in the vocabulary
* `word_to_vec_map`: dictionary mapping words to their GloVe vector representation.