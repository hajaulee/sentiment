import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import numpy as np
from sklearn.model_selection import train_test_split
from data_sources import *    
from visualization import *
from embedding import *
from model import *
np.random.seed(1)

CLASSES_NUMBER = 2
NET_NAME = 'sentiment'

if __name__ == "__main__":

    # Read train and test files
    X, Y = read_csv('imdb_labelled.txt.csv')

    # Read 50 feature dimension glove file
    word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('glove.6B.50d.txt')

    # Compute max length of sentences set
    maxLen = len(max(X, key=len).split())

    # Preprocess input data
    X, Y = preprocess_data(X, Y, CLASSES_NUMBER, word_to_index, maxLen)
    
    # Split train/test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    # Model and model summmary
    model = SentimentAnalysis((maxLen,), CLASSES_NUMBER, embedding_layer)

    # Train model
    model.fit(X_train, Y_train, epochs=2000, batch_size=32, workers=4,
    	shuffle=True, validation_data=(X_test, Y_test), callbacks=callbacks(NET_NAME))

    # Visualize History of Traing model
    show_training_history("./model/"+NET_NAME+"-history.txt")
    # Evaluate model, loss and accuracy
    loss, acc = model.evaluate(X_test, Y_test, verbose=1)
    print()
    print("Test accuracy = ", acc)