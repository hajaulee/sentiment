import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation, LeakyReLU
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from keras.callbacks import ModelCheckpoint, Callback
import datetime

class TrainingHistory(Callback):
  def __init__(self, net):
    self.net = net
  
  def on_train_begin(self, logs={}):
    self.histories = []

  def on_epoch_end(self, epoch, logs={}):
    self.histories.append([logs.get('loss'), logs.get('accuracy'), 
                           logs.get('val_loss'), logs.get('val_accuracy')])
    np.savetxt("./model/"+self.net+"-history.txt", self.histories, delimiter=",")


def today():
  return '{:02d}{:02d}'.format(datetime.date.today().month, datetime.date.today().day)

def callbacks(net_name):
    # checkpoint
    filepath="./model/"+net_name+"-"+ today() +"-weights-{epoch:02d}-{loss:.2f}-{accuracy:.2f}-{val_loss:.2f}-{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    log_history = TrainingHistory(net_name)
    callbacks = [checkpoint, log_history]
    return callbacks

def SentimentAnalysis(input_shape, classes_nb, embedding_layer):
    """
    Function creating the Emojify-v2 model's graph.

    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in Keras
    """
    # Define sentence_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
    sentence_indices = Input(shape=input_shape, dtype=np.int32)

    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings = embedding_layer(sentence_indices)

    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a batch of sequences.
    X = LSTM(128, return_sequences=True)(embeddings)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a single hidden state, not a batch of sequences.
    X = LSTM(128)(X)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    X = Dense(256)(X)
    X = LeakyReLU(alpha=0.15)(X)
    X = Dropout(0.5)(X)
    X = Dense(128)(X)
    X = LeakyReLU(alpha=0.15)(X)
    
    X = Dense(classes_nb)(X)
    # Add a softmax activation
    X = Activation('softmax')(X)

    # Create Model instance which converts sentence_indices into X.
    model = Model(sentence_indices, X)

    # Show summary of model
    model.summary()

    # Compiple model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model