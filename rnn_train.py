"""RNN training script"""

import music21 as music
from music21 import converter, instrument, note, chord
import keras
import os
import scipy
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras import activations
from keras.models import Model
from keras import backend as K
from keras.utils import np_utils
from keras.layers import Activation, Dropout, LSTM, RepeatVector, ConvLSTM2DCell, Reshape, BatchNormalization
from keras.layers.core import Dense, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras import applications
from keras import optimizers
from keras import regularizers
from keras.optimizers import Adam


midipath = "/content/drive/My Drive/Midi_Dataset/percussions/*.mid"

def read_notes(files):
  notes = []

  for file in glob.glob(files):
    midi = converter.parse(file)
    print("Parsing %s" % file)
    
    notes_to_parse = None
    
    try: # file has instrument parts
      s2 = instrument.partitionByInstrument(midi)
      notes_to_parse = s2.parts[0].recurse()
    except: # file has notes in a flat structure
      notes_to_parse = midi.flat.notes
        
    for element in notes_to_parse:
      if isinstance(element, note.Note):
        notes.append(str(element.pitch))
      elif isinstance(element, chord.Chord):
        notes.append('.'.join(str(n) for n in element.normalOrder))
          
  with open('/content/notes', 'wb') as filepath:
    pickle.dump(notes, filepath)
    
  return notes

              
def prepare_sequences(notes, n_vocab):
    """Prepare the sequences to train the model"""
    sequence_length = 128

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)
  
  
def build_network(network_input, n_vocab):
  """Build the structure of RNN model"""
  opt = Adam(lr=0.000077, 
              beta_1=0.990, 
              beta_2=0.999, 
              epsilon=None, 
              decay=0.0)
  
  model = Sequential()
  model.add(BatchNormalization())
  model.add(LSTM(256,
                 input_shape=(network_input.shape[1], network_input.shape[2]),
                 return_sequences=True,
                 #recurrent_regularizer=regularizers.l1_l2(0.0002, 0.0001),
                 unit_forget_bias=True,
                 activation='tanh',
                 name='lstm_input'))
  model.add(BatchNormalization())
  model.add(Dropout(0.09))
  model.add(LSTM(128, 
                 return_sequences=True,
                 activation='tanh', 
                 unit_forget_bias=False,
                 #recurrent_regularizer=regularizers.l1_l2(0.003, 0.0003),
                 name='lstm_2'))
  model.add(Dense(4096,
                  activation='relu',
                  name='dense_1'))
  #model.add(RepeatVector(5))
  """model.add(LSTM(128, 
                 return_sequences=True, 
                 activation='softmax', 
                 unit_forget_bias=True,
                 activity_regularizer=regularizers.l1_l2(0.03, 0.04),
                 name='lstm_2'))"""
  #model.add(BatchNormalization())
  model.add(Dropout(0.03))
  model.add(LSTM(256, 
                 #return_sequences=True,
                 activation='tanh', 
                 unit_forget_bias=False,
                 #recurrent_regularizer=regularizers.l1_l2(0.003, 0.0003),
                 name='lstm_3'))
  #model.add(RepeatVector(5))
  """model.add(ConvLSTM2D(filters=64,
                           kernel_size=(3, 3),
                           strides=(1, 1),
                           padding='valid',
                           recurrent_regularizer=regularizers.l1_l2(0.00003, 0.00001),
                           name='convlstm_1'))"""
  #model.add(BatchNormalization())
  """model.add(LSTM(512, 
                 return_sequences=True,
                 #activation='softmax', 
                 unit_forget_bias=True,
                 activity_regularizer=regularizers.l1_l2(0.0003, 0.0007),
                 name='lstm_4'))
  model.add(LSTM(512, 
                 #return_sequences=True,
                 #activation='softmax', 
                 unit_forget_bias=True,
                 recurrent_regularizer=regularizers.l1_l2(0.0007, 0.0002),
                 name='lstm_5'))"""
  #model.add(Dropout(0.3))
  model.add(Dense(4096,
                  name='dense_2',
                  activation='relu'))
  #model.add(RepeatVector(3))
  model.add(Dropout(0.1))
  model.add(Dense(n_vocab,
                  activation='softmax',
                  name='dense_out'))
  
  model.compile(loss='categorical_crossentropy', 
                optimizer='rmsprop', 
                metrics=['accuracy'])

  return model


def train(model, network_input, network_output):
  """Training config"""
  filepath="saved-model-{epoch:02d}-{loss:.2f}.hdf5"
  checkpoint = ModelCheckpoint(filepath,
                               monitor='loss', 
                               verbose=0, 
                               save_best_only=True, 
                               save_weights_only=False, 
                               mode='auto', period=1)
  callbacks_list = [checkpoint] 

  model.fit(network_input, 
            network_output, 
            epochs=100, 
            batch_size=64,
            callbacks=callbacks_list)
  
  
def train_rnn():
    """Train model on dataset to generate music"""
    notes = read_notes(midipath)

    # get amount of pitch names
    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = build_network(network_input, n_vocab)

    train(model, network_input, network_output)
    

if __name__ == '__main__':
    train_rnn()
