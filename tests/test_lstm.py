#!/usr/bin/env python

"""Tests for LSTM with keras."""

from RNN.Lstm import Lstm
import tensorflow as tf

def main():
    print(f'**********************************************')
    print(f'                Test LSTM                     ')
    print(f'**********************************************')
    print(f'Loading data ...')
    (input_train, y_train),(input_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)
    print(f'{len(input_train)},train sequences')
    print(f'{len(input_test)}, test sequences')

    print(f'Pad sequences (samples x time)')
    input_train = tf.keras.preprocessing.sequence.pad_sequences(input_train, maxlen=500) #maxlen es timesteps, sino se indica toma como valor la longitud de la secuencia más larga de la lista.
    input_test = tf.keras.preprocessing.sequence.pad_sequences(input_test, maxlen=500)
    print(f'input_train shape:{input_train.shape}')
    print(f'input_test shape:{input_test.shape}')
    rnn = Lstm(mode="Functional")
    rnn.initialize_values(max_features=10000, maxlen=500, batch_size=128, epochs=10)
    rnn.run(input_train, y_train)

    #print(f'>>> La salida de la red rnn es: {output}')
    

if __name__ == '__main__':
    main()
    
