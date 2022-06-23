
#%%
from pickletools import optimize
import tensorflow as tf
import matplotlib.pyplot as plt
#from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.layers import Embedding, SimpleRNN


def main():

    print(f'**********************************************')
    print(f'Calcular una red neuronal recurrente con Keras')
    print(f'**********************************************')

    max_features = 10000
    maxlen = 500
    batch_size = 32

    
    print(f'Loading data ...')
    (input_train, y_train),(input_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=max_features)
    print(f'{len(input_train)},train sequences')
    print(f'{len(input_test)}, test sequences')

    print(f'Pad sequences (samples x time)')
    input_train = tf.keras.preprocessing.sequence.pad_sequences(input_train, maxlen=maxlen) #maxlen es timesteps, sino se indica toma como valor la longitud de la secuencia más larga de la lista.
    input_test = tf.keras.preprocessing.sequence.pad_sequences(input_test, maxlen=maxlen)
    print(f'input_train shape:{input_train.shape}')
    print(f'input_test shape:{input_test.shape}')
    
    print(f'********************************************************')
    print(f' Training the model with Embedding and Simple RNN layers')
    print(f'                Functional                              ')
    print(f'********************************************************')
    
    input_tensor = tf.keras.Input(shape=(500,))
    print(f'input_tensor loaded')
    x = tf.keras.layers.Embedding(max_features, 32)(input_tensor)
    print(f'Embedding loaded')
    x = tf.keras.layers.SimpleRNN(32)(x)
    print(f'SimpleRNN loaded')
    output_tensor = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model_functional = tf.keras.Model(input_tensor, output_tensor)
    model_functional.compile(optimizer='rmsprop',
                    loss='binary_crossentropy',
                    metrics=['acc'])
    
    history_funtional = model_functional.fit(input_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

    model_functional.summary()
    
    
    print(f'Drawing...')
    acc = history_funtional.history['acc']
    val_acc = history_funtional.history['val_acc']
    loss = history_funtional.history['loss']
    val_loss = history_funtional.history['val_loss']

    epochs = range(1, len(acc) + 1)
   
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Functional: Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Functional:Training and validation loss')
    plt.legend()

    plt.show()

    print(f'********************************************************')
    print(f' Training the model with Embedding and Simple RNN layers')
    print(f'                Sequential                              ')
    print(f'********************************************************')
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(max_features, 32))
    model.add(tf.keras.layers.SimpleRNN(32))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                    loss='binary_crossentropy',
                    metrics=['acc'])

    history = model.fit(input_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
    
    model.summary() 
    
    print(f'Drawing...')
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)
   
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Sequential: Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Sequential: Training and validation loss')
    plt.legend()

    plt.show()
    
    
   
    

if __name__ == '__main__':
    main()
# %%
