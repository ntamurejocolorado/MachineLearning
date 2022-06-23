from pickletools import optimize
import tensorflow as tf
import matplotlib.pyplot as plt



class Lstm():
    print(f'**********************************************')
    print(f'                Calcular LSTM                 ')
    print(f'**********************************************')
    def __init__(self, mode="Sequential"):
        print("Hello class rnn keras")
        self.mode = mode
        self.max_features = None
        self.maxlen = None
        self.batch_size = None
        self.epochs = None
        
    def run(self, input_train, y_train):
        print("----------------- Selection API-------------------")
        model = self.selection_api(self.mode)
        print("----------------- Compile and fit -------------------")
        history = self.compile_and_fit(model, input_train, y_train)
        print("------------------------ values from fit ---------------------")
        acc, val_acc, loss, val_loss, epochs = self.get_values_from_fit(history)
        print("---------------- Show -----------------------")
        self.show(acc, val_acc, loss, val_loss, epochs)
        
    def initialize_values(self, max_features, maxlen, batch_size, epochs):
        self.set_max_features(max_features)
        self.set_maxlen(maxlen)
        self.set_batch_size(batch_size)
        self.set_epochs(epochs)
        
    def selection_api(self,mode):
        if mode == "Functional":
            print("Funcional")
            return self.functional_api(500)
            
        elif mode == "Sequential":
            print("Secuencial")
            return self.sequential_api()
            
        else:
            print("Revisa el nombre")
            return False
    
    def functional_api(self, shape_data):
        print(f'********************************************************')
        print(f'                Functional                              ')
        print(f'********************************************************')
        
        input_tensor = tf.keras.Input(shape=(shape_data,))
        x = tf.keras.layers.Embedding(self.max_features, 32)(input_tensor)
        x = tf.keras.layers.LSTM(32)(x)
        output_tensor = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        model_functional = tf.keras.Model(input_tensor, output_tensor)
        
        model_functional.summary()
        return model_functional
    
    def sequential_api(self):
        print(f'********************************************************')
        print(f'                Sequential                              ')
        print(f'********************************************************')
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Embedding(self.max_features, 32))
        model.add(tf.keras.layers.LSTM(32))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        return model
    
    def compile_and_fit(self, model, input_train, y_train):
        model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
        history = model.fit(input_train, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.2)
        return history
    
    def get_values_from_fit(self, history):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(acc) + 1)
        return acc, val_acc, loss, val_loss, epochs

    def set_max_features(self, value):
        self.max_features = value
        
    def set_maxlen(self, value):
        self.maxlen = value
        
    def set_batch_size(self, value):
        self.batch_size = value
        
    def set_epochs(self, value):
        self.epochs = value
        
    def show(self, acc, val_acc, loss, val_loss, epochs):
    
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title(str(self.mode)+': Training and validation accuracy')
        plt.legend()
        plt.savefig(str(self.mode)+'_accuracy.png')

        plt.figure()

        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title(str(self.mode)+':Training and validation loss')
        plt.legend()
        plt.savefig(str(self.mode)+'_loss.png')
        plt.show()

