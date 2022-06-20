import numpy as np  

class Simple_RNN():
    def __init__(self):
        ''' Default values'''
        self._timesteps = 5
        self._input_features = 3
        self._output_features = 2
        self.successive_outputs = []

    def set_timesteps(self, value):
        self._timesteps = value

    def set_input_features(self, value):
        self._input_features = value

    def set_output_features(self, value):
        self._output_features = value

    def get_timesteps(self):
        return self._timesteps

    def get_input_features(self):
        return self._input_features

    def get_output_features(self):
        return self._output_features

    def initialize_random_values(self):
        print(f'>>> Inicializar inputs y state con valores aleatorios')
        inputs = np.random.random((self.get_timesteps(), self.get_input_features()))
        state_t = np.zeros((self.get_output_features(),))
        return inputs, state_t

    def create_random_weight_matrix(self):
        print(f'>>> Crear las matrices de pesos (aleatorios)')
        W = np.random.random((self.get_output_features(), self.get_input_features()))
        U = np.random.random((self.get_output_features(), self.get_output_features()))
        b = np.random.random((self.get_output_features(),))
        return W, U, b

    def run(self, inputs, state_t, W, U, b):
        print(f'>>>>>>> Calculando rnn....')
        for idx, input_t in enumerate(inputs):
            print(f'>>>>>>>>>  step:{idx+1}')
            output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)

            self.successive_outputs.append(output_t)

            state_t = output_t

        final_output_sequence = np.stack(self.successive_outputs, axis=0)
        print(f'shape of successive outputs{self.successive_outputs[-1].shape}')
        print(f'shape of output (timesteps, output_features){final_output_sequence.shape}')
        return final_output_sequence
    