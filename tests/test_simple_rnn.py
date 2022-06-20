#!/usr/bin/env python

"""Tests for RNN."""

from RNN.Simple_RNN import Simple_RNN

def main():
    print(f'*******************************************')
    print(f'Calcular una red neuronal recurrente simple')
    print(f'*******************************************')
    r = Simple_RNN()
    inputs, state_t = r.initialize_random_values()
    W, U, b = r.create_random_weight_matrix()
    output = r.run(inputs, state_t, W, U, b)
    print(f'')
    print(f'>>> La salida de la red rnn es: {output}')
    

if __name__ == '__main__':
    main()
    
