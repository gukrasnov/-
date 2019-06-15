#!/usr/bin/env python
# coding: utf-8

# # Обучение Нейронной сети: 2 вход и 1 выход!

# In[1]:


import numpy as np
import scipy.special

f_activation = lambda x:scipy.special.expit(x)

def f_derivative(x):
  return (1 - x) * x

input_layer = np.array([[0, 0, 1],
                        [0, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]])

output_layer = np.array([[0],
                         [0],
                         [0],
                         [1]])

synaptic_weight = 2 * np.random.random((3, 1)) - 1

ls, al, DELTA_W_i = 0.7, 0.3, 0

for i in range(600):
  
  O_input = np.dot(input_layer, synaptic_weight)
  
  O_output = f_activation(O_input)
  
  error = ((output_layer - O_output) ** 2) / 1
  
  SIGMA_O = (output_layer - O_output) * f_derivative(O_output)
  
  GRAD_W = np.dot(input_layer.T, SIGMA_O)
  
  DELTA_W = ls * GRAD_W + al * DELTA_W_i
  
  synaptic_weight += DELTA_W
  
print(f'input_layer + bias\n{input_layer}')

print(f'output_layer\n{output_layer}')

print(f'O_input\n{O_input}')

print(f'O_output\n{O_output}')

print(f'error\n{error}')


# 

# 
