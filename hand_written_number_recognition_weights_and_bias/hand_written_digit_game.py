import pygame, sys
from pygame.locals import *
from math import ceil
import numpy as np
import os


'''
Digit recognition Neural Network.

code written by Bryn Ghiffar.

'''

os.chdir('C:\\Users\\bryng\\OneDrive - Monash University\\PythonPrograms\\neural_network_from_scratch\\hand_written_number_recognition_weights_and_bias')
layer_1_weights = np.load('layer_1_weights.npy')
layer_1_bias = np.load('layer_1_bias.npy')
output_layer_weights = np.load('output_layer_weights.npy')
output_layer_bias = np.load('output_layer_bias.npy')

def sigmoid(x):
    '''
    sigmoid activation function for neural network
    '''
    return (1 + np.exp(- x)) ** -1


class brush:
    
    def __init__(self) -> None:
        '''
        Initializes the array of the Neural Network
        '''
        self.draw_matrix = np.array([[0 for _ in range(28)] for _ in range(28)])
    
    def draw(self, window) -> None:
        '''
        Draws the pixels on the screen as well as stores values in draw_matrix instance
        '''
        is_left_button_pressed = pygame.mouse.get_pressed()[0]
        is_right_button_pressed = pygame.mouse.get_pressed()[2]
        mouse_pos = list(pygame.mouse.get_pos())
        mouse_pos[0] = self.round_val(mouse_pos[0])
        mouse_pos[1] = self.round_val(mouse_pos[1])


        if is_left_button_pressed: # hold left click to erase the board
            pygame.draw.rect(window,(255, 255, 255), (mouse_pos[0], mouse_pos[1], 10, 10))
            self.draw_matrix[mouse_pos[0] // 10][mouse_pos[1] // 10] = 1

        elif is_right_button_pressed: # right click to erase the white board
            window.fill((0,0,0))
            self.draw_matrix = np.array([[0 for _ in range(28)] for _ in range(28)])
        
        elif pygame.mouse.get_pressed()[1]: # mouse wheel button to guess the number
            print(self.guess_number())

    def round_val(self, num):
        '''
        mouse position is rounded. So, input is in pixelated form.
        '''
        return int(round(num / 10)) * 10

    def guess_number(self):
        '''
        Guess's the number based on draw_matrix instance input.
        '''
        input_vector = np.transpose(self.draw_matrix)
        input_vector = np.reshape(input_vector, 784)
        network_1 = sigmoid(np.dot(layer_1_weights, input_vector) + layer_1_bias)
        network_2 = sigmoid(np.dot(output_layer_weights, network_1) + output_layer_bias)

        return np.argmax(network_2)



brush_1 = brush()


# Do not mind the red squiggles
pygame.init()

window = pygame.display.set_mode((280, 280)) # This is the window size

pygame.display.set_caption("Digit Recognition NN")
EXIT = False
while not EXIT:
    # content

    brush_1.draw(window)

    pygame.display.update()
                # Do not mind the red squiggles here
    for event in pygame.event.get():
        if event.type == QUIT:
            EXIT = True