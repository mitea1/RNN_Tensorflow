import numpy
import numpy as np

from src.Board import Board


class Log:

    def __init__(self):
        self.states = []
        self.length = 0
        self.winner_char = ''

    # while logging
    # add a matrix representation of a state
    def add_state(self, state_matrix):
        self.states.append(state_matrix)
        self.length += 1

    # add the char of the winning player
    def add_winner(self, winner_char):
        self.winner_char = winner_char

    # after logging
    # who won the logged game
    def get_winner(self):
        return self.winner_char

    # get a specific state
    def get_state(self, index):
        if index < self.length:
            return self.states[index]
        else:
            return None

    def get_states(self, start_index, end_index):
        if end_index < self.length:
            return self.states[start_index:end_index]
