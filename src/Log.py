import numpy
import numpy as np

from src.Board import Board


class Log:

    def __init__(self):
        self.states = []
        self.length = 0
        self.winner_char = ''
        self.winning_rate_a = 0
        self.winning_rate_b = 0

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

    # sets the actual winning rate of a player at the moment the log was saved
    def set_winning_rate(self,player,rate):
        if player == 'a':
            self.winning_rate_a = rate
        elif player == 'b':
            self.winning_rate_b = rate
