import numpy as np
class Log:

    def __init__(self):
        self.state_matrices = []
        self.length = 0
        self.winner_char = ''

    # add a matrix representation of a state
    def add_state(self, step_matrix):
        self.state_matrices.append(step_matrix)
        self.length += 1

    # add the char of the winning player
    def add_winner(self,winner_char):
        self.winner_char = winner_char

    # get a specific state
    def get_state(self, index):
        if index < self.length:
            return self.state_matrices[index]
        else:
            return None

    # get all states except for the last
    def get_all_states(self):
        return np.array(self.state_matrices[0:-1])

    # get the next step after index
    def get_next_step(self, index):
        if index + 1 < self.length:
            return self.state_matrices[index + 1] - self.state_matrices[index]
        else:
            return None
    # get all corresponding next steps to the steps get_all_states returns
    def get_all_next_steps(self):
        return np.array(self.state_matrices[1:])