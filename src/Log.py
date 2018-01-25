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

    def get_states(self, start_index, end_index):
        if end_index < self.length:
            return self.state_matrices[start_index:end_index]

    def get_next_steps(self, start_index, end_index):
        next_steps = []
        num_elements = end_index-start_index
        for i in range(0, num_elements):
            next_steps.append(self.state_matrices[start_index + 1 + i] - self.state_matrices[start_index + i])
        return next_steps

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
        next_steps = []
        for i in range(0, self.length-1):
            next_steps.append(self.state_matrices[i+1] - self.state_matrices[i])
        return np.array(next_steps)
