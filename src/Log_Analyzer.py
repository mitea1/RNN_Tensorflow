import numpy as np
from src.Board import Board
from src.Log import Log


class Log_Analyzer:

    def __init__(self, log):
        self.log = log

    def get_next_steps(self, start_index, end_index):
        next_steps = []
        num_elements = end_index - start_index
        for i in range(0, num_elements):
            delta_state = self.log.states[start_index + 1 + i] - self.log.states[start_index + i]
            max_index = np.argmax(delta_state)  # get the cell number which has changed
            column = max_index % Board.NUM_COLUMNS  # get the column of that cell
            action = np.zeros(Board.NUM_COLUMNS)  # build an action tensor
            action[column] = 1  # set the column to 1 according to the action cell
            next_steps.append(action)
        return next_steps

        # get all states except for the last

    def get_all_states(self, padding=False, including_last=False):
        # Pad the rest of data with 0 states e.g.(0,0,0,0,0,0,0)
        if padding:
            number_of_states = len(self.log.states)
            number_of_max_states = (Board.NUM_ROWS * Board.NUM_COLUMNS)
            if including_last:
                number_of_max_states = number_of_states + 1
            for i in range(number_of_states, number_of_max_states):
                self.log.states.append(np.zeros((Board.NUM_ROWS, Board.NUM_COLUMNS, len(Board.EMPTY_CELL))))

        return np.array(self.log.states)

    # get the action after index: data e.g. ([0,0,1,0,0,0],[0,1])
    def get_action(self, index):
        if index + 1 < self.log.length:
            delta_state = self.log.states[index + 1] - self.log.states[index]
            max_index = np.argmax(delta_state)
            celltype = self._get_celltype(max_index)
            column = self._get_column(max_index)
            next_step = np.zeros(Board.NUM_COLUMNS)
            next_step[column] = 1
            return next_step, celltype
        else:
            return None, None

    # get all corresponding actions to the steps get_all_states returns
    # data e.g ([0,0,1,0,0,0],cell_type)
    def get_all_actions(self, cell_type=None, padding=False):
        actions = []

        for i in range(0, self.log.length - 1):
            # Only add actions of the given cell type if defined or add just every action
            if cell_type is None:
                actions.append(self.get_action(i)[0])
            elif self.get_action(i)[1] == cell_type:
                actions.append(self.get_action(i)[0])

        # Pad the rest of data with 0 action e.g.(0,0,0,0,0,0,0)
        if padding:
            number_of_max_actions = Board.NUM_ROWS * Board.NUM_COLUMNS
            for i in range(self.log.length - 1, number_of_max_actions):
                actions.append(np.zeros(Board.NUM_COLUMNS))

        return np.array(actions)

    def get_last_action(self):
        return self.get_action(self.log.length-1)

    # Return rewards according to the action e.g.[-1,1,-1,1,-1,1,-1,1,-1,] actions of winning player are rated 1
    # actions of losing player are rated -1
    def get_rewards(self):
        number_of_max_actions = Board.NUM_ROWS * Board.NUM_COLUMNS
        rewards = np.zeros(number_of_max_actions)
        for i in range(0, self.length - 1):
            if self.get_action(i)[1] == self.winner_char:
                rewards[i] = 1
            else:
                rewards[i] = -1
        return rewards
        # get all corresponding next steps to the steps get_all_states returns

    def get_all_next_steps(self):
        next_steps = []
        for i in range(0, self.log.length - 1):
            next_steps.append(self.get_action(i))
        return np.array(next_steps)

    #
    def _get_celltype(self, max_index):
        if max_index % 2 == 0:
            return Board.X_OCCUPIED_CELL
        else:
            return Board.O_OCCUPIED_CELL

    # return column where action happend for player [1,0] and [0,1]
    def _get_column(self, max_index):
        # player [1,0] did action
        if max_index % 2 == 0:
            return int(max_index / 2) % Board.NUM_COLUMNS
        # player [0,1] did action
        else:
            return int((max_index - 1) / 2) % Board.NUM_COLUMNS

