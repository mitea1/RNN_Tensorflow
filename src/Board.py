import numpy as np

from src.GameState import GameState


class Board:
    NUM_ROWS = 6
    NUM_COLUMNS = 7
    X_OCCUPIED_CELL = [1, 0]
    O_OCCUPIED_CELL = [0, 1]
    EMPTY_CELL = [0, 0]
    CHAR_EMPTY = '_'
    CHAR_PLAYER_A = 'x'
    CHAR_PLAYER_B = 'o'
    LINE_LENGTH = 4

    def __init__(self, num_rows=NUM_ROWS, num_columns=NUM_COLUMNS):
        self.rows = num_rows
        self.columns = num_columns
        self.board_matrix = np.zeros((self.rows, self.columns, len(self.EMPTY_CELL)))
        self.char_to_state_vector = {self.CHAR_PLAYER_A: self.X_OCCUPIED_CELL, self.CHAR_PLAYER_B: self.O_OCCUPIED_CELL, self.CHAR_EMPTY: self.EMPTY_CELL}
        self.board_char_representation = np.chararray((self.rows, self.columns), itemsize=1, unicode=1)
        self.board_char_representation[:] = self.CHAR_EMPTY

    def clear(self):
        self.board_matrix = np.zeros((self.rows, self.columns, len(self.EMPTY_CELL)))
        self.board_char_representation[:] = self.CHAR_EMPTY

    # Gets the latest char representation matrix from the board
    def convert_to_char_representation(self):
        rows, columns, cells = self.board_matrix.shape
        for i in range(0, rows):
            for j in range(0, columns):
                entry = self.board_matrix[i,j]
                if entry[0] == 1:
                    self.board_char_representation[i, j] = self.CHAR_PLAYER_A
                elif entry[1] == 1:
                    self.board_char_representation[i, j] = self.CHAR_PLAYER_B
                else:
                    self.board_char_representation[i, j] = self.CHAR_EMPTY

    # prints out the latest char representation of the board
    def print_char_representation(self):
        self.convert_to_char_representation()
        print(self.board_char_representation)
        print()

    # define the state of a given cell
    def set_cell_state(self, row, column, state):
        if row < self.rows and column < self.columns \
                and self.board_matrix[row, column, 0] == 0 and self.board_matrix[row, column, 1] == 0:
            self.board_matrix[row, column] = state
            return GameState.SET_SUCCESSFUL
        else:
            return GameState.TRY_TO_SET


    # find out if there are 4 connected on the board
    def are_4_connected(self):
        self.convert_to_char_representation()
        for row in range(0, self.rows):
            for column in range(0, self.columns):
                actual_char = self.board_char_representation[row, column]
                if actual_char == self.CHAR_PLAYER_A or actual_char == self.CHAR_PLAYER_B:
                    checking_next_column_char = True
                    checking_next_diagdown_char = True
                    checking_next_diagup_char = True
                    checking_next_row_char = True

                    line_length_count = 1
                    while(checking_next_column_char):
                        if (column + line_length_count) > (self.columns - 1):
                            checking_next_column_char = False
                        else:
                            next_column_char = self.board_char_representation[row, column + line_length_count]
                            if next_column_char == actual_char:
                                line_length_count += 1
                            else:
                                checking_next_column_char = False
                            if line_length_count == self.LINE_LENGTH:
                                return True, actual_char

                    line_length_count = 1
                    while (checking_next_diagdown_char):
                        if (column + line_length_count > self.columns - 1) or (row + line_length_count > self.rows - 1):
                            checking_next_diagdown_char = False
                        else:
                            next_diagonal_char = self.board_char_representation[row + line_length_count, column + line_length_count]
                            if next_diagonal_char == actual_char:
                                line_length_count += 1
                            else:
                                checking_next_diagdown_char = False
                            if line_length_count == self.LINE_LENGTH:
                                return True, actual_char

                    line_length_count = 1
                    while (checking_next_diagup_char):
                        if (column + line_length_count > self.columns - 1) or (row - line_length_count < 0):
                            checking_next_diagup_char = False
                        else:
                            next_diagonal_char = self.board_char_representation[row - line_length_count, column + line_length_count]
                            if next_diagonal_char == actual_char:
                                line_length_count += 1
                            else:
                                checking_next_diagup_char = False
                            if line_length_count == self.LINE_LENGTH:
                                return True, actual_char

                    line_length_count = 1
                    while (checking_next_row_char):
                       if row + line_length_count > self.rows - 1:
                           checking_next_row_char = False
                       else:
                           next_row_char = self.board_char_representation[row+line_length_count, column]
                           if next_row_char == actual_char:
                               line_length_count += 1
                           else:
                               checking_next_row_char = False
                           if line_length_count == 4:
                               return True, actual_char
        return False, None
