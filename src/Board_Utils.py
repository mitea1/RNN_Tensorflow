import numpy as np


def convert_to_opposite_matrix(board_matrix):
    rows, columns, cells = np.shape(board_matrix)
    for i in range(0, rows):
        for j in range(0, columns):
            if board_matrix[i][j][0] == 1 and board_matrix[i][j][1] == 0:
                board_matrix[i][j][0] = 0
                board_matrix[i][j][1] = 1
            elif board_matrix[i][j][0] == 0 and board_matrix[i][j][1] == 1:
                board_matrix[i][j][0] = 1
                board_matrix[i][j][1] = 0

    return board_matrix


def convert_to_opposite_array(array):
    if array[0] == 1 and array[1] == 0:
        array[0] = 0
        array[1] = 1
    elif array[0] == 0 and array[1] == 1:
        array[0] = 1
        array[1] = 0
    return array