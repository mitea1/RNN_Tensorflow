import pickle

from src.Engine import Engine,GameState
from src.Board import Board
from src.Logger import Logger
from src.Log import Log
import random
import os

LOCATION = os.getcwd()
FILE_NAME_LOGGER = LOCATION+'/data/log/logger.pkl'
eng = Engine()
board = eng.get_board()
logger = Logger()
player_char = [Board.CHAR_PLAYER_A, Board.CHAR_PLAYER_B]
number_of_Games = 20000
print_increment = 100
game_is_over = False

print(board.NUM_COLUMNS)

while number_of_Games > 0:
    is_running = True
    i = 0
    log_entry = Log()
    while is_running:
        char = player_char[i % 2]
        game_state = GameState.TRY_TO_SET
        free_cells = 6 * 7
        while game_state == GameState.TRY_TO_SET:
            column = random.randint(0, Board.NUM_COLUMNS - 1)
            game_state = eng.step(char, column)
            board_matrix_representation = eng.get_board().board_matrix.copy()
        log_entry.add_state(board_matrix_representation)
        i += 1

        # Check if game is over
        if game_state == GameState.WON_BY_PLAYER_1:
            is_running = False
            log_entry.add_winner(eng.get_board().X_OCCUPIED_CELL.copy())
            logger.add_log(log_entry)
            eng.get_board().clear()
            number_of_Games -= 1
        elif game_state == GameState.WON_BY_PLAYER_2:
            is_running = False
            log_entry.add_winner(eng.get_board().O_OCCUPIED_CELL.copy())
            logger.add_log(log_entry)
            eng.get_board().clear()
            number_of_Games -= 1
        elif game_state == GameState.REMIS:
            is_running = False
            log_entry.add_winner(eng.get_board().EMPTY_CELL.copy())
            eng.get_board().clear()
            number_of_Games -= 1
    if number_of_Games % print_increment == 0:
        print('Game '+ str(number_of_Games))

print(logger.get_log(0).get_next_step(2))
print(logger.get_log(0).get_next_step(3))
with open(FILE_NAME_LOGGER, 'wb') as output:  # Overwrites any existing file.
    pickle.dump(logger, output, pickle.HIGHEST_PROTOCOL)

