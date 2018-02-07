import pickle
from time import sleep

import numpy


from src.Engine import Engine,GameState
from src.Board import Board
from src.Logger import Logger
from src.Log import Log
from src.RNN_Model import RNN_Model
import random
import os
import time

USE_NN = True
PLAY_CONSOLE = True
LOCATION = os.getcwd()
LOG_TIME = time.asctime()
FILE_NAME_LOGGER = LOCATION+'/data/log/logger' + LOG_TIME + '.pkl'
eng = Engine()
if USE_NN:
    rnn = RNN_Model()
    #rnn.load('my_test_model-1000')
    rnn.train()
    #rnn.save()
board = eng.get_board()
logger = Logger()
player_char = [Board.CHAR_PLAYER_A, Board.CHAR_PLAYER_B]
number_of_Games = 20000
print_increment = 100
game_is_over = False
a_has_won = 0
b_has_won = 0

print(board.NUM_COLUMNS)

i = 0
while number_of_Games > 0:
    is_running = True
    log_entry = Log()
    #print(eng.board.board_char_representation)
    while is_running:
        char = player_char[i % 2]
        game_state = GameState.TRY_TO_SET
        free_cells = 6 * 7
        while game_state == GameState.TRY_TO_SET:
            if char == Board.CHAR_PLAYER_A:
                if USE_NN:
                    board_matrix_representation = eng.get_board().board_matrix.copy()
                    prediction = rnn.predict(board_matrix_representation)
                    max_index = numpy.argmax(prediction)
                    column = max_index % Board.NUM_COLUMNS
                else:
                    column = random.randint(0, Board.NUM_COLUMNS - 1)
            elif char == Board.CHAR_PLAYER_B:
                if PLAY_CONSOLE:
                    column = int(input("Please enter something: "))
                else:
                    column = random.randint(0, Board.NUM_COLUMNS - 1)

            #column = random.randint(0, Board.NUM_COLUMNS - 1)
            game_state = eng.step(char, column)
            board_matrix_representation = eng.get_board().board_matrix.copy()
        log_entry.add_state(board_matrix_representation)
        i += 1
        print(eng.board.board_char_representation)
        print(number_of_Games)
        # Check if game is over
        if game_state == GameState.WON_BY_PLAYER_1:
            is_running = False
            log_entry.add_winner(eng.get_board().X_OCCUPIED_CELL.copy())
            logger.add_log(log_entry)
            eng.get_board().clear()
            number_of_Games -= 1
            a_has_won += 1
            print('Game won by Player A')
        elif game_state == GameState.WON_BY_PLAYER_2:
            is_running = False
            log_entry.add_winner(eng.get_board().O_OCCUPIED_CELL.copy())
            logger.add_log(log_entry)
            eng.get_board().clear()
            number_of_Games -= 1
            b_has_won +=1
            print('Game won by Player B')
        elif game_state == GameState.REMIS:
            is_running = False
            log_entry.add_winner(eng.get_board().EMPTY_CELL.copy())
            eng.get_board().clear()
            number_of_Games -= 1
            print('Remis')
    if number_of_Games % print_increment == 0:
        print('Game '+ str(number_of_Games))

print('A: '+str(a_has_won))
print('B: '+str(b_has_won))
print(logger.get_log(0).get_next_step(2))
print(logger.get_log(0).get_next_step(3))
with open(FILE_NAME_LOGGER, 'wb') as output:  # Overwrites any existing file.
    pickle.dump(logger, output, pickle.HIGHEST_PROTOCOL)

