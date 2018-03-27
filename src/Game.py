import pickle
from time import sleep

import numpy


from src.Engine import Engine,GameState
from src.Board import Board
from src.Logger import Logger
from src.Log import Log
from src.AI import AI
from src.Basic_NN import Basic_NN
from src.Board_Utils import convert_to_opposite_matrix
import random
import os
import time

USE_NN = False
TRAIN_NN = False
PLAY_AGAINST_NN = False
PLAY_CONSOLE = False
LOCATION = os.getcwd()
LOG_TIME = time.asctime()
LOG_NAME = 'Basic_NN ' + LOG_TIME
FILE_NAME_LOGGER = LOCATION+'/data/log/' + LOG_NAME + '.pkl'
eng = Engine()
if USE_NN:
    ai = Basic_NN()
    #ai.load('Basic_NN-125')
    #rnn.train()
    #rnn.save()
board = eng.get_board()
logger = Logger()
player_char = [Board.CHAR_PLAYER_A, Board.CHAR_PLAYER_B]
number_of_Games = 10000
print_increment = 10
game_is_over = False
number_games_a_has_won = 0
number_games_b_has_won = 0

print(board.NUM_COLUMNS)

i = 0
while number_of_Games > 0:
    is_running = True
    log_entry = Log()

    while is_running:
        char = player_char[i % 2]
        game_state = GameState.TRY_TO_SET
        free_cells = board.NUM_ROWS * board.NUM_COLUMNS

        while game_state == GameState.TRY_TO_SET:

            # Player A is playing
            if char == Board.CHAR_PLAYER_A:
                if USE_NN:
                    board_matrix_representation = eng.get_board().board_matrix.copy()
                    prediction = ai.predict(board_matrix_representation)
                    print(prediction)
                    max_index = numpy.argmax(prediction)
                    column = max_index % Board.NUM_COLUMNS
                else:
                    column = random.randint(0, Board.NUM_COLUMNS - 1)

            # Player B is playing
            elif char == Board.CHAR_PLAYER_B:
                if PLAY_CONSOLE:
                    column = int(input("Please enter something: "))
                elif PLAY_AGAINST_NN:
                    board_matrix_representation = eng.get_board().board_matrix.copy()
                    board_matrix_representation = convert_to_opposite_matrix(board_matrix_representation)
                    prediction = ai.predict(board_matrix_representation)
                    max_index = numpy.argmax(prediction)
                    column = max_index % Board.NUM_COLUMNS
                else:
                    column = random.randint(0, Board.NUM_COLUMNS - 1)

            game_state = eng.step(char, column)
            board_matrix_representation = eng.get_board().board_matrix.copy()
        log_entry.add_state(board_matrix_representation)
        i += 1
        print(eng.board.board_char_representation)
        #print(number_of_Games)
        # Check if game is over
        # Player A has won
        if game_state == GameState.WON_BY_PLAYER_1:
            print(eng.board.board_char_representation)
            is_running = False
            log_entry.add_winner(eng.get_board().X_OCCUPIED_CELL.copy())
            logger.add_log(log_entry)
            eng.get_board().clear()
            number_of_Games -= 1
            number_games_a_has_won += 1
            print('Game won by Player A')

        # Player B has won
        elif game_state == GameState.WON_BY_PLAYER_2:
            print(eng.board.board_char_representation)
            is_running = False
            log_entry.add_winner(eng.get_board().O_OCCUPIED_CELL.copy())
            logger.add_log(log_entry)
            eng.get_board().clear()
            number_of_Games -= 1
            number_games_b_has_won +=1
            print('Game won by Player B')

        # Nobody has won
        elif game_state == GameState.REMIS:
            print(eng.board.board_char_representation)
            is_running = False
            log_entry.add_winner(eng.get_board().EMPTY_CELL.copy())
            eng.get_board().clear()
            number_of_Games -= 1
            print('Remis')

        # Game ended due to set fail
        elif game_state == GameState.SET_FAIL:
            print(eng.board.board_char_representation)
            is_running = False
            log_entry.add_winner(eng.get_board().EMPTY_CELL.copy())
            eng.get_board().clear()
            number_of_Games -= 1
            print('Set Fail')


    # Train NN with the last game
    if TRAIN_NN:
        ai.train(log_entry)
        if number_of_Games % 1000 == 0:
            ai.save(126)

    if number_of_Games % print_increment == 0:
        print('Game '+ str(number_of_Games))

# Final stats
print('A: ' + str(number_games_a_has_won))
print('B: ' + str(number_games_b_has_won))
with open(FILE_NAME_LOGGER, 'wb') as output:  # Overwrites any existing file.
    pickle.dump(logger, output, pickle.HIGHEST_PROTOCOL)

