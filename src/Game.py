import pickle
from time import sleep

import numpy


from src.Engine import Engine,GameState
from src.Board import Board
from src.Logger import Logger
from src.Log import Log
from src.Log_Analyzer import Log_Analyzer
from src.AI import AI
from src.Basic_NN import Basic_NN
from src.Board_Utils import convert_to_opposite_matrix
import random
import os
import time
import tensorflow as tf

# Game Settings
PLAYER_A_IS_AI = True
PLAYER_B_IS_NN = True
PLAYER_B_IS_CONSOLE = False
TRAIN_AI = True

#
LOCATION = os.getcwd()
LOG_TIME = time.asctime()
LOG_NAME = 'CNN ' + LOG_TIME
AI_MODEL_NAME = 'CNN-131'
FILE_NAME_LOGGER = LOCATION+'/data/log/' + LOG_NAME + '.pkl'
AI_STEP = 132


# Initialize Game
game_engine = Engine()
if PLAYER_A_IS_AI:
    ai = AI('cnn')
    #ai.load(AI_MODEL_NAME)


board = game_engine.get_board()
logger = Logger()
player_chars = [Board.CHAR_PLAYER_A, Board.CHAR_PLAYER_B]
number_of_Games = 100000
print_increment = 10
game_is_over = False
count_games_a_has_won = 0
count_games_b_has_won = 0
games_played = 0
winning_rate_a = 0


#tensorboard stuff
tf.summary.scalar('winning_rate_a', winning_rate_a)
merged = tf.summary.merge_all()
test_writer = tf.summary.FileWriter(LOCATION + '/tensorboard')
sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

while number_of_Games > 0:
    actual_game_is_running = True
    actual_game_log = Log()
    game_steps = 0
    steps_player_a = 0
    steps_player_b = 0
    while actual_game_is_running:
        char = player_chars[(number_of_Games + game_steps) % 2]
        actual_game_state = GameState.TRY_TO_SET
        free_cells = board.NUM_ROWS * board.NUM_COLUMNS
        player_a_steps = []
        player_b_steps = []


        while actual_game_state == GameState.TRY_TO_SET:

            actions = Log_Analyzer(actual_game_log).get_all_actions(padding=True)[0:-1:2]
            actions = numpy.array(actions).reshape((1, 21, Board.NUM_COLUMNS))

            # Player A is playing
            if char == Board.CHAR_PLAYER_A:
                if PLAYER_A_IS_AI:
                    board_matrix_representation = game_engine.get_board().board_matrix.copy()
                    prediction = ai.predict(board_matrix_representation)
                    #print(prediction[0][steps_player_a])
                    #column = numpy.argmax(prediction[0][steps_player_a])
                    #print(["{0:0.4f}".format(i) for i in prediction])
                    column = numpy.argmax(prediction)
                else:
                    column = random.randint(0, Board.NUM_COLUMNS - 1)
                steps_player_a += 1
            # Player B is playing
            elif char == Board.CHAR_PLAYER_B:
                if PLAYER_B_IS_CONSOLE:
                    column = int(input("Please enter something: "))
                elif PLAYER_B_IS_NN  and steps_player_b % random.randint(1, 3) == 0:
                    board_matrix_representation = game_engine.get_board().board_matrix.copy()
                    prediction = ai.predict(board_matrix_representation)
                    #print(prediction[0][steps_player_b])
                    #column = numpy.argmax(prediction[0][steps_player_b])
                    #print(["{0:0.4f}".format(i) for i in prediction])
                    column = numpy.argmax(prediction)
                else:
                    column = random.randint(0, Board.NUM_COLUMNS - 1)
                steps_player_b += 1
            actual_game_state = game_engine.step(char, column)
            board_matrix_representation = game_engine.get_board().board_matrix.copy()
        actual_game_log.add_state(board_matrix_representation)
        game_steps += 1
        print(game_engine.board.board_char_representation)
        print(number_of_Games)

        # Check if game is over
        # Player A has won
        if actual_game_state == GameState.WON_BY_PLAYER_1:
            print(game_engine.board.board_char_representation)
            actual_game_is_running = False
            actual_game_log.add_winner(game_engine.get_board().X_OCCUPIED_CELL.copy())
            logger.add_log(actual_game_log)
            game_engine.get_board().clear()
            number_of_Games -= 1
            count_games_a_has_won += 1
            print('Game won by Player A')

        # Player B has won
        elif actual_game_state == GameState.WON_BY_PLAYER_2:
            print(game_engine.board.board_char_representation)
            actual_game_is_running = False
            actual_game_log.add_winner(game_engine.get_board().O_OCCUPIED_CELL.copy())
            logger.add_log(actual_game_log)
            game_engine.get_board().clear()
            number_of_Games -= 1
            count_games_b_has_won +=1
            print('Game won by Player B')

        # Nobody has won
        elif actual_game_state == GameState.REMIS:
            print(game_engine.board.board_char_representation)
            actual_game_is_running = False
            actual_game_log.add_winner(game_engine.get_board().EMPTY_CELL.copy())
            game_engine.get_board().clear()
            number_of_Games -= 1
            print('Remis')

        # Game ended due to set fail
        elif actual_game_state == GameState.SET_FAIL:
            print(game_engine.board.board_char_representation)
            actual_game_is_running = False
            actual_game_log.add_winner(game_engine.get_board().EMPTY_CELL.copy())
            game_engine.get_board().clear()
            number_of_Games -= 1
            print('Set Fail')

        games_played += 1
        actual_game_log.set_winning_rate('a', (count_games_a_has_won / games_played))
        actual_game_log.set_winning_rate('b', (count_games_b_has_won / games_played))

    #summary = sess.run([merged])
    #test_writer.add_summary(summary, game_steps)

    # Train NN with the last game
    if TRAIN_AI and actual_game_log.get_winner() != game_engine.get_board().EMPTY_CELL:
        print('Train AI')
        ai.train(actual_game_log)
        if number_of_Games % 100 == 0:
            ai.save(AI_STEP)

    if number_of_Games % print_increment == 0:
        print('Game '+ str(number_of_Games))
    print('winrate A: ' + str(count_games_a_has_won / games_played))

# Final stats
print('A: ' + str(count_games_a_has_won))
print('B: ' + str(count_games_b_has_won))
with open(FILE_NAME_LOGGER, 'wb') as output:  # Overwrites any existing file.
    pickle.dump(logger, output, pickle.HIGHEST_PROTOCOL)

