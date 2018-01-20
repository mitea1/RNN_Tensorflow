import random

from src.Board import Board


class Engine:

    def __init__(self):
        self.board = Board()

    def step(self, player_char, column):
        step_success = False
        game_is_running = True
        for i in range(0, Board.NUM_ROWS):
            row = Board.NUM_ROWS - 1 - i
            if self.board.board_char_representation[row][column] == Board.CHAR_EMPTY:
                if player_char == Board.CHAR_PLAYER_A:
                    self.board.set_cell_state(row, column, Board.X_OCCUPIED_CELL)
                    step_success = True
                    break
                elif player_char == Board.CHAR_PLAYER_B:
                    self.board.set_cell_state(row, column, Board.O_OCCUPIED_CELL)
                    step_success = True
                    break
        self.board.convert_to_char_representation()
        self.board.print_char_representation()
        are_4_connected, char = self.board.are_4_connected()
        if(are_4_connected):
            print("4 connected by "+ char)
            game_is_running = False
        return step_success, game_is_running

    def get_board(self):
        return self.board


eng = Engine()
player_char = [Board.CHAR_PLAYER_A, Board.CHAR_PLAYER_B]
number_of_Games = 10
game_is_over = False

print(eng.get_board().NUM_COLUMNS)

while number_of_Games > 0:
    is_running = True
    i = 0
    while is_running:
        char = player_char[i%2]
        step_was_successful = False
        while not step_was_successful:
            column = random.randint(0, Board.NUM_COLUMNS-1)
            step_was_successful, is_running = eng.step(char, column)
        i += 1
    print(eng.get_board().NUM_COLUMNS)
    eng.get_board().clear()
    number_of_Games -= 1

print(eng.get_board().board_matrix)