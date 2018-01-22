from enum import Enum

from src.Board import Board


class GameState(Enum):
    OFFLINE = -1
    RUNNING = 0
    TRY_TO_SET = 1
    SET_SUCCESSFUL = 2
    WON_BY_PLAYER_1 = 3
    WON_BY_PLAYER_2 = 4
    REMIS = 5

class Engine:



    def __init__(self):
        self.board = Board()
        self.state = GameState.OFFLINE

    def step(self, player_char, column):
        game_state = GameState.TRY_TO_SET
        # Try to set
        for i in range(0, Board.NUM_ROWS):
            row = Board.NUM_ROWS - 1 - i
            # Find empty cells
            if self.board.board_char_representation[row][column] == Board.CHAR_EMPTY:
                # Set sells according to player
                if player_char == Board.CHAR_PLAYER_A:
                    self.board.set_cell_state(row, column, Board.X_OCCUPIED_CELL)
                    game_state = GameState.SET_SUCCESSFUL
                    break
                elif player_char == Board.CHAR_PLAYER_B:
                    self.board.set_cell_state(row, column, Board.O_OCCUPIED_CELL)
                    game_state = GameState.SET_SUCCESSFUL
                    break
        self.board.convert_to_char_representation()
        are_4_connected, connected_char = self.board.are_4_connected()
        free_cells = (self.board.board_char_representation == '_').sum()
        # Somebody won find out who
        if are_4_connected:
            if connected_char == Board.CHAR_PLAYER_A:
                game_state = GameState.WON_BY_PLAYER_1
            elif connected_char == Board.CHAR_PLAYER_B:
                game_state = GameState.WON_BY_PLAYER_2
        # No free cells but nobody won
        if free_cells == 0:
            game_state = GameState.REMIS
        return game_state

    def get_board(self):
        return self.board
