from enum import Enum


class GameState(Enum):
    OFFLINE = -1
    RUNNING = 0
    TRY_TO_SET = 1
    SET_SUCCESSFUL = 2
    WON_BY_PLAYER_1 = 3
    WON_BY_PLAYER_2 = 4
    REMIS = 5