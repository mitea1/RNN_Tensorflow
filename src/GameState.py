from enum import Enum


class GameState(Enum):
    OFFLINE = -1
    RUNNING = 0
    TRY_TO_SET = 1
    SET_SUCCESSFUL = 2
    SET_FAIL = 3
    WON_BY_PLAYER_1 = 4
    WON_BY_PLAYER_2 = 5
    REMIS = 6