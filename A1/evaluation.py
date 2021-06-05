import numpy as np


# Inspired by Keith Galli's evaluate_window function: 
# https://github.com/KeithGalli/Connect4-Python/blob/503c0b4807001e7ea43a039cb234a4e55c4b226c/connect4_with_ai.py
# Lines 67-83
def evaluate_window(window, disc):
    score = 0
    opp_disc = -1 if disc == 1 else 1
    
    # Victory
    # if window.count(disc) == 4:
    #     score += 100000
    # Possible win with 1 move
    if window.count(disc) == 3 and window.count(0) == 1:
	    score += 100
    # Possible win with 2 moves
    elif window.count(disc) == 2 and window.count(0) == 2:
	    score += 1

    # if window.count(opp_disc) == 4:
    #     score -= 100000
    # Possible loss with 1 move
    elif window.count(opp_disc) == 3 and window.count(0) == 1:
	    score -= 100
    elif window.count(opp_disc) == 2 and window.count(0) == 2:
        score -= 1

    return score


# Inspired by Keith Galli's score_position function: 
# https://github.com/KeithGalli/Connect4-Python/blob/503c0b4807001e7ea43a039cb234a4e55c4b226c/connect4_with_ai.py
# Lines 85-118
def utility(board, discType):
    score = 0
    ROW_COUNT = len(board)
    COLUMN_COUNT = len(board[0])
    WINDOW_LENGTH = 4
    # Score center column
    # center_array = [int(i) for i in list(board[:, COLUMN_COUNT//2])]
    # center_count = center_array.count(discType)
    # score += center_count * 3

	# Score horizontal
    for r in range(ROW_COUNT):
        row_array = list(board[r])
        for c in range(COLUMN_COUNT-3):
            window = row_array[c:c+WINDOW_LENGTH]
            score += evaluate_window(window, discType)

	# Score vertical
    for c in range(COLUMN_COUNT):
        col_array =list(board[:,c])
        for r in range(ROW_COUNT-3):
            window = col_array[r:r+WINDOW_LENGTH]
            score += evaluate_window(window, discType)

	# Score positive diagonal
    for r in range(ROW_COUNT-3):
        for c in range(COLUMN_COUNT-3):
            window = [board[r+i][c+i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, discType)

   # Score negative diagonal
    for r in range(ROW_COUNT-3):
        for c in range(COLUMN_COUNT-3):
            window = [board[r+3-i][c+i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, discType)

    return score