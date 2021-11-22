from __future__ import print_function
import pickle
import torch
from utils.game import Board, Game
from MinMax.min_max_search import MinMaxSearchPlayer
from MinMaxRefined.min_max_search import MinMaxRefinedSearchPlayer
from MCTS.mcts_pure import MCTSPlayer as MCTS_Pure
from MCTS.mcts_alphaZero import MCTSPlayer
from MCTS.policy_value_net_numpy import PolicyValueNetNumpy

version = "New"

from config import config

c = config()


def load_min_max_player():
    return MinMaxSearchPlayer(c.width, c.height)


def load_min_max_player_refine():
    return MinMaxRefinedSearchPlayer(c.width, c.height)


def load_MCTS_player():
    return MCTS_Pure(c_puct=5, n_playout=c.play_out_num)


def load_AlphaZero_player():
    policy_param = pickle.load(open(c.init_model, "rb"), encoding="bytes")
    best_policy = PolicyValueNetNumpy(c.width, c.height, policy_param)
    mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=c.n_in_row, n_playout=c.play_out_num)
    return mcts_player


def get_player(name):
    if name == "MinMax":
        return load_min_max_player()
    if name == "MinMaxRefined":
        return load_min_max_player_refine()
    if name == "MCTS":
        return load_MCTS_player()
    if name == "AlphaZero":
        return load_AlphaZero_player()
    print("Name Error!!")


def run(player1, player2):
    n = c.n_in_row
    width, height = c.width, c.height
    board = Board(width=width, height=height, n_in_row=n)
    game = Game(board)

    player_bot1 = get_player(player1)
    player_bot2 = get_player(player2)

    # set start_player=0 for human first
    players = {1: player1, 2: player2, -1: "tie"}
    total_games = 5
    player1_win = {}
    time_board = [0, 0]
    for i in range(total_games):
        winner, time_board_ = game.start_play(player_bot1, player_bot2, start_player=0, is_shown=c.show)
        time_board[0] += time_board_[0]
        time_board[1] += time_board_[1]
        if players[winner] not in player1_win.keys():
            player1_win[players[winner]] = 1
        else:
            player1_win[players[winner]] += 1
    print(player1_win)
    player2_win = {}
    for i in range(total_games):
        winner, time_board_ = game.start_play(player_bot1, player_bot2, start_player=1, is_shown=c.show)
        time_board[0] += time_board_[0]
        time_board[1] += time_board_[1]
        if players[winner] not in player2_win.keys():
            player2_win[players[winner]] = 1
        else:
            player2_win[players[winner]] += 1
    print(player2_win)
    f1 = open("data/{}_bot_win_10".format(player1), "wb+")
    f2 = open("data/{}_bot_win_10".format(player2), "wb+")
    pickle.dump(player1_win, f1)
    pickle.dump(player2_win, f2)
    return time_board


if __name__ == "__main__":
    time_board = [0, 0]
    for i in range(2):
        t = run("MinMaxRefined", "MCTS")
        time_board[0] += t[0]
        time_board[1] += t[1]
    print(time_board)
