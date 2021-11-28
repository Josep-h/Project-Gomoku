from __future__ import print_function
import pickle
import torch
from utils.game import Board, Game
from MinMax.min_max_search import MinMaxSearchPlayer
from MinMaxRefined.min_max_search import MinMaxRefinedSearchPlayer
from MCTS.mcts_pure import MCTSPlayer as MCTS_Pure
from MCTS.mcts_alphaZero import MCTSPlayer
from MCTS.policy_value_net_numpy import PolicyValueNetNumpy
from MCTS.policy_value_net_pytorch import PolicyValueNet

version = "New"

from config import config

c = config()


def load_min_max_player(depth):
    return MinMaxSearchPlayer(c.width, c.height, depth)


def load_min_max_player_refine(depth):
    return MinMaxRefinedSearchPlayer(c.width, c.height, depth)


def load_MCTS_player(play_out_num):
    return MCTS_Pure(c_puct=5, n_playout=play_out_num)


def load_AlphaZero_player(play_out_num=800):
    model_file = "data/models/best_policy.model"
    best_policy = PolicyValueNet(c.width, c.height, model_file)
    mcts_player = MCTSPlayer(
        best_policy.policy_value_fn, c_puct=5, n_playout=play_out_num
    )  # set larger n_playout for better performance
    return mcts_player


def get_player(name):
    if name == "MinMax-4":
        return load_min_max_player(4)
    if name == "MinMax-5":
        return load_min_max_player(5)
    if name == "MinMaxRefined-4":
        return load_min_max_player_refine(4)
    if name == "MinMaxRefined-5":
        return load_min_max_player_refine(5)
    if name == "MCTS2000":
        return load_MCTS_player(2000)
    if name == "MCTS4000":
        return load_MCTS_player(4000)
    if name == "AlphaZero":
        return load_AlphaZero_player()
    print("Name Error!!")


def run(player1, player2):
    n = c.n_in_row
    width, height = c.width, c.height
    board = Board(width=width, height=height, n_in_row=n)
    game = Game(board)

    # set start_player=0 for human first
    players = {1: player1, 2: player2, -1: "tie"}
    total_games = c.total_games
    player1_win = {}
    time_board = [0, 0]
    total_turn = 0
    count_, discount_ = 0, 0
    for i in range(total_games):
        player_bot1 = get_player(player1)
        player_bot2 = get_player(player2)
        print("Game: {}".format(i))
        winner, time_board_, turn = game.start_play(player_bot1, player_bot2, start_player=0, is_shown=c.show)
        time_board[0] += time_board_[0]
        time_board[1] += time_board_[1]
        total_turn += turn
        # count_ += player_bot1.search_tree.count_
        # discount_ += player_bot1.search_tree.discount_
        if players[winner] not in player1_win.keys():
            player1_win[players[winner]] = 1
        else:
            player1_win[players[winner]] += 1
    print(player1_win)
    player2_win = {}
    for i in range(total_games):
        player_bot1 = get_player(player1)
        player_bot2 = get_player(player2)
        print("Game: {}".format(i + c.total_games))
        winner, time_board_, turn = game.start_play(player_bot1, player_bot2, start_player=1, is_shown=c.show)
        time_board[0] += time_board_[0]
        time_board[1] += time_board_[1]
        total_turn += turn
        # count_ += player_bot1.search_tree.count_
        # discount_ += player_bot1.search_tree.discount_
        if players[winner] not in player2_win.keys():
            player2_win[players[winner]] = 1
        else:
            player2_win[players[winner]] += 1
    print(player2_win)
    f1 = open("data/{}_bot_win_10".format(player1), "wb+")
    f2 = open("data/{}_bot_win_10".format(player2), "wb+")
    pickle.dump(player1_win, f1)
    pickle.dump(player2_win, f2)
    return [time_board[0] / (total_turn / 2), time_board[1] / (total_turn / 2)], count_, discount_


if __name__ == "__main__":
    # for i in range(2):
    #     t = run("MinMaxRefined", "MCTS")
    #     time_board[0] += t[0]
    #     time_board[1] += t[1]
    # print(time_board)

    mean_time = 0
    # c.depth = d
    methods = [
        "MCTS2000",
        "MCTS4000",
        "MinMaxRefined-4",
        "MinMaxRefined-5",
    ]
    # methods = ["MCTS2000", "MCTS4000"]
    with torch.no_grad():
        for i in range(len(methods)):
            tp = c.total_games
            ma = "AlphaZero"
            mb = methods[i]
            print(ma, mb)
            t, count, discount = run(ma, mb)
            c.total_games = tp
