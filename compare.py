from __future__ import print_function
import pickle
import torch
from utils.game import Board, Game
from MinMax.min_max_search import MinMaxSearchPlayer
from MCTS.mcts_pure import MCTSPlayer as MCTS_Pure
from MCTS.mcts_alphaZero import MCTSPlayer
from MCTS.policy_value_net_numpy import PolicyValueNetNumpy

version = "New"


def run():
    n = 5
    width, height = 8, 8
    # model_file = "data/models/current_policy.model"
    model_file = "data/models/best_policy_8_8_5.model"
    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)
        try:
            policy_param = pickle.load(open(model_file, "rb"))
        except:
            if version == "New":
                policy_param = list(torch.load(model_file).values())
            else:
                policy_param = pickle.load(open(model_file, "rb"), encoding="bytes")  # To support python3

        best_policy = PolicyValueNetNumpy(width, height, policy_param)
        mcts_player = MCTSPlayer(
            best_policy.policy_value_fn, c_puct=5, n_playout=400
        )  # set larger n_playout for better performance

        bot = MinMaxSearchPlayer(width, height)

        # set start_player=0 for human first
        players = {1: "min_max_player", 2: "alpha_zero_player", -1: "tie"}
        total_games = 5
        minmax_first_bot_win = {}
        for i in range(total_games):
            winner = game.start_play(bot, mcts_player, start_player=0, is_shown=1)
            if players[winner] not in minmax_first_bot_win.keys():
                minmax_first_bot_win[players[winner]] = 1
            else:
                minmax_first_bot_win[players[winner]] += 1
        print(minmax_first_bot_win)
        alpha_first_bot_win = {}
        for i in range(total_games):
            winner = game.start_play(bot, mcts_player, start_player=1, is_shown=1)
            if players[winner] not in alpha_first_bot_win.keys():
                alpha_first_bot_win[players[winner]] = 1
            else:
                alpha_first_bot_win[players[winner]] += 1
        print(alpha_first_bot_win)
        f_minmax = open("data/minmax_first_bot_win_10", "wb+")
        f_alpha = open("data/alpha_first_bot_win_10", "wb+")
        pickle.dump(minmax_first_bot_win, f_minmax)
        pickle.dump(alpha_first_bot_win, f_alpha)

    except KeyboardInterrupt:
        print("\n\rquit")


if __name__ == "__main__":
    run()
