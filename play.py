# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""

from __future__ import print_function
import pickle

import torch
from utils.game import Board, Game
from MinMax.min_max_search import MinMaxSearchPlayer
from MinMaxRefined.min_max_search import MinMaxRefinedSearchPlayer
from MCTS.policy_value_net_numpy import PolicyValueNetNumpy
from MCTS.policy_value_net_pytorch import PolicyValueNet
from MCTS.mcts_alphaZero import MCTSPlayer

# from policy_value_net import PolicyValueNet  # Theano and Lasagne
# from policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet  # Keras

from config import config

c = config()


def run(model):
    # 可选的模型与棋盘大小有：
    # 8*8棋盘的五子连珠：best_policy_8_8_5.model
    # 6*6棋盘的四子连珠：best_policy_6_6_4.model
    # 极大极小值搜索在任意大小的棋盘五子连珠
    try:
        board = Board(width=c.width, height=c.height, n_in_row=c.n_in_row)
        game = Game(board)
        if model == "MCTS 8x8":
            board = Board(width=8, height=8, n_in_row=c.n_in_row)
            game = Game(board)
            model_file = "data/models/best_policy_8_8_5.model"
            policy_param = pickle.load(open(model_file, "rb"), encoding="bytes")
            best_policy = PolicyValueNetNumpy(8, 8, policy_param)
            # alpha zero 蒙特卡洛搜索
            mcts_player = MCTSPlayer(
                best_policy.policy_value_fn, c_puct=5, n_playout=400
            )  # set larger n_playout for better performance
            game.start_play_with_UI(mcts_player, start_player=1)
        elif model == "MinMax":
            # 极大极小值搜索
            minmax_player = MinMaxSearchPlayer(c.width, c.height)
            game.start_play_with_UI(minmax_player, start_player=1)
        elif model == "Refined":
            minmax_player = MinMaxRefinedSearchPlayer(c.width, c.height)
            game.start_play_with_UI(minmax_player, start_player=1)
        else:
            model_file = "data/models/best_policy.model"
            best_policy = PolicyValueNet(c.width, c.height, model_file)
            mcts_player = MCTSPlayer(
                best_policy.policy_value_fn, c_puct=5, n_playout=800
            )  # set larger n_playout for better performance
            game.start_play_with_UI(mcts_player, start_player=1)

    except KeyboardInterrupt:
        print("\n\rquit")


# model = "MCTS 8x8"
model = "MCTS"
# model = "MinMaxRefined"
# model = "MinMax"

if __name__ == "__main__":
    run(model)
