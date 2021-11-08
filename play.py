# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""

from __future__ import print_function
import pickle
from utils.game import Board, Game
from MinMax.min_max_search import MinMaxSearchPlayer

# from policy_value_net import PolicyValueNet  # Theano and Lasagne
# from policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet  # Keras


def run():
    # 可选的模型与棋盘大小有：
    # 8*8棋盘的五子连珠：best_policy_8_8_5.model
    # 6*6棋盘的四子连珠：best_policy_6_6_4.model
    # 极大极小值搜索在任意大小的棋盘五子连珠
    n = 5
    width, height = 15, 15
    # model_file = 'best_policy_8_8_5.model'
    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)
        # try:
        #     policy_param = pickle.load(open(model_file, 'rb'))
        # except:
        #     policy_param = pickle.load(open(model_file, 'rb'),
        #                                encoding='bytes')  # To support python3
        # best_policy = PolicyValueNetNumpy(width, height, policy_param)

        # 极大极小值搜索
        minmax_player = MinMaxSearchPlayer(width, height)
        # alpha zero 蒙特卡洛搜索
        # mcts_player = MCTSPlayer(best_policy.policy_value_fn,
        #                          c_puct=5,
        #                          n_playout=400)  # set larger n_playout for better performance
        game.start_play_with_UI(minmax_player, start_player=1)
    except KeyboardInterrupt:
        print("\n\rquit")


if __name__ == "__main__":
    run()
