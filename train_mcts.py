# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
"""

from __future__ import print_function
import random
from time import time
import numpy as np
from collections import defaultdict, deque
from utils.game import Board, Game
from MCTS.mcts_pure import MCTSPlayer as MCTS_Pure
from MCTS.mcts_alphaZero import MCTSPlayer
from MinMaxRefined.min_max_search import MinMaxRefinedSearchPlayer

# from policy_value_net import PolicyValueNet  # Theano and Lasagne
from MCTS.policy_value_net_pytorch import PolicyValueNet  # Pytorch
import torch

# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet # Keras
from config import config

c = config()


class TrainPipeline:
    def __init__(self):
        # params of the board and the game
        self.board = Board(width=c.width, height=c.height, n_in_row=c.n_in_row)
        self.game = Game(self.board)
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 400  # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)

        self.epochs = 10  # num of train_steps for each update
        self.kl_targ = 0.02

        self.game_batch_num = 1500
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000
        if c.init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(c.width, c.height, model_file=c.init_model, use_gpu=c.use_gpu)
            checkpoint = torch.load(c.init_model)
            self.lr_multiplier = checkpoint["lr_m"]
            self.learn_rate = checkpoint["lr"]
            # self.learn_rate = 0.01
            # self.data_buffer = checkpoint["data_buffer"]
            print("data_load_done!")
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(c.width, c.height, use_gpu=c.use_gpu)
        self.mcts_player = MCTSPlayer(
            self.policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout, is_selfplay=1
        )

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(mcts_porb.reshape(c.height, c.width)), i)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        with torch.no_grad():
            for i in range(n_games):
                winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
                play_data = list(play_data)[:]
                self.episode_len = len(play_data)
                # augment the data
                play_data = self.get_equi_data(play_data)
                self.data_buffer.extend(play_data)

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        if c.use_gpu:
            old_probs = old_probs.cpu().numpy()
        else:
            old_probs = old_probs.numpy()
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch, mcts_probs_batch, winner_batch, self.learn_rate * self.lr_multiplier
            )
            with torch.no_grad():
                new_probs, new_v = self.policy_value_net.policy_value(state_batch)
                if c.use_gpu:
                    new_probs = new_probs.cpu().numpy()
                else:
                    new_probs = new_probs.numpy()
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        # explained_var_old = 1 - np.var(np.array(winner_batch) - old_v.flatten()) / np.var(np.array(winner_batch))
        # explained_var_new = 1 - np.var(np.array(winner_batch) - new_v.flatten()) / np.var(np.array(winner_batch))
        print(
            (
                "kl:{:.5f},"
                "lr_multiplier:{:.3f},"
                "loss:{},"
                "entropy:{},"
                # "explained_var_old:{:.3f},"
                # "explained_var_new:{:.3f}"
            ).format(kl, self.lr_multiplier, loss, entropy)
        )
        return loss, entropy

    def policy_evaluate(self):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(
            self.policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout
        )
        # pure_mcts_player = MCTS_Pure(c_puct=5, n_playout=self.pure_mcts_playout_num)
        pure_mcts_player = MinMaxRefinedSearchPlayer(c.width, c.height)
        win_cnt = defaultdict(int)
        for i in range(c.n_games):
            winner, _, _ = self.game.start_play(current_mcts_player, pure_mcts_player, start_player=i % 2, is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / c.n_games
        print(
            "num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num, win_cnt[1], win_cnt[2], win_cnt[-1]
            )
        )
        return win_ratio

    def run(self):
        """run the training pipeline"""
        try:
            for i in range(self.game_batch_num):
                time1 = time()
                self.collect_selfplay_data(c.play_batch_size)
                time2 = time()
                print("[{:.5}] batch i:{}, episode_len:{}".format(time2 - time1, i + 1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                # check the performance of the current model,
                # and save the model params
                if (i + 1) % c.check_freq == 0:
                    print("current self-play batch: {}".format(i + 1))
                    win_ratio = self.policy_evaluate()

                    self.policy_value_net.save_model(
                        "data/models/current_policy_{}.model".format(i),
                        i,
                        self.learn_rate,
                        self.lr_multiplier,
                        self.data_buffer,
                    )
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.save_model(
                            "data/models/best_policy.model",
                            i,
                            self.learn_rate,
                            self.lr_multiplier,
                            self.data_buffer,
                        )
                        if self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 5000:
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print("\n\rquit")


if __name__ == "__main__":
    training_pipeline = TrainPipeline()
    training_pipeline.run()
