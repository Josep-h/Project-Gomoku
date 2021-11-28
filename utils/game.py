# -*- coding: utf-8 -*-
"""
@author: Junxiao Song
"""


from __future__ import print_function

import sys

sys.path.append("..")

import numpy as np
from utils.GUI_v1_4 import GUI
from config import config

c = config()


class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get("width", 8))
        self.height = int(kwargs.get("height", 8))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get("n_in_row", 5))
        self.players = [1, 2]  # player1 and player2

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception("board width and height can not be " "less than {}".format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """

        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width, move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width, move_oppo % self.height] = 1.0
            # indicate the last move location
            square_state[2][self.last_move // self.width, self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = self.players[0] if self.current_player == self.players[1] else self.players[1]
        self.last_move = move

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row * 2 - 1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if w in range(width - n + 1) and len(set(states.get(i, -1) for i in range(m, m + n))) == 1:
                return True, player

            if h in range(height - n + 1) and len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1:
                return True, player

            if (
                w in range(width - n + 1)
                and h in range(height - n + 1)
                and len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1
            ):
                return True, player

            if (
                w in range(n - 1, width)
                and h in range(height - n + 1)
                and len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1
            ):
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class Game(object):
    """game server"""

    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end="")
        print("\r\n")
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end="")
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print("X".center(8), end="")
                elif p == player2:
                    print("O".center(8), end="")
                else:
                    print("_".center(8), end="")
            print("\r\n\r\n")

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception("start_player should be either 0 (player1 first) " "or 1 (player2 first)")
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        time_board = [0, 0]
        turn = 0
        while True:
            # print(turn)
            turn += 1
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move, _, time_ = player_in_turn.get_action(self.board)
            time_board[current_player - 1] += time_
            if isinstance(move, tuple):
                move = (c.width - move[1] - 1) * c.width + move[0]
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    print(time_board)
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner, time_board, turn

    def start_play_with_UI(self, AI, start_player=0):
        """
        a GUI for playing
        """
        AI.reset_player()
        AI.set_player_ind(1)
        self.board.init_board()
        current_player = SP = start_player
        UI = GUI(self.board.width)
        end = False
        num = -1
        winner = 0
        while True:
            print("current_player", current_player)
            if current_player == 0:
                UI.show_messages("Your turn")
            else:
                UI.show_messages("AI's turn")
            if current_player == 1 and not end:
                # move, move_probs = AI.get_action(self.board, is_selfplay=False, print_probs_value=1)
                move, move_probs, time_ = AI.get_action(self.board)
                num = num + 1
                print("move", move)
            else:
                if end:
                    if winner == 1:
                        UI.show_messages("Winner is black")
                    elif winner == 2:
                        UI.show_messages("Winner is white")
                    else:
                        UI.show_messages("Tie")
                inp = UI.get_input()
                if inp[0] == "move" and not end:
                    if type(inp[1]) != int:
                        move = UI.loc_2_move(inp[1])
                    else:
                        move = inp[1]
                    num = num + 1
                elif inp[0] == "RestartGame":
                    end = False
                    current_player = SP
                    self.board.init_board()
                    UI.restart_game()
                    AI.reset_player()
                    num = -1
                    winner = 0
                    continue
                elif inp[0] == "ResetScore":
                    UI.reset_score()
                    continue
                elif inp[0] == "quit":
                    exit()
                    continue
                elif inp[0] == "SwitchPlayer":
                    end = False
                    self.board.init_board()
                    UI.restart_game(False)
                    UI.reset_score()
                    AI.reset_player()
                    num = -1
                    winner = 0
                    SP = (SP + 1) % 2
                    current_player = SP
                    continue
                else:
                    # print('ignored inp:', inp)
                    continue
            # print('player %r move : %r'%(current_player,[move//self.board.width,move%self.board.width]))
            if not end:
                # print(move, type(move), current_player)
                UI.render_step(move, self.board.current_player, num)
                self.board.do_move(move)
                # print('move', move)
                # print(2, self.board.get_current_player())
                current_player = (current_player + 1) % 2
                # UI.render_step(move, current_player)
                end, winner = self.board.game_end()
                if end:
                    # print(AI.search_tree.discount_ / float(AI.search_tree.count_))
                    if winner != -1:
                        print("Winner is player", winner)
                        UI.add_score(winner)
                    else:
                        print("Game end. Tie")
                    print(UI.score)

    def start_AI_play_with_UI(self, AI1, AI2, start_player=0):
        """
        a GUI for playing
        """
        AI1.reset_player()
        AI2.reset_player()
        self.board.init_board()
        current_player = SP = start_player
        UI = GUI(self.board.width)
        end = False
        num = -1
        while True:
            print("current_player", current_player)
            if current_player == 0:
                UI.show_messages("AI1 turn")
                num = num + 1
            else:
                UI.show_messages("AI2's turn")
                num += 1
            if current_player == 1 and not end:
                # move, move_probs = AI.get_action(self.board, is_selfplay=False, print_probs_value=1)
                move, move_probs, time_ = AI2.get_action(self.board)
                print("move", move)
            else:
                if current_player == 0 and not end:
                    move, move_probs, time_ = AI1.get_action(self.board)
                    print("move2", move)
                if end:

                    continue
                # else:
                # break
            # print('player %r move : %r'%(current_player,[move//self.board.width,move%self.board.width]))
            if not end:
                # print(move, type(move), current_player)
                UI.render_step(move, self.board.current_player, num)
                self.board.do_move(move)
                # print('move', move)
                # print(2, self.board.get_current_player())
                current_player = (current_player + 1) % 2
                # UI.render_step(move, current_player)
                end, winner = self.board.game_end()
                if end:
                    if winner != -1:
                        print("Game end. Winner is player %d" % winner)
                        UI.add_score(winner)
                    else:
                        print("Game end. Tie")
                    print(UI.score)
                    print()

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs, time_ = player.get_action(self.board, temp=temp, return_prob=1)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
