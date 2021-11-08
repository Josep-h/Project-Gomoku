## 智能博弈五子棋AI设计

本程序为智能博弈五子棋AI的设计，主要包括极大极小值搜索和alpha zero算法在五子棋上的实现。
极大极小值搜索的实现参考了csdn博客https://blog.csdn.net/marble_xu/article/details/90647361，
alpha zero的实现参考了https://github.com/junxiaosong/AlphaZero_Gomoku

#### 代码结构：
- game.py 五子棋游戏逻辑实现
- GUI_v1_4.py UI界面程序的实现
- human_play.py 运行入口
- bot_play.py 机器对弈脚本
- mcts_alphazero.py 基于alphazero的蒙特卡洛树搜索
- mcts_pure.py 纯蒙特卡洛树搜索
- min_max_search.py 极大极小值搜索实现
- num_define.py 定义了极大极小值搜索配置的一些数值
- policy_value_net_numpy.py、policy_value_net_pytorch.py策略价值网络实现，numpy为原实现，pytorch版本为对模型修改的版本。
- train.py 模型训练脚本

#### 程序运行方式：
执行human_play.py即可运行人机对战
训练执行train.py

#### 环境要求：
pytorch、numpy、pygame