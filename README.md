# Project-Gomoku

项目成员：

- 解润芃（自动化所，202118014628004）
- 倪子懿（自动化所，202128014628018）


此文件夹为《计算博弈原理与应用（2021年-2022年秋季学期）》的课程设计使用的代码及相关的报告。本README文件将对使用的代码，及其运行方法进行简要介绍。

### 代码环境

- Python 3.8
- PyTorch 1.10.0
- CUDA 11.3

### 实现代码

- MinMax 博弈树方法
    - 传统Alpha-Beta剪枝方法
    - 加入Zobrist置换表技术的方法
- 蒙特卡洛搜索树方法
- AlphaZero方法
- Muzero方法

### 运行代码

```
python play.py - 打开人机对弈的GUI界面（具体使用的对弈算法需要修改play.py）
python train_mcts.py - 训练AlphaZero（相应的config内容在config.py中进行修改）
python compare.py - 比较各种模型的性能
```

### 文件

```
\- MinMax - 带Alpha-Beta剪枝的MinMax搜索算法
\- MinMaxRefined - 基于Zobrist置换表技术的MinMax搜索算法
\- MCTS - 蒙特卡洛搜索方法及AlphaZero方法
\- MuZero_simplified - MuZero方法
\- utils - 辅助游戏运行的组件，例如棋盘、GUI等
- play.py - 运行下棋的 GUI 界面
- compare.py - 不同 AI 互相对局
- train_mcts.py - 训练 MCTS 方法
- config.py - 参数设定
```

### 参考代码

1. MinMax搜索相关代码参考：https://blog.csdn.net/marble_xu/article/details/90647361
2. AlphaZero相关代码参考：https://github.com/junxiaosong/AlphaZero_Gomoku
3. MuZero相关代码参考：https://github.com/werner-duvaud/muzero-general