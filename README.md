# Project-Gomoku

A project of computational game theory.

## 代码环境

- Python 3.8
- PyTorch 1.10.0
- CUDA 11.3

## 实现

计划算法分为个：

- MinMax 博弈树方法（传统Alpha-Beta剪枝方法、加入Zobrist技术）
- 蒙特卡洛搜索树方法
- AlphaGo Zero模型
- Muzero模型

参考代码在 `/AlphaZero_Gomoku` 中。

## 进展

- [x] MinMax 算法执行。
- [x] MinMax 算法的可视化对局。
- [x] MCTS 的训练（但尚不知道能否获得可以使用的 AI，但目测 Loss 确实是在下降。
- [x] MCTS 和 MinMax 的对比函数。
- [x] MCTS 加载预训结束的模型。
- [x] MCTS 算法的可视化对局。
- [ ] 训练 MCTS。
- [ ] 禁手（待定）。

## 文件

```
play.py - 运行下棋的 GUI 界面。
compare.py - 不同 AI 互相对局。
train_mcts.py - 训练 MCTS 方法。
```
