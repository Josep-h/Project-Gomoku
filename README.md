# Project-Gomoku

A project of computational game theory.

## 代码环境

- Python 3.8
- PyTorch 1.10.0
- CUDA 11.3

## 实现

计划算法分为个：

- MinMax 博弈树方法（可以运行）
- DeepCFR 方法
- 蒙特卡洛搜索树方法

参考代码在 `/AlphaZero_Gomoku` 中。

## 进展

[x] MinMax 算法执行。
[x] MinMax 算法的可视化。
[x] MCTS 的训练（但尚不知道能否获得可以使用的AI，但目测Loss确实是在下降。
[x] MCTS 和 MinMax 的对比函数。
[] 给出加入对比的 API。
[] 禁手（待定）。

## 文件

```
play.py - 运行下棋的GUI界面。
```