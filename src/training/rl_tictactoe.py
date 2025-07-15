# -*- coding: utf-8 -*-
"""rl_tictactoe.py – 後手（×）専用 Monte‑Carlo 強化学習エージェント

*   エージェントは常に **後手 ×**
*   先手（◯）はランダム手　※必要なら self‑play 等に差し替え可
*   ε‑greedy を線形減衰 (1.0 → 0.01) で高速収束
*   step_penalty = ‑0.04 で「粘って勝つ」行動を許容
*   終局報酬は **自分視点** (+1=勝ち, ‑1=負け, 0=引分/中間) に変換
*   main として実行すると 2,000,000 エピソード学習→Q を保存→勝率評価
"""

from __future__ import annotations
from typing import List, Dict, Tuple
from enum import Enum
from collections import defaultdict
import copy
import random
import pickle
import os

################################################################################
# 型定義 / 列挙体
################################################################################
Board = List[int]  # flatten 3×3 board (len==9)

class Cell(Enum):
    EMPTY = 0
    O     = 1   # 先手（環境側）
    X     = -1  # 後手（エージェント）

    def __str__(self) -> str:
        return {self.EMPTY: " ", self.O: "O", self.X: "X"}[self]

################################################################################
# 環境
################################################################################
class TicTacToeEnv:
    WIN_REWARD  = 1.0
    DRAW_REWARD = 0.0

    _WIN_PATTERNS: Tuple[Tuple[int, int, int], ...] = (
        (0, 1, 2), (3, 4, 5), (6, 7, 8),      # rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),      # cols
        (0, 4, 8), (2, 4, 6),                 # diagonals
    )

    def __init__(self, *, step_penalty: float = -0.04):
        self.step_penalty = step_penalty
        self.board: Board = [Cell.EMPTY.value] * 9

    def reset(self) -> None:
        self.board = [Cell.EMPTY.value] * 9

    #----------------------------------------------------------------------
    # 手を進める
    #----------------------------------------------------------------------
    def step(self, action: int):
        if self.board[action] != Cell.EMPTY.value:
            raise ValueError(f"Illegal move: {action} not empty")
        self.board[action] = self._current_player().value
        reward, done = self._reward_done()
        return copy.copy(self.board), reward, done

    #----------------------------------------------------------------------
    # 内部ヘルパー
    #----------------------------------------------------------------------
    def _current_player(self) -> Cell:
        # O と X の数が等しいとき O の手番
        return Cell.O if self.board.count(Cell.O.value) == self.board.count(Cell.X.value) else Cell.X

    def _reward_done(self):
        winner = self._winner()
        if winner is not None:
            return winner.value * self.WIN_REWARD, True
        if Cell.EMPTY.value not in self.board:
            return self.DRAW_REWARD, True
        return self.step_penalty, False

    def _winner(self):
        for a, b, c in self._WIN_PATTERNS:
            if self.board[a] != Cell.EMPTY.value and self.board[a] == self.board[b] == self.board[c]:
                return Cell(self.board[a])
        return None

    #----------------------------------------------------------------------
    # デバッグ表示
    #----------------------------------------------------------------------
    def render(self):
        sym = {Cell.O.value: "○", Cell.X.value: "×", Cell.EMPTY.value: " "}
        for i in range(0, 9, 3):
            print("".join(sym[v] for v in self.board[i:i+3]))

################################################################################
# モンテカルロエージェント (後手 ×)
################################################################################
class MonteCarloAgent:
    def __init__(self, env: TicTacToeEnv, *, epsilon_start=1.0, epsilon_end=0.01,
                 min_alpha=0.01, gamma=0.9, learning=True):
        self.env = env
        self.eps_start = epsilon_start
        self.eps_end   = epsilon_end
        self.min_alpha = min_alpha
        self.gamma     = gamma
        self.learning  = learning
        self.epsilon   = epsilon_start   # 更新用

        self.Q: Dict[str, List[float]] = defaultdict(lambda: [0.0]*9)
        self.N: Dict[str, List[int]]   = defaultdict(lambda: [0]*9)

    #-------------------------------
    # 基本ヘルパー
    #-------------------------------
    def _legal(self):
        return [i for i,v in enumerate(self.env.board) if v == Cell.EMPTY.value]

    def _board_key(self):
        return "".join(str(Cell(v)) for v in self.env.board)

    def _choose_action(self):
        legal = self._legal()
        if self.learning and random.random() < self.epsilon:
            return random.choice(legal)
        q = self.Q[self._board_key()]
        best = max(q[m] for m in legal)
        return random.choice([m for m in legal if q[m]==best])

    #-------------------------------
    # 1 エピソード
    #-------------------------------
    def play_episode(self, ep_idx=0, total_eps=1):
        if self.learning:
            ratio = ep_idx/total_eps
            self.epsilon = max(self.eps_end, self.eps_start*(1-ratio))

        self.env.reset()
        episode = []
        done = False

        # ❶ 先手 ◯ (ランダム)
        opp = random.choice(self._legal())
        self.env.step(opp)

        while not done:
            # ❷ エージェント ×
            state = self._board_key()
            act   = self._choose_action()
            _, env_r, done = self.env.step(act)

            # 自分視点の報酬 (+1=勝,-1=負,中間そのまま)
            reward = env_r
            if done and env_r != self.env.DRAW_REWARD:
                reward = -env_r
            episode.append(dict(state=state, action=act, reward=reward))

            if done:
                break

            # ❸ 相手 ◯ 手 (ランダム)
            opp = random.choice(self._legal())
            _, _, done = self.env.step(opp)

        if self.learning:
            self._update_Q(episode)
        return [s["reward"] for s in episode]

    #-------------------------------
    # Monte‑Carlo 価値更新
    #-------------------------------
    def _update_Q(self, episode):
        for i, step in enumerate(episode):
            s, a = step["state"], step["action"]
            G, disc = 0.0, 1.0
            for j in range(i, len(episode)):
                G += disc * episode[j]["reward"]
                disc *= self.gamma
            self.N[s][a] += 1
            alpha = max(1/self.N[s][a], self.min_alpha)
            self.Q[s][a] += alpha*(G - self.Q[s][a])

    #-------------------------------
    # 保存 / 読込
    #-------------------------------
    def save(self, path: str, *, save_N=True):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        obj = {"Q": dict(self.Q)}
        if save_N:
            obj["N"] = dict(self.N)
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.Q = defaultdict(lambda:[0.0]*9, obj["Q"])
        self.N = defaultdict(lambda:[0]*9,    obj.get("N", {}))

################################################################################
# 学習 & 評価 (スクリプト実行時)
################################################################################
if __name__ == "__main__":
    env   = TicTacToeEnv(step_penalty=-0.04)
    agent = MonteCarloAgent(env)

    EPISODES = 2_000_000
    REPORT   = 100_000

    for ep in range(EPISODES):
        agent.play_episode(ep, EPISODES)
        if (ep+1) % REPORT == 0:
            print(f"{ep+1:,} episodes done")

    # 評価
    agent.learning = False
    agent.epsilon  = 0.0
    wins = sum(agent.play_episode()[-1] > 0 for _ in range(10_000))
    print(f"Win rate (×後手) : {wins/100:.2f}%")

    agent.save("train_result/mc_tictactoe.pkl")
    print("✅ Q-table saved to train_result/mc_tictactoe.pkl")
