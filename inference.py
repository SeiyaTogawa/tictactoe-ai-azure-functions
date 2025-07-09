# inference.py  –  Stateless Tic-Tac-Toe (AI = 後手 ×, ブロック優先)

from __future__ import annotations
from typing import List, Dict
from collections import defaultdict
from enum import Enum
import pickle, random, os, os.path as _p

# ---------------------------------------------------------------------------
# 型・定数
# ---------------------------------------------------------------------------
Board = List[int]                                            # 9 要素, {-1,0,1}
PKL_PATH = os.getenv(
    "TTT_PKL",
    _p.join(_p.dirname(__file__), "train_result/mc_tictactoe.pkl")
)

class Cell(Enum):
    EMPTY = 0
    O = 1        # ユーザー (先手)
    X = -1       # エージェント (後手)

    def __str__(self) -> str:
        return {0: " ", 1: "O", -1: "X"}[self.value]

_WIN_PATTERNS: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
)

# ---------------------------------------------------------------------------
# 純関数ヘルパ
# ---------------------------------------------------------------------------
def _winner(b: Board) -> Cell | None:
    for a, c, d in _WIN_PATTERNS:
        if b[a] != 0 and b[a] == b[c] == b[d]:
            return Cell(b[a])
    return None

def _legal(b: Board) -> List[int]:
    return [i for i, v in enumerate(b) if v == 0]

def _bstr(b: Board) -> str:
    return "".join(str(Cell(v)) for v in b)

def _pretty(b: Board) -> str:
    sym = {0: " ", 1: "○", -1: "×"}
    return "\n".join("".join(sym[b[i + j]] for j in range(3)) for i in (0, 3, 6))

def _load_q(path=PKL_PATH) -> Dict[str, List[float]]:
    try:
        with open(path, "rb") as f:
            q = pickle.load(f)["Q"]
        return defaultdict(lambda: [0.0]*9, q)
    except FileNotFoundError:
        return defaultdict(lambda: [0.0]*9)

_Q = _load_q()                                              # キャッシュ

# ---------------------------------------------------------------------------
# 盤面評価付きプレイ
# ---------------------------------------------------------------------------
def play_turn(board: Board, my_move: int) -> Board:
    """ユーザー(◯) が `my_move` に置いたあと AI(×) が応答する."""
    if board[my_move] != 0:
        raise ValueError(f"position {my_move} is not empty")

    board = board.copy()
    board[my_move] = Cell.O.value

    if _winner(board) or 0 not in board:        # ユーザーが勝ち/引分
        return board

    legal = _legal(board)

    # --- 優先 1: AI の勝ち手 ---
    for m in legal:
        tmp = board.copy(); tmp[m] = Cell.X.value
        if _winner(tmp) == Cell.X:
            board[m] = Cell.X.value
            return board

    # --- 優先 2: ブロック手 (◯の勝ち阻止) ---
    for m in legal:
        tmp = board.copy(); tmp[m] = Cell.O.value
        if _winner(tmp) == Cell.O:
            board[m] = Cell.X.value
            return board

    # --- 優先 3: Q 最大 (同値は乱択) ---
    key = _bstr(board)
    q = _Q[key]
    best_q = max(q[m] for m in legal)
    best_moves = [m for m in legal if q[m] == best_q]
    board[random.choice(best_moves)] = Cell.X.value
    return board

# ---------------------------------------------------------------------------
# JSON 向けラッパ
# ---------------------------------------------------------------------------
def play_turn_ex(board: Board, my_move: int) -> dict:
    new_board = play_turn(board, my_move)

    win = _winner(new_board)
    if win == Cell.O:
        msg = "あなたの勝ちです!おめでとう！もう一回だ！"
    elif win == Cell.X:
        msg = "あなたの負けです。残念！もう一回！"
    elif 0 not in new_board:
        msg = "引き分けです。惜しい！もう一回！"
    else:
        msg = ""

    return {
        "board_list": new_board,
        "board_str" : _pretty(new_board),
        "message"   : msg,
    }
