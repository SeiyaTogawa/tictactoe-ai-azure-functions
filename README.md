# 〇×ゲーム強化学習エージェント by Copilot Studio

このプロジェクトは、強化学習で訓練された AI を使用した○×ゲームの Azure Functions API です。

## 概要

- **ユーザー**: ○（先手）
- **AI**: ×（後手）
- **AI の戦略**: 
  1. 勝てる手があれば勝つ
  2. ユーザーの勝ちを阻止（ブロック）
  3. Q値が最大の手を選択（強化学習）

## プロジェクト構造

```
ttt_api/
├── src/
│   ├── ttt-api/
│   │   ├── function_app.py      # Azure Functions のエントリーポイント
│   │   ├── inference.py         # 〇×ゲームの AI ロジック
│   │   └── train_result/        # 訓練結果の保存先
│   │       └── mc_tictactoe.pkl # 訓練済みの Q-table
│   └── training/                # 訓練スクリプト
│       ├── rl_tictactoe.py      # 強化学習のコア実装
│       └── train_and_save.py    # 訓練とモデル保存スクリプト
├── models/                      # 事前訓練済みモデル
│   └── mc_tictactoe.pkl        # 本番用モデル
├── data/                        # 訓練データ
│   └── game_logs/              # ゲームログ
├── notebooks/                   # Jupyter ノートブック
└── scripts/                     # ユーティリティスクリプト
```

## API エンドポイント

### POST `/api/playturn`

プレイヤーの手を受け取り、AI が次の手を返します。

**リクエスト:**
```json
{
  "board": [0, 0, 0, 0, 0, 0, 0, 0, 0],
  "move": 4
}
```

**レスポンス:**
```json
{
  "board_list": [0, 0, 0, 0, 1, 0, 0, 0, -1],
  "board_str": "   \n ○ \n  ×",
  "message": ""
}
```

## 盤面の表現

- `0`: 空のマス
- `1`: ○（プレイヤー）
- `-1`: ×（AI）

盤面は 9 要素の配列で、以下のインデックスに対応：
```
0 | 1 | 2
---------
3 | 4 | 5
---------
6 | 7 | 8
```

## 開発環境のセットアップ

### 前提条件
- Python 3.8+
- Azure Functions Core Tools
- Azure CLI

### インストール

1. 依存関係をインストール:
```bash
pip install -r requirements.txt
```

2. ローカルで実行:
```bash
func start
```

## デプロイ

Azure Functions にデプロイするには：

```bash
func azure functionapp publish <your-function-app-name>
```

## 使用例

```bash
curl -X POST https://your-function-app.azurewebsites.net/api/playturn \
  -H "Content-Type: application/json" \
  -d '{
    "board": [0, 0, 0, 0, 0, 0, 0, 0, 0],
    "move": 4
  }'
```

## 技術詳細

### 強化学習
- モンテカルロ法で訓練された Q-table を使用
- 訓練済みモデルは [`mc_tictactoe.pkl`](models/mc_tictactoe.pkl) に保存

### AI の戦略
1. **勝利手**: AI が勝てる手があれば優先
2. **ブロック手**: プレイヤーの勝ちを阻止
3. **Q値最大**: 強化学習で学習した最適な手を選択

## ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。