from rl_tictactoe import TicTacToeEnv, MonteCarloAgent
import time, os

def main():
    env   = TicTacToeEnv(step_penalty=-0.04)
    agent = MonteCarloAgent(env, epsilon=1.0)  # ε は run 中に上書き

    EPISODES = 300_000
    for ep in range(EPISODES):
        agent.play_episode(ep, EPISODES)

    # ---------- 評価 ----------
    agent.learning = False
    agent.epsilon  = 0.0
    wins = sum(agent.play_episode()[-1] >= 0 for _ in range(10_000))
    print(f"Win rate (×後手) : {wins/100:.2f}%")

    # ---------- 保存 ----------
    path = "train_result/mc_tictactoe.pkl"
    os.makedirs("train_result", exist_ok=True)
    agent.save(path)
    print(f"✅ Saved to {path}")

if __name__ == "__main__":
    main()
