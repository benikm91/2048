from game2048.game_logic import Game
from game2048.r_learning import Q_agent, basic_reward

num_eps = 100000

agent = Q_agent(n=4, reward=basic_reward, alpha=0.1, file="new_agent.npy")
Q_agent.train_run(num_eps, agent=agent, file="my_agent.npy", start_ep=0)
