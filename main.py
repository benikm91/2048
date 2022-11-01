from game2048.game_logic import Game
from game2048.r_learning import Q_agent

AGENT_FILE_PATH = "game2048/best_agent.npy"

agent = Q_agent.load_agent(AGENT_FILE_PATH)
est = agent.evaluate
results = Game.trial(estimator=est, num=100)
print(results)