import os
from env import CliffWalkingEnv
from sarsa import sarsa
from q_learning import q_learning
import utils

os.makedirs('output',exist_ok=True)
env = CliffWalkingEnv()
Q_sarsa, ret_sarsa = sarsa(env)
utils.plot_policy(env,Q_sarsa,'SARSA Policy','output/policy_sarsa.png')
Q_q, ret_q = q_learning(env)
utils.plot_policy(env,Q_q,'Q-learning Policy','output/policy_q.png')
utils.plot_returns({'SARSA':ret_sarsa,'Q-learning':ret_q},'output/returns.png')
