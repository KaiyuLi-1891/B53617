from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import torch


class PPO():
    def __init__(self, env=Env2DAirfoil(), file_name="ppo_Airfoil"):
        self.env = env
        check_env(self.env)
        self.model = PPO("MlpPolicy", self.env, learning_rate=0.0005, n_steps=2000, verbose=1)
        self.file_name = file_name

    def train(self, total_timesteps=100000, epoch=1):
        for i in range(epoch):
            self.model.learn(total_timesteps=total_timesteps)
            self.env.reset()

    def show_net(self):
        policy_net = self.model.policy
        print(policy_net)

    def save_model(self):
        self.model.save(self.file_name)

    def load_model(self):
        del self.model
        self.model = PPO.load(self.file_name)

    def evaluate_model(self):
        mean_reward, std_reward = evaluate_policy(self.model, self.env, n_eval_episodes=5)
        print("Mean Reward:", mean_reward)
        pritn("Standard Deviation of Reward:", std_reward)
