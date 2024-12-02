# hyperparameters
import gymnasium as gym
import stable_baselines3
from gymnasium.wrappers import FlattenObservation
from pynput.keyboard import Listener
from skimage.feature import learn_gmm
from stable_baselines3 import DDPG, A2C, SAC, DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.sac import MlpPolicy, CnnPolicy
from tqdm import tqdm

from ZumaInterface.ZumaAgent import ZumaAgent
from ZumaInterface.envs.ZumaEnv import ZumaEnv
from stable_baselines3.common.env_checker import check_env
import time


learning_rate = 0.01
n_episodes = 100
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

gym.envs.register(
    id="ZumaInterface/ZumaEnv-v0",
    entry_point="ZumaInterface.envs.ZumaEnv:ZumaEnv",
)

# env = gym.make("ZumaInterface/ZumaEnv-v0")



# agent = ZumaAgent(
#     env=env,
#     learning_rate=learning_rate,
#     initial_epsilon=start_epsilon,
#     epsilon_decay=epsilon_decay,
#     final_epsilon=final_epsilon,
# )


canceled = False


def on_press(key):
    try:
        if key.char == "q":
            global canceled
            canceled = True
    except AttributeError:
        pass


if __name__ == "__main__":
    env_vec = make_vec_env("ZumaInterface/ZumaEnv-v0")
    env_stacked = VecFrameStack(env_vec, 4)
    model = PPO("CnnPolicy", env_stacked, verbose=1, policy_kwargs=dict(normalize_images=True))
    model.learn(total_timesteps=1000, log_interval=10)


    # NOTE: Env test
    # env = ZumaEnv()
    # obs, info = env.reset()
    # n_steps = 10
    # for _ in range(n_steps):
    #     # Random action
    #     action = env.action_space.sample()
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     if terminated:
    #         obs, info = env.reset()




