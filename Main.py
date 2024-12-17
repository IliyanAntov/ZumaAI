# hyperparameters
import time

import tqdm
import rich
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import DDPG, A2C, SAC, DQN, PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, StackedObservations, VecMonitor, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import ProgressBarCallback

from stable_baselines3.sac import CnnPolicy
# from stable_baselines3.ppo import CnnPolicy


from ZumaInterface.envs.ZumaEnv import ZumaEnv
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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


def make_env(env_name, env_index):
    def _make():
        env = gym.make(env_name,
                       env_index=env_index,
                       max_episode_steps=500)
        # env = ZumaEnv(env_index=env_index)
        # env = TimeLimit(env, 500)
        return env
    return _make


if __name__ == "__main__":
    env_name = "ZumaInterface/ZumaEnv-v0"
    instances = 10
    # env_vec = SubprocVecEnv([lambda: make_env(env_name, 0), lambda: make_env(env_name, 1)])
    envs = [make_env(env_name, i) for i in range(instances)]

    env_vec = SubprocVecEnv(envs)
    # env_vec = DummyVecEnv(envs)
    # env_vec = AsyncVectorEnv(envs)

    # env_vec = make_vec_env("ZumaInterface/ZumaEnv-v0",
    #                        seed=80085,
    #                        n_envs=2)

    env_monitor = VecMonitor(env_vec)
    env_stacked = VecFrameStack(env_monitor, 2)
    # env = ResizeObservation(env, (100, 100))
    print(env_stacked.observation_space)
    # check_env(env)
    checkpoint_callback = CheckpointCallback(save_freq=10_000,
                                             save_path='./model_checkpoints/')

    model = SAC(CnnPolicy,
                env_stacked,
                learning_rate=3e-4,
                learning_starts=1,
                buffer_size=500_000,
                batch_size=10_000,
                gamma=0.99,
                tau=0.005,
                train_freq=1,
                gradient_steps=1,
                ent_coef="auto",
                verbose=1,
                tensorboard_log="./tensorboard_logs/",
                device="cuda"
                )

    # model = PPO(CnnPolicy,
    #             env_stacked,
    #             n_steps=128,
    #             n_epochs=4,
    #             batch_size=128,
    #             learning_rate=0.00025,
    #             clip_range=0.1,
    #             vf_coef=0.5,
    #             ent_coef=0.01,
    #             verbose=1,
    #             # tensorboard_log="./tensorboard_logs/",
    #             # device="cpu",
    #
    #             )

    model.learn(total_timesteps=3_000_000,
                log_interval=1,
                callback=[checkpoint_callback],
                )

    model.save("./models/model")

    # NOTE: Load trained model
    # model = SAC.load("./models/model")
    # obs = env.reset()
    # for i in range(1000):
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)


    # NOTE: Env test
    # env = ZumaEnv()
    # obs, info = env.reset()
    # n_steps = 10
    # for i in range(n_steps):
    #     # Random action
    #     action = env.action_space.sample()
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     img = env.render()
    #     img.save(str(i) + ".png")
    #     # print()
    #     if terminated:
    #         obs, info = env.reset()




