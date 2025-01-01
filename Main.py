# hyperparameters
import time

import numpy as np
import tqdm
import rich
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import DDPG, A2C, SAC, DQN, PPO, TD3
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise
from stable_baselines3.common.vec_env import VecFrameStack, StackedObservations, VecMonitor, DummyVecEnv, SubprocVecEnv, \
    VecVideoRecorder
from stable_baselines3.common.callbacks import ProgressBarCallback

# from stable_baselines3.sac import CnnPolicy
# from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.td3 import CnnPolicy


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
                       max_episode_steps=250)
        # check_env(env)
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
    # env_stacked = VecVideoRecorder(env_monitor, "logs/",
    #                        record_video_trigger=lambda x: x == 0, video_length=20)
    # env = ResizeObservation(env, (100, 100))
    print(env_stacked.observation_space)
    # check_env(env)
    checkpoint_callback = CheckpointCallback(save_freq=10_000,
                                             save_path='./model_checkpoints/')

    n_actions = env_monitor.action_space.shape[-1]
    base_noise = NormalActionNoise(mean=np.zeros(1), sigma=0.1 * np.ones(n_actions))
    # vec_noise = VectorizedActionNoise(base_noise, 10)
    # model = SAC.load("./models/model")
    # model.action_noise = vec_noise
    # model.ent_coef = -1.0
    # model.set_env(env_stacked)

    # model = SAC(CnnPolicy,
    #             env_stacked,
    #             learning_rate=3e-4,
    #             learning_starts=1,
    #             buffer_size=30_000,
    #             batch_size=256,
    #             gamma=0.98,
    #             tau=0.005,
    #             train_freq=1,
    #             gradient_steps=1,
    #             ent_coef=1.0,
    #             # target_entropy=1.0,
    #             verbose=1,
    #             tensorboard_log="./tensorboard_logs/",
    #             device="cuda",
    #             action_noise=vec_noise,
    #             policy_kwargs={"use_sde": True},
    #             )

    # model = PPO(CnnPolicy,
    #             env_stacked,
    #             n_steps=128,
    #             n_epochs=4,
    #             batch_size=256,
    #             learning_rate=0.00025,
    #             clip_range=0.1,
    #             vf_coef=0.5,
    #             ent_coef=0.01,
    #             verbose=1,
    #             tensorboard_log="./tensorboard_logs/",
    #             device="cuda",
    #             )

    # model = SAC(CnnPolicy,
    #             env_stacked,
    #             learning_rate=3e-4,
    #             learning_starts=1,
    #             buffer_size=50_000,
    #             batch_size=256,
    #             gamma=0.5,
    #             tau=0.005,
    #             train_freq=1,
    #             gradient_steps=1,
    #             verbose=1,
    #             tensorboard_log="./tensorboard_logs/",
    #             device="cuda",
    #             # action_noise=vec_noise,
    #             # policy_kwargs={"use_sde": True},
    #             )
    model = TD3(CnnPolicy,
                env_stacked,
                buffer_size=100_000,
                gamma=0.95,
                verbose=1,
                tensorboard_log="./tensorboard_logs/",
                device="cuda",
                action_noise=base_noise,
                # optimize_memory_usage=True
                )

    model.learn(total_timesteps=4_000_000,
                log_interval=1,
                callback=[checkpoint_callback],
                # reset_num_timesteps=False,
                )

    model.save("./models/model")

    # NOTE: Load trained model
    # model = SAC.load("./models/model")
    # obs = env.reset()
    # for i in range(1000):
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)


    # NOTE: Env test
    # env = ZumaEnv(0)
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




