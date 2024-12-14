# hyperparameters
import tqdm
import rich
import gymnasium as gym
from stable_baselines3 import DDPG, A2C, SAC, DQN, PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, StackedObservations, VecMonitor, DummyVecEnv
from stable_baselines3.common.callbacks import ProgressBarCallback

# from stable_baselines3.sac import CnnPolicy
from stable_baselines3.ppo import CnnPolicy

from ZumaInterface.envs.ZumaEnv import ZumaEnv

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

if __name__ == "__main__":
    env_vec = make_vec_env("ZumaInterface/ZumaEnv-v0",
                           seed=80085)
    # env = ResizeObservation(env, (100, 100))
    env_stacked = VecFrameStack(env_vec, 4)
    print(env_stacked.observation_space)
    # check_env(env)
    checkpoint_callback = CheckpointCallback(save_freq=10_000,
                                             save_path='./model_checkpoints/')

    # model = SAC(CnnPolicy,
    #             env_stacked,
    #             learning_rate=3e-4,
    #             learning_starts=10,
    #             train_freq=1,
    #             gamma=0.99,
    #             buffer_size=500_000,
    #             batch_size=256,
    #             verbose=1,
    #             tensorboard_log="./tensorboard_logs/",
    #             device="cpu"
    #             )

    model = PPO(CnnPolicy,
                env_stacked,
                n_steps=128,
                n_epochs=4,
                batch_size=128,
                learning_rate=0.00025,
                clip_range=0.1,
                vf_coef=0.5,
                ent_coef=0.01,
                verbose=1,
                tensorboard_log="./tensorboard_logs/",
                device="cpu",

                )

    model.learn(total_timesteps=1_000_000,
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




