import pandas as pd
import numpy as np
import time

# RL models from sb3
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from preprocessing.preprocessor import *
from new_env.train_env import TradingEnv
from new_env.trade_env import StockTradeEnv


def train_A2C(env_train, timesteps=25000):
    """a2c"""

    start = time.time()
    model = A2C('MlpPolicy', env_train, verbose=1)
    model.learn(total_timesteps=timesteps)
    end = time.time()
    print('Training time (A2C): ', (end - start) / 60, ' minutes')
    return model


def train_DDPG(env_train, timesteps=50):
    """DDPG model"""
    # add the noise objects for DDPG
    n_actions = env_train.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    start = time.time()
    model = DDPG('MlpPolicy', env_train, action_noise=action_noise)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    print('Training time (DDPG): ', (end - start) / 60, ' minutes')
    return model



def train_SAC(env_train, timesteps=10000):
    start = time.time()
    model = SAC('MlpPolicy', env_train, verbose=1)
    model.learn(total_timesteps=timesteps, log_interval=4)
    end = time.time()
    print('Training time (SAC): ', (end - start) / 60, ' minutes')
    return model


def train_PPO(env_train, timesteps=10000):
    """PPO model"""

    start = time.time()
    model = PPO('MlpPolicy', env_train, ent_coef=0.005)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    print('Training time (PPO): ', (end - start) / 60, ' minutes')
    return model


def DRL_prediction(t_df, model, model_name):
    # make a prediction based on trained model
    # trading env
    trade_data = t_df
    env_trade = DummyVecEnv([lambda: StockTradeEnv(trade_data, 5, model_name)])
    obs_trade = env_trade.reset()
    print(obs_trade)
    # window_size + 1 最大交易次数
    for i in range(len(trade_data) - 5):
        print("running.....")
        action, _states = model.predict(obs_trade)
        obs_trade, rewards, dones, info = env_trade.step(action)
        env_trade.render()


if __name__ == '__main__':
    # df = read_csv()
    maotai = '600519.SH'
    hengrui = '600276.SH'
    wanhua = '600309.SH'
    zhao = '600036.SH'

    # 时间
    start_time, end_time = '20020101', '20210101'

    df = read_data_from_tushare(maotai, start_time, end_time)
    # new_df = add_technical_indicator(df)
    # df = add_turbulence(df)
    train_df, trade_df = data_split(df)

    ev_train = DummyVecEnv([lambda: TradingEnv(train_df, 5)])
    ddpg_ev_train = DummyVecEnv([lambda: TradingEnv(trade_df, 5)])
    ddpg = train_DDPG(ddpg_ev_train, 10000)
    # a2c = train_A2C(ev_train, 10000)
    # ppo = train_PPO(ev_train, 10000)
    # DRL_prediction(trade_df, ppo, "ppo")
    # DRL_prediction(trade_df, a2c, "a2c")
    # sac = train_SAC(ev_train, 10000)
    # DRL_prediction(trade_df, sac, "sac")

