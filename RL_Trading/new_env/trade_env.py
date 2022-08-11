import gym
import pandas as pd
from gym import spaces
from gym.utils import seeding
import numpy as np
import random
import matplotlib.pyplot as plt
from preprocessing.preprocessor import read_csv

# shares normalization factor
# 100 shares per trade
HMAX_NORMALIZE = 100
# initial amount of money we have in our account
INITIAL_ACCOUNT_BALANCE = 1000000
# total number of stocks in our portfolio
STOCK_DIM = 1
# transaction fee: 1/1000 reasonable percentage
TRANSACTION_FEE_PERCENT = 0.001
REWARD_SCALING = 1e-4


class StockTradeEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    counter = 0

    def __init__(self, df, window_size, model_name):
        self.df = df
        self.model = model_name
        self.window_size = window_size
        self.features = self._process_data()
        print(self.features.shape)
        self.shape = (window_size, self.features.shape[1])
        print(self.shape)
        # action_space
        self.action_space = spaces.Box(low=-1, high=1, shape=(STOCK_DIM,))
        # observation_space
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=self.shape, dtype=np.float16)
        # account
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.df) - 1
        self._done = None
        self.turbulence_threshold = 140
        self.turbulence = 0
        self._current_tick = None
        self._current_trade_price = 0
        self.state = None
        # initialize reward
        self.reward = 0
        self.cost = 0
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        self.trades = 0
        # trade history
        self._position = None
        self._position_history = None
        self.history = None

    def _get_observation(self):
        return self.features[(self._current_tick - self.window_size):self._current_tick]

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def _sell_stock(self, action):
        if self.turbulence < self.turbulence_threshold:
            if self.shares_held > 0:
                self._position = action
                self._position_history.append(self._position)
                # update balance
                self.balance += self._current_trade_price * min(abs(action), self.shares_held) * \
                                (1 - TRANSACTION_FEE_PERCENT)
                self.shares_held -= min(abs(action), self.shares_held)
                self.cost = self._current_trade_price * min(abs(action), self.shares_held) * TRANSACTION_FEE_PERCENT
                self.trades += 1

            else:
                pass
        else:
            # clear out all position
            pass

    def _buy_stock(self, action):
        if self.turbulence < self.turbulence_threshold:
            self._position = action
            self._position_history.append(self._position)
            available_amount = self.balance // self._current_trade_price

            # update balance
            self.balance -= (self._current_trade_price * min(available_amount, action)) * (1 - TRANSACTION_FEE_PERCENT)
            self.shares_held += min(available_amount, action)

            self.cost += self._current_trade_price * min(available_amount, action) * TRANSACTION_FEE_PERCENT
            self.trades += 1
        else:
            pass

    def step(self, actions):
        self._done = self.net_worth <= 0 or self._current_tick == self._end_tick

        if self._done:
            print("running..")
            StockTradeEnv.counter += 1
            plt.plot(self.asset_memory, 'r')
            plt.savefig('..\\results\\trade_res\\{}_account_value.png'.format(self.model))
            plt.close()

            # print("end_total_asset:{}".format(end_total_asset))
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv('..\\results\\trade_res\\{}_account_value.csv'.format(self.model))
            # print("total_cost: ", self.cost)
            # print("total_trades: ", self.trades)
            df_total_value.columns = ['account_value']
            df_total_value['daily_return'] = df_total_value.pct_change(1)
            sharpe = (252 ** 0.5) * df_total_value['daily_return'].mean() / \
                     df_total_value['daily_return'].std()
            # print("Sharpe: ",sharpe)
            # print("=================================")
            df_rewards = pd.DataFrame(self.rewards_memory)
            # df_rewards.to_csv('results\\account')

            return self.state, self.reward, self._done, {}
        else:
            # Set the current price to a random price within the time step
            # price of held
            current_price = random.uniform(
                self.df.loc[self._current_tick, "open"], self.df.loc[self._current_tick, "close"])
            self._current_trade_price = current_price

            real_action = int(actions * HMAX_NORMALIZE)
            begin_total_asset = self.balance + current_price * self.shares_held

            if real_action > 0:
                self._buy_stock(real_action)
            else:
                self._sell_stock(real_action)
            end_total_asset = self.balance + self.features.loc[self._current_tick, "close"] * self.shares_held
            # next tick data
            self._current_tick += 1
            self.turbulence = 0
            obs = self._get_observation()
            self.state = obs

            self.asset_memory.append(end_total_asset)
            # print("end_total_asset:{}".format(end_total_asset))
            self.reward = end_total_asset - begin_total_asset
            # print("step_reward:{}".format(self.reward))
            self.rewards_memory.append(self.reward)
            self.reward = self.reward * REWARD_SCALING
            info = dict(
                position=self._position,
                day_profit=self.reward
            )
            self._update_history(info)
            return obs, self.reward, self._done, info

    def reset(self):
        print("i am running")
        self._done = False
        self._current_tick = self._start_tick
        self._current_trade_price = 0
        # self._last_trade_tick = self._current_tick - 1
        self.cost = 0
        self.trades = 0
        self.rewards_memory = []
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.shares_held = 0
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.turbulence = 0
        self._position = 0
        self._position_history = (self.window_size * [None]) + [self._position]
        self.history = {}
        self.state = self._get_observation()
        return self._get_observation()

    def _process_data(self, ta=False):
        # simple version
        if ta:
            cols = ['open', 'high', 'low', 'close', 'dif', 'dea', 'rsi', 'cci', 'adx']
            features = self.df[cols]
            return features
        else:
            print(self.df)
            # columns = ['Open', 'High', 'Low', 'Close']
            cols = ['open', 'high', 'low', 'close']
            features = self.df[cols]
            return features

    def render(self, mode='human'):
        print("trade nums ", self.trades)
        print(self.history)
        print(len(self._position_history), self._position_history)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
