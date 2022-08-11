import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random
import matplotlib.pyplot as plt
from preprocessing.preprocessor import read_csv

# initial amount of money we have in our account
INITIAL_ACCOUNT_BALANCE = 1000000
# transaction fee: 1/1000 reasonable percentage
TRANSACTION_FEE_PERCENT = 0.001
REWARD_SCALING = 1e-4


class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size):

        assert df.ndim == 2
        self.df = df
        self.window_size = window_size
        self.features = self._process_data()
        self.shape = (window_size, self.features.shape[1])
        # spaces
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)
        self.observation_space = spaces.Box(
            low=np.inf, high=np.inf, shape=self.shape, dtype=np.float16)
        # account
        self.net_worth = None
        self.balance = None
        self.shares_held = 0
        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.df) - 1
        self._done = False
        self._current_tick = None

        # self._last_trade_tick = None
        # current trade price
        self._current_trade_price = None
        # last trade price
        # self._last_trade_price = None
        self._total_reward = None
        self._total_profit = None
        self._total_rendering = None

        self.trade_history = None

    def _get_observation(self):
        return self.features[(self._current_tick - self.window_size):self._current_tick]

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        current_price = random.uniform(
            self.df.loc[self._current_tick, "Open"], self.df.loc[self._current_tick, "Close"])
        self._current_trade_price = current_price
        action_type = action[0]
        amount = action[1]
        if action_type < 1:
            # Buy amount % of balance in shares
            total_possible = int(self.balance / current_price)
            shares_bought = int(total_possible * amount)
            additional_cost = shares_bought * current_price

            self.balance -= additional_cost
            self.shares_held += shares_bought

        elif action_type < 2:
            # Sell amount % of shares held
            shares_sold = int(self.shares_held * amount)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price
        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

    def _calculate_reward(self, action):
        step_reward = 0

        current_price = self._current_trade_price
        close_price = self.features.loc[self._current_tick, "Close"]
        price_diff = close_price - current_price

        step_reward += price_diff
        return step_reward

    def step(self, action):

        if self.net_worth <= 0 or self._current_tick == self._end_tick:
            self._done = True

        self._take_action(action)
        reward = self._calculate_reward(action)
        obs = self._get_observation()
        return obs, reward, self._done, {}

    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        # self._last_trade_tick = self._current_tick - 1
        self._total_reward = 0.
        self._total_profit = 1.  # question
        self.trade_history = {}
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.total_shares_sold = 0
        self.total_sales_value = 0
        return self._get_observation()

    def _process_data(self, ta=False):
        # simple version
        if ta:
            pass
        else:
            print(self.df)
            columns = ['Open', 'High', 'Low', 'Close']
            features = self.df[columns]
            return features

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self._current_tick}')
        print(f'Balance: {self.balance}')
        print(
            f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')

        print(
            f'Net worth: {self.net_worth} ')
        print(f'Profit: {profit}')

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]