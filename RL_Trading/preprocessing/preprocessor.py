import numpy as np
import pandas as pd
import tushare as ts
import talib as ta
import matplotlib.pyplot as plt


def read_data_from_tushare(stock_code, start, end):
    pro = ts.pro_api('a092eed8972e0e19ce46d82f1f743f943f0afd601465e6a904a92ca0')
    # df = pro.daily(ts_code='000001.SZ', start_date='20180701', end_date='20190718')

    df = ts.pro_bar(ts_code=stock_code, start_date=start, end_date=end)
    sorted_df = df[::-1]
    r_index = pd.RangeIndex(0, len(df))
    new_df = sorted_df.set_index(r_index)
    return new_df


def read_csv(path=None):
    df = pd.read_csv('../data/AAPL.csv', encoding='utf-8')
    df = df.sort_values('Date')
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    r_index = pd.RangeIndex(0, len(df))
    new_df = df.set_index(r_index)
    return new_df


def data_split(df, ratio=0.3):
    # split the dataset into training or testing using date
    num_of_tick = len(df)
    split_point = int(num_of_tick * 0.7)
    train_data = df[:split_point]
    test_data = df[split_point:]
    r_index = pd.RangeIndex(0, len(test_data))
    test_data = test_data.set_index(r_index)
    return train_data, test_data


def add_technical_indicator(df):
    """df : DataFrame"""
    print(df.head())
    dif, dea, _ = ta.MACD(df.close, fastperiod=12, slowperiod=26, signalperiod=9)
    rsi = ta.RSI(df.close, timeperiod=14)
    cci = ta.CCI(df.high, df.low, df.close, timeperiod=7)
    adx = ta.ADX(df.high, df.low, df.close, timeperiod=7)
    df['dif'] = dif
    df['dea'] = dea
    df['rsi'] = rsi
    df['cci'] = cci
    df['adx'] = adx
    df.fillna(value=0, inplace=True)
    return df


def add_turbulence(df):
    # 后续完善
    df['turbulence'] = 0
    return df


if __name__ == '__main__':
    pass

