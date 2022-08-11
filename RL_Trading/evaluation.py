import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def day_profit(data):
    # 计算每天的累计收益
    data_new = data / data.iloc[0]
    print(data_new.head())
    # 区间累计收益率
    total_return = data_new.iloc[-1] - 1  # 用最后一个值减去1
    total_return = pd.DataFrame(total_return.values, columns=["累计收益"], index=total_return.index)
    print(total_return)
    return data_new


def annual_profit(data):
    # 查看数据长度
    data_length = len(data)
    print("数据长度:", data_length)

    # 计算年化收益率
    annual_return = pow(data.iloc[-1], 250 / data_length) - 1

    # 调试输出
    print("年化收益:")
    print(annual_return)


def maxdrawdown(data):
    # 计算累计最大值
    Max = data.cummax()

    # 调试输出
    print(data.head())
    print(Max.head())

    # 计算每天回撤
    drawdown_daily = (data.cummax() - data) / data.cummax()

    # 调试输出
    print(drawdown_daily.head())
    # 计算最大回测
    total_drawdown = drawdown_daily.max()
    # 改成df
    total_drawdown = pd.DataFrame([str(i * 100) + "%" for i in total_drawdown], columns=["最大回撤"],
                                  index=total_drawdown.index)
    # 调试输出
    print(total_drawdown.head())


def sharp_ratio(data):
    # 向后填补缺失值
    data_fill = data.fillna(method='pad')

    # 计算每日收益率
    return_rate = data_fill.apply(lambda x: x / x.shift(1) - 1)[1:]  # 去除首个NaN

    # 调试输出
    print(return_rate.head())

    # 计算超额回报率
    exReturn = return_rate - 0.03 / 250

    # 计算夏普比率
    sharpe_rate = np.sqrt(len(exReturn)) * exReturn.mean() / exReturn.std()

    # 夏普比率的输出结果
    SHR = pd.DataFrame(sharpe_rate, columns=['夏普比率'])

    # 输出伊利和茅台的夏普比率
    print(SHR)


if __name__ == '__main__':
    # read data
    df = pd.read_csv(r"results/trade_res/sac_account_value.csv", index_col=0)
    print(df.head())
    new_df = day_profit(df)
    annual_profit(new_df)
    maxdrawdown(df)
    sharp_ratio(df)
