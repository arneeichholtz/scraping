import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# MAIN FUNCTION
def simulate_investing(interval, simulations):
    quantity = 1000
    curr_price = df_price.iloc[-1]["price"]
    sell_value = quantity * curr_price

    investments = np.zeros(shape=(simulations, interval))
    batches = np.linspace(0, 252, interval+1)
    # if (252 % interval) != 0 the batches are not integers, values are floor-rounded for low and high

    for i in range(interval):
        q = quantity / interval
        buy_inds = np.random.randint(low=batches[i], high=batches[i+1], size=simulations)
        investments[:, i] = q * df_price.iloc[buy_inds]["price"]

    total_investment = np.sum(investments, axis=1)
    profits = sell_value - total_investment
    avg_profit = np.mean(profits)
    return np.mean(profits) # Average profit over all simulations

if __name__ == "__main__":

    path = "^GSPC-fro=1421764200-to=1692279000.csv"
    df = pd.read_csv(path)
    df.head()

    print(f"Number of datapoints: {len(df)}")
    
    df["price"] = (df["high"] + df["low"]) / 2 # Making df
    df_price = df[["date", "price"]]

    # TEST CASE
    # Profit for investing all at once
    # 20 jan 2015 is datapoint 0, 20 jan 2016 is datapoint 252
    np.random.seed(0)
    quantity = 1000
    buy_indices = np.random.randint(252, size=(10))
    investments = [quantity * df_price.iloc[i]["price"] for i in buy_indices]

    curr_price = df_price.iloc[-1]["price"]
    sell_value = quantity * curr_price
    profits = sell_value - investments
    avg_profit = np.mean(profits)
    print(f"Average profit: ${avg_profit:.0f}")

    # CAllING FUNCTION - intervals and simulations are variable
    intervals = np.arange(1, 11)
    simulations = 100000
    profits = [simulate_investing(inter, simulations) for inter in intervals]

    # PLOTTING
    plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
    plt.plot(intervals, profits)
    plt.title(f"Profit for different investing intervals in the year from 20/01/15\n{simulations//1000}k simulations per interval", fontsize=10)
    plt.xlabel("Number of Intervals")
    plt.ylabel("Average profit")
    plt.show()

    low = min(profits)
    high = max(profits)
    print(f"The difference between the best-earning interval (max) and worst-earning interval (min) is {((high-low)/low)*100:.3f}%")

    # only 0.023% -- not significant

