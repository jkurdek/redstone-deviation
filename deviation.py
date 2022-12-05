import os
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

import pandas as pd
from pymongo import MongoClient
from pymongo.collection import Collection

import matplotlib.pyplot as plt

plt.style.use("ggplot")

CONNECTION_STRING = os.getenv("CONNECTION_STRING")
MAX_PROCESSES = 4
DATA_DIR = "data"
PLOTS_DIR = "plots"
DB_NAME = "redstone"
COLLECTION_NAME = "prices"
PROVIDER = "TEHhCDWy-vGmPSZsYJyM0aP_MM4xESgyIZdf5mVODzg"
SYMBOLS = [
    "ETH",
    "USDT",
    "PNG",
    "AVAX",
    "XAVA",
    "LINK",
    "BTC",
    "FRAX",
    "YAK",
    "QI",
    "USDC",
    "YYAV3SA1",
    "sAVAX",
    "SAV2",
    "TJ_AVAX_USDC_LP",
    "PNG_AVAX_USDC_LP",
    "YY_TJ_AVAX_USDC_LP",
    "MOO_TJ_AVAX_USDC_LP",
]
INTERVALS = ["5min", "10min", "30min", "1H", "1D"]


def get_db() -> Collection:
    client = MongoClient(CONNECTION_STRING)
    return client[DB_NAME]


def get_values_to_csv(symbol: str) -> str:
    db = get_db()
    collection = db[COLLECTION_NAME]
    cursor = collection.find(
        {"symbol": symbol, "provider": PROVIDER}, {"timestamp": 1, "_id": False, "value": 1}
    ).batch_size(20000)

    csv_name = f"{DATA_DIR}/{symbol}.csv"
    print(f"Fetching prices for: {symbol}")
    with open(csv_name, "w") as f:
        f.write("timestamp,value\n")
        for doc in cursor:
            f.write(f"{doc['timestamp']},{doc['value']}\n")

    return csv_name


def get_deviation(val1: float, val2: float) -> float:
    return 100 * abs(val1 - val2) / min(val2, val1)


def plot_prices_with_max_deviation(symbol: str, values: pd.DataFrame, max_deviations: list) -> None:
    plt.figure(figsize=(40, 10))
    plt.title(f"{symbol} Value")
    plt.plot(values.index, values["value"], label="Value", color="blue")
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Value", fontsize=14)

    colors = ["red", "green", "orange", "purple", "brown", "pink"]
    line_styles = [(0, (1, 10)), (0, (5, 10)), (0, (3, 10, 1, 10)), (0, (3, 10, 1, 10, 1, 10)), "-"]
    for interval, max_deviation, color, style in zip(INTERVALS, max_deviations, colors, line_styles):
        plt.axvline(
            x=max_deviation[0],
            label=f"Max Deviation ({interval}): {max_deviation[1]:0.2f}%",
            color=color,
            linestyle=style,
            alpha=0.5,
        )

    plt.legend()
    plt.savefig(f"{PLOTS_DIR}/{symbol}.png")


def calculate_deviations(csv_name: str) -> None:
    df = pd.read_csv(csv_name)

    time_range = df["timestamp"].max() - df["timestamp"].min()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", origin="unix")
    df = df.set_index("timestamp")
    symbol = csv_name.split("/")[-1].split(".")[0]

    message = "\n\n"

    message += f"Symbol: {symbol}\n"
    message += f"Data Points: {len(df)}\n"
    message += f"Date Range: {df.index[0]} -- {df.index[-1]}\n"
    message += f"Average Interval: {(time_range / len(df))/1000:0.2f}s\n"

    intervals_deviations = []
    for interval in INTERVALS:
        deviations = df.rolling(interval).apply(lambda x: get_deviation(x[0], x[-1]))
        intervals_deviations.append([deviations.idxmax()[0], deviations.max()[0]])
        message += f"Interval: {interval}, Max Deviation: {deviations.max()[0]:0.2f}%, Max Deviation Interval End: {deviations.idxmax()[0]}\n"

    print(message)
    plot_prices_with_max_deviation(symbol, df, intervals_deviations)


def main():
    pool = ThreadPool(len(SYMBOLS))

    files = pool.map(get_values_to_csv, SYMBOLS)
    pool.close()
    pool.join()

    pool_mp = Pool(MAX_PROCESSES)
    pool_mp.map(calculate_deviations, files)


if __name__ == "__main__":
    main()
