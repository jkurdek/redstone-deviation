import os
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongo.collection import Collection

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
OUTLIER_WINDOW_SIZE = "5D"
OUTLIER_STD_COUNT = 20.0
OUTLIER_MIN_MEAN_DIFF = 0.02


def get_db() -> Collection:
    client = MongoClient(CONNECTION_STRING)
    return client[DB_NAME]


def get_values_to_csv(symbol: str) -> str:
    db = get_db()
    collection = db[COLLECTION_NAME]
    cursor = collection.find(
        {"symbol": symbol, "provider": PROVIDER}, {"timestamp": 1, "_id": False, "value": 1, "source": 1}
    ).batch_size(20000)

    csv_name = f"{DATA_DIR}/{symbol}.csv"
    print(f"Fetching prices for: {symbol}")
    with open(csv_name, "w") as f:
        f.write("timestamp,value,total_sources,error_sources\n")
        for doc in cursor:
            errors, total = -1, -1
            if "source" in doc:
                errors = sum([1 for v in doc["source"].values() if v == "error"])
                total = len(doc["source"])
            f.write(f"{doc['timestamp']},{doc['value']}, {total}, {errors}\n")

    return csv_name


def fetch_data_on_records(df: pd.DataFrame, symbol: str) -> List[dict]:
    db = get_db()
    collection = db[COLLECTION_NAME]

    timestamps = df.index

    # timestamp to ms
    timestamps = [int(timestamp.value) // 10**6 for timestamp in timestamps]

    data = []

    for timestamp in timestamps:
        c = collection.find_one(
            {"symbol": symbol, "provider": PROVIDER, "timestamp": timestamp},
            {"timestamp": 1, "_id": False, "value": 1, "source": 1},
        )
        if c is None:
            print(f"ERROR: Missing record for {symbol} at {timestamp}")
        c["timestamp_human"] = pd.to_datetime(c["timestamp"], unit="ms", origin="unix")
        data.append(c)

    return data


def get_deviation(x: np.ndarray) -> float:
    return 100 * abs(x[0] - x[-1]) / min(x[0], x[-1])


def plot_prices_with_max_deviation(title: str, values: pd.DataFrame, max_deviations: list) -> None:
    plt.figure(figsize=(40, 10))
    plt.title(title)
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
    plt.savefig(f"{PLOTS_DIR}/{title.replace(' ', '_')}.png")


def get_outliers(df: pd.DataFrame):
    rolling = df.rolling(OUTLIER_WINDOW_SIZE)
    mean = rolling.mean()
    std = rolling.std()

    higher = mean + OUTLIER_STD_COUNT * std
    lower = mean - OUTLIER_STD_COUNT * std

    outliers = df[
        (df.value > higher.value) & (df.value > mean.value * (1.0 + OUTLIER_MIN_MEAN_DIFF))
        | (df.value < lower.value) & (df.value < mean.value * (1.0 - OUTLIER_MIN_MEAN_DIFF))
    ]

    return outliers


def plot_values_and_sources(title: str, df: pd.DataFrame):
    df_valid = df[df["total_sources"] > 0].copy()
    df_valid["error_ratio"] = df_valid["error_sources"] / df_valid["total_sources"]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(80, 20))
    fig.suptitle(title)

    ax1.plot(df_valid.index, df_valid["value"], label="Value", color="blue")
    ax1.set_ylabel("Value", fontsize=14)

    ax2.scatter(df_valid.index, df_valid["error_ratio"], label="Error sources ratio", color="red", s=4)
    ax2.set_ylabel("Sources Error Ratio", fontsize=14)
    ax2.set_xlabel("Time", fontsize=14)
    ax2.set_ylim(0, 1)

    plt.legend()
    fig.savefig(f"{PLOTS_DIR}/{title.replace(' ', '_')}.png")


def plot_prices_with_outliers(title: str, values: pd.DataFrame, outliers: pd.DataFrame) -> None:
    plt.figure(figsize=(40, 10))
    plt.title(title)
    plt.plot(values.index, values["value"], label="Value", color="blue")
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Value", fontsize=14)

    plt.scatter(outliers.index, outliers["value"], label="Outliers", color="red", alpha=0.5)

    plt.legend()
    plt.savefig(f"{PLOTS_DIR}/{title.replace(' ', '_')}_outliers.png")


def calculate_deviations(df: pd.DataFrame) -> List[Tuple[float, float]]:
    intervals_deviations = []
    for interval in INTERVALS:
        deviations = df.rolling(interval).apply(get_deviation, engine="numba", raw=True)
        intervals_deviations.append([deviations.idxmax()[0], deviations.max()[0]])

    return intervals_deviations


def process_values(csv_name: str) -> None:
    df = pd.read_csv(csv_name)

    df["timestamp"] = df["timestamp"].astype(int)

    time_range = df["timestamp"].max() - df["timestamp"].min()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", origin="unix")
    df = df.set_index("timestamp")
    symbol = csv_name.split("/")[-1].split(".")[0]

    message = "\n\n"
    message += f"Symbol: {symbol}\n"
    message += f"Data Points: {len(df)}\n"
    message += f"Date Range: {df.index[0]} -- {df.index[-1]}\n"
    message += f"Average Interval: {(time_range / len(df))/1000:0.2f}s\n"

    df_vals = df.loc[:, ["value"]]
    intervals_deviations = calculate_deviations(df_vals)
    plot_values_and_sources(f"{symbol} source error", df)
    plot_prices_with_max_deviation(symbol, df_vals, intervals_deviations)

    for (time, deviation), interval in zip(intervals_deviations, INTERVALS):
        message += f"Interval: {interval}, Max Deviation: {deviation:0.2f}%, Max Deviation Interval End: {time}\n"

    outliers = get_outliers(df_vals)

    if len(outliers) > 0:
        plot_prices_with_outliers(symbol, df_vals, outliers)
        outlier_data = fetch_data_on_records(outliers, symbol)
        message += f"\n\nOutliers for {symbol}:\n"
        for outlier in outlier_data:
            message += f"{outlier}\n"
        message += "\n\n"

        df_vals = df_vals.drop(outliers.index)
        intervals_deviations = calculate_deviations(df_vals)
        plot_prices_with_max_deviation(f"{symbol} outliers removed", df_vals, intervals_deviations)

        message += f"Symbol: {symbol} outliers removed\n"
        for (time, deviation), interval in zip(intervals_deviations, INTERVALS):
            message += f"Interval: {interval}, Max Deviation: {deviation:0.2f}%, Max Deviation Interval End: {time}\n"

    print(message)


def main():
    pool = ThreadPool(len(SYMBOLS))

    files = pool.map(get_values_to_csv, SYMBOLS)
    pool.close()
    pool.join()

    pool_mp = Pool(MAX_PROCESSES)
    pool_mp.map(process_values, files)


if __name__ == "__main__":
    main()
