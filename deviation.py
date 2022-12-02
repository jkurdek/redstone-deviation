import os
from multiprocessing.dummy import Pool as ThreadPool

import pandas as pd
from pymongo import MongoClient
from pymongo.collection import Collection

CONNECTION_STRING = os.getenv("CONNECTION_STRING")
DATA_DIR = "data"
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


def calculate_deviations(csv_name: str) -> None:
    print("\n\n")
    df = pd.read_csv(csv_name)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", origin="unix")
    df = df.set_index("timestamp")
    symbol = csv_name.split("/")[-1].split(".")[0]
    for interval in INTERVALS:
        deviations = df.rolling(interval).apply(lambda x: get_deviation(x[0], x[-1]))
        print(
            f"{symbol}: Interval: {interval}, Max Deviation: {deviations.max()[0]:0.2f}%, Max Deviation Interval End: {deviations.idxmax()[0]}"
        )


def main():
    pool = ThreadPool(len(SYMBOLS))

    files = pool.map(get_values_to_csv, SYMBOLS)

    for name in files:
        calculate_deviations(name)


if __name__ == "__main__":
    main()
