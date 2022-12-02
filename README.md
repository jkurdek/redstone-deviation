# Token deviation analysis
Script for fetching historical values of tokens and anylysing their maximum deviation.

## Requirments
- Python

## Set-up
- Install the required packages
```
pip install -r requirements.txt
```
Alternatively:
```
pip install pandas
pip install pymongo
```
- Create variable storing mongoDB key
``` 
export CONNECTION_STRING=<KEY>
```

## Run
```
python deviation.py
```

***NOTE***: The script will take some time to fetch all the data.

