import pandas as pd
import numpy as np
import quandl

quandl.ApiConfig.api_key = 'V59x5f6HPkpBbtHbP6Pa'

def download_equity(ticker):
    # equities
    df = quandl.get_table('WIKI/PRICES', qopts = { 'columns': ['date', 'close'] }, ticker = [ticker], paginate=True)
    df = df.set_index('date')

    # calculate daily returns
    df['return'] = df['close'] / df['close'].shift(1)
    df['lreturn'] = df['return'].apply(np.log)

    # save locally
    filename = ticker + '.pkl'
    df.to_pickle(filename)

    return df

def load_equity(ticker):
    return pd.read_pickle(ticker + '.pkl')
