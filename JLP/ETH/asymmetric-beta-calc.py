import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def fetch_price_data(coin_id, vs_currency, days):
    """Fetch historical price data from CoinGecko."""
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
    params = {
        'vs_currency': vs_currency,
        'days': str(days),  # historical data for specified days
        'interval': 'daily'
    }
    response = requests.get(url, params=params)
    data = response.json()
    return pd.DataFrame(data['prices'], columns=['timestamp', 'price'])

def calculate_asymmetric_beta(asset_prices, benchmark_prices):
    """Calculate the asymmetric beta (upside and downside)."""
    # Calculate returns
    asset_returns = np.log(asset_prices / asset_prices.shift(1))
    benchmark_returns = np.log(benchmark_prices / benchmark_prices.shift(1))

    # Upside beta: beta during positive benchmark returns
    upside_returns = asset_returns[benchmark_returns > 0]
    upside_benchmark_returns = benchmark_returns[benchmark_returns > 0]
    upside_cov = np.cov(upside_returns[1:], upside_benchmark_returns[1:])[0, 1]
    upside_beta = upside_cov / np.var(upside_benchmark_returns[1:])

    # Downside beta: beta during negative benchmark returns
    downside_returns = asset_returns[benchmark_returns < 0]
    downside_benchmark_returns = benchmark_returns[benchmark_returns < 0]
    downside_cov = np.cov(downside_returns[1:], downside_benchmark_returns[1:])[0, 1]
    downside_beta = downside_cov / np.var(downside_benchmark_returns[1:])
    
    return upside_beta, downside_beta

def main():
    # Define the assets (Ethereum and Bitcoin)
    ethereum_id = 'ethereum'
    bitcoin_id = 'bitcoin'
    
    # Define the period for price data (90d, 365d)
    periods = [365]

    for period in periods:
        # Fetch price data
        eth_data = fetch_price_data(ethereum_id, 'usd', period)
        btc_data = fetch_price_data(bitcoin_id, 'usd', period)

        # Calculate asymmetric beta
        upside_beta, downside_beta = calculate_asymmetric_beta(eth_data['price'], btc_data['price'])
        
        print(f'{period}d Upside Beta of Ethereum vs Bitcoin: {upside_beta:.3f}')
        print(f'{period}d Downside Beta of Ethereum vs Bitcoin: {downside_beta:.3f}')

if __name__ == "__main__":
    main()
