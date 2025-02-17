import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from time import sleep

def fetch_price_data(coin_id, vs_currency, days):
    """Fetch historical price data from CoinGecko."""
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
    params = {
        'vs_currency': vs_currency,
        'days': str(days),  # historical data for specified days
        'interval': 'daily'
    }
    
    # Make the API request
    response = requests.get(url, params=params)
    
    # Debugging: Print out the raw response content
    # print(f"Response for {coin_id} over the last {days} days: {response.status_code}")
    # print(response.json())  # Print the entire JSON response for inspection
    
    # Check if 'prices' key exists in the response
    data = response.json()
    if 'prices' not in data:
        print(f"Warning: No price data available for {coin_id} over the last {days} days.")
        return pd.DataFrame()  # Return an empty DataFrame to handle the error gracefully
    
    return pd.DataFrame(data['prices'], columns=['timestamp', 'price'])

def calculate_beta(asset_prices, benchmark_prices):
    """Calculate the beta between asset and benchmark."""
    # Returns calculation
    asset_returns = np.log(asset_prices / asset_prices.shift(1))
    benchmark_returns = np.log(benchmark_prices / benchmark_prices.shift(1))
    
    # Beta calculation using covariance and variance
    covariance = np.cov(asset_returns[1:], benchmark_returns[1:])[0, 1]
    benchmark_variance = np.var(benchmark_returns[1:])
    beta = covariance / benchmark_variance
    return beta

def main():
    # Define the assets (Ethereum and Bitcoin)
    ethereum_id = 'ethereum'
    bitcoin_id = 'bitcoin'
    
    # Define the period for price data (30d, 90d, 365d)
    periods = [365]

    for period in periods:
        # Fetch price data
        eth_data = fetch_price_data(ethereum_id, 'usd', period)
        btc_data = fetch_price_data(bitcoin_id, 'usd', period)

        # Check if data was successfully fetched
        if eth_data.empty or btc_data.empty:
            print(f"Skipping {period}d period due to missing data.")
            continue  # Skip calculation for this period if data is empty

        # Calculate Beta
        beta = calculate_beta(eth_data['price'], btc_data['price'])
        print(f'{period}d Basic Beta of Ethereum vs Bitcoin: {beta:.3f}')

if __name__ == "__main__":
    main()
