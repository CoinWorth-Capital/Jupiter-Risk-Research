import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression

# Fetch historical price data for Solana and Bitcoin
def get_historical_prices(crypto_id, days=365):  # Increased to 90 days for better data
    url = f'https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart?vs_currency=usd&days={days}'
    response = requests.get(url)
    data = response.json()
    prices = [price[1] for price in data['prices']]  # Get prices from response
    return prices

# Get historical price data for both Solana and Bitcoin
solana_prices = get_historical_prices('solana')
bitcoin_prices = get_historical_prices('bitcoin')

# Convert to DataFrame for easier manipulation
df = pd.DataFrame({
    'solana': solana_prices,
    'bitcoin': bitcoin_prices
})

# Calculate daily returns (percentage change)
df['solana_return'] = df['solana'].pct_change()
df['bitcoin_return'] = df['bitcoin'].pct_change()

# Drop NaN values that result from percentage change calculation
df.dropna(inplace=True)

# Separate upside and downside returns
df['bitcoin_up'] = np.where(df['bitcoin_return'] > 0, df['bitcoin_return'], np.nan)
df['bitcoin_down'] = np.where(df['bitcoin_return'] < 0, df['bitcoin_return'], np.nan)

df['solana_up'] = np.where(df['bitcoin_return'] > 0, df['solana_return'], np.nan)
df['solana_down'] = np.where(df['bitcoin_return'] < 0, df['solana_return'], np.nan)

# Filter out NaN values
df_up = df.dropna(subset=['bitcoin_up', 'solana_up'])
df_down = df.dropna(subset=['bitcoin_down', 'solana_down'])

# Upside Beta calculation (regression only on positive Bitcoin returns)
X_up = df_up['bitcoin_up'].values.reshape(-1, 1)
y_up = df_up['solana_up'].values

reg_up = LinearRegression()
reg_up.fit(X_up, y_up)

upside_beta = reg_up.coef_[0]

# Downside Beta calculation (regression only on negative Bitcoin returns)
X_down = df_down['bitcoin_down'].values.reshape(-1, 1)
y_down = df_down['solana_down'].values

reg_down = LinearRegression()
reg_down.fit(X_down, y_down)

downside_beta = reg_down.coef_[0]

# Output the results
print(f"Upside Beta of Solana vs Bitcoin: {upside_beta}")
print(f"Downside Beta of Solana vs Bitcoin: {downside_beta}")
