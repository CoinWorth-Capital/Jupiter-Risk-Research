import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression

# Fetch historical price data for Solana and Bitcoin
def get_historical_prices(crypto_id, days=365):
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

# Perform linear regression to calculate beta
X = df['bitcoin_return'].values.reshape(-1, 1)  # Independent variable (Bitcoin returns)
y = df['solana_return'].values  # Dependent variable (Solana returns)

reg = LinearRegression()
reg.fit(X, y)

beta = reg.coef_[0]
print(f"Beta of Solana vs Bitcoin: {beta}")

# 30d Beta of Solana vs Bitcoin: 1.4021399602261384
# 90d Beta of Solana vs Bitcoin: 1.259827226799599
# 365d Beta of Solana vs Bitcoin: 1.1443149796466268