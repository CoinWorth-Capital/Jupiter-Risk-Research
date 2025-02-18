import os
from dotenv import load_dotenv

load_dotenv()

COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")
COINGECKO_BASE_URL = "https://pro-api.coingecko.com/api/v3/"