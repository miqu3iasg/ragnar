# Ref: https://docs.tavily.com/documentation/quickstart

import os

from dotenv import load_dotenv

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise RuntimeError(
        "TAVILY_API_KEY not found. Check if the .env file exists and contains the TAVILY_API_KEY variable."
    )
