"""Central configuration for the travel planner."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# LLM (Groq). Check console.groq.com/docs/models: some models (e.g. Llama 3.3 70B) are enterprise-only.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
LLM_TEMPERATURE = 0.2
# gpt-oss models accept "low" | "medium" | "high". Lower is faster; tool-driven planning does not need much.
GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low")

# Weather
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT_SECONDS = 10

# Alternate spellings seen in user input -> canonical dataset city names
CITY_ALIASES = {
    "bengaluru": "Bangalore",
    "bombay": "Mumbai",
    "madras": "Chennai",
    "calcutta": "Kolkata",
    "new delhi": "Delhi",
    "panaji": "Goa",
    "panjim": "Goa",
}
