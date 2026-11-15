"""Country flag library (Wikipedia file pages -> PNG cache in assets/flags/<ISO3>.png).
python flags.py [ISO3 ...]    from flags import flag"""
import os, sys
from functools import lru_cache
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logos as L

DIR = os.path.join(os.path.dirname(L.HERE), "assets", "flags")
NAMES = {"IND": "India", "CHN": "China", "USA": "the United States", "IDN": "Indonesia", "PAK": "Pakistan", "NGA": "Nigeria",
         "BRA": "Brazil", "BGD": "Bangladesh", "RUS": "Russia", "ETH": "Ethiopia", "DEU": "Germany", "JPN": "Japan",
         "GBR": "the United Kingdom", "FRA": "France", "ITA": "Italy", "CAN": "Canada", "MEX": "Mexico", "ESP": "Spain",
         "KOR": "South Korea", "AUS": "Australia", "SAU": "Saudi Arabia", "TUR": "Turkey", "ARE": "the United Arab Emirates",
         "EGY": "Egypt", "IRN": "Iran", "PHL": "the Philippines", "VNM": "Vietnam", "COD": "the Democratic Republic of the Congo",
         "ZAF": "South Africa", "THA": "Thailand", "NLD": "the Netherlands", "CHE": "Switzerland", "POL": "Poland", "ARG": "Argentina"}

def fetch(iso):
    path = os.path.join(DIR, iso + ".png")
    if os.path.exists(path): return "cached"
    old = L.DIR; L.DIR = DIR
    try:
        return L.fetch_one(iso, [f"Flag of {NAMES[iso]}.svg"]) or "NOT FOUND"
    finally:
        L.DIR = old

@lru_cache(maxsize=None)
def flag(iso):
    p = os.path.join(DIR, iso + ".png")
    return Image.open(p).convert("RGBA") if os.path.exists(p) else None

if __name__ == "__main__":
    for iso in sys.argv[1:] or list(NAMES):
        print(iso, fetch(iso), flush=True)
