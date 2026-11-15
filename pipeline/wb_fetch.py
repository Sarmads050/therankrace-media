"""World Bank top-N fetcher. python wb_fetch.py INDICATOR [N] -> prints latest common year top N real countries."""
import json, sys, urllib.request
UA = {"User-Agent": "dataviz-video-script/1.0"}
def get(u, tries=6):
    import time
    for i in range(tries):
        try:
            return json.loads(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=240).read())
        except Exception:
            if i == tries - 1: raise
            time.sleep(20)
def top(ind, n=12, years=range(2025, 2018, -1)):
    meta = get("https://api.worldbank.org/v2/country?format=json&per_page=400")[1]
    real = {c["id"]: c for c in meta if c["region"]["value"] != "Aggregates"}
    for y in years:
        rows = get(f"https://api.worldbank.org/v2/country/all/indicator/{ind}?format=json&date={y}&per_page=400")
        if len(rows) < 2 or not rows[1]: continue
        data = [(r["country"]["value"], r["countryiso3code"], r["value"]) for r in rows[1] if r["countryiso3code"] in real and r["value"]]
        if len(data) > 150:
            data.sort(key=lambda t: -t[2]); return y, data[:n], rows[0].get("lastupdated")
    return None
if __name__ == "__main__":
    r = top(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 12)
    print(json.dumps(r, indent=1))
