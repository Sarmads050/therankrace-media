"""Brand logo library: fetch official logos from Wikipedia/Wikimedia file pages once, cache as PNG, reuse in every video.

    python logos.py            # fetch every brand in CANDIDATES that is not cached yet
    from logos import logo     # logo("Samsung") -> RGBA PIL image or None

Only the file pages are queried (title -> rendered PNG thumbnail). Nothing is executed. Brands whose logo cannot be found get
None and the video draws a lettered badge instead.
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from functools import lru_cache

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(os.path.dirname(HERE), "assets", "logos")
UA = {"User-Agent": "dataviz-video-script/1.0"}

# brand -> Wikimedia file titles to try, best first (the first one that exists is used)
CANDIDATES = {
    "Samsung": ["Samsung wordmark.svg", "Samsung Electronics logo (english).svg"],
    "Nokia": ["Nokia wordmark.svg", "Nokia - 2005 logo.svg"],
    "Huawei": ["Huawei Standard logo.svg"],
    "Oppo": ["OPPO LOGO 2019.svg", "OPPO logo.svg"],
    "Apple": ["Apple logo black.svg"],
    "Infinix": ["Infinix logo.svg"],
    "Vivo": ["Vivo (China) logo.svg"],
    "Motorola": ["Motorola new logo.svg"],
    "QMobile": ["Qmobile.PNG"],
    "Tecno": ["Tecno Mobile logo.svg"],
    "Xiaomi": ["Xiaomi logo (2021-).svg", "Xiaomi logo.svg"],
    "HTC": ["HTC.svg", "HTC logo.svg", "HTC Logo.svg", "HTC-Logo.svg", "HTC logo 2011.svg"],
    "LG": ["LG logo (2014).svg", "LG Electronics logo.svg", "LG symbol.svg", "LG logo.svg"],
    "Lenovo": ["Lenovo Logo.svg", "Lenovo logo (2015 onwards) 2.svg"],
    "Realme": ["Realme 2024 logo.svg"],
    "Sony": ["Sony logo.svg"],
    "Itel": ["Itel-mobile-logo-vector.svg"],
    "RIM": ["Blackberry Logo.svg", "Research In Motion Logo.svg"],
    "Google": ["Google \"G\" logo.svg", "Google \"G\" Logo.svg", "Google G Logo.svg", "Pixel wordmark.svg"],
}


def _get(url, tries=6):
    wait = 5
    for _ in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40).read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                time.sleep(wait)
                wait *= 2
                continue
            if e.code == 404:
                return None
            raise
        except Exception:  # noqa
            time.sleep(wait)
    return None


def fetch_one(brand, titles, width=600):
    for t in titles:
        time.sleep(2.5)
        api = ("https://en.wikipedia.org/w/api.php?action=query&format=json&prop=imageinfo&iiprop=url&iiurlwidth=%d&titles=%s"
               % (width, urllib.parse.quote("File:" + t)))
        raw = _get(api)
        if not raw:
            continue
        page = next(iter(json.loads(raw)["query"]["pages"].values()))
        info = (page.get("imageinfo") or [None])[0]
        if not info:
            continue
        url = info.get("thumburl") or info.get("url")
        time.sleep(2.5)
        data = _get(url)
        if not data:
            continue
        os.makedirs(DIR, exist_ok=True)
        path = os.path.join(DIR, brand + ".png")
        with open(path + ".tmp", "wb") as f:
            f.write(data)
        try:
            im = Image.open(path + ".tmp").convert("RGBA")
            im.load()
        except Exception:  # noqa
            os.remove(path + ".tmp")
            continue
        im.save(path)
        os.remove(path + ".tmp")
        return t
    return None


@lru_cache(maxsize=None)
def logo(brand):
    p = os.path.join(DIR, brand + ".png")
    return Image.open(p).convert("RGBA") if os.path.exists(p) else None


if __name__ == "__main__":
    want = sys.argv[1:] or list(CANDIDATES)
    for b in want:
        if os.path.exists(os.path.join(DIR, b + ".png")):
            print(b, "cached")
            continue
        got = fetch_one(b, CANDIDATES[b])
        print(b, "->", got or "NOT FOUND", flush=True)
