"""Race topic builders: full-history World Bank series -> data/race/<slug>.json (config for bar_race_gen.py).

    python race_topics.py fetch            # refresh data/series/*.json from the World Bank API (1960-latest)
    python race_topics.py pop gdp          # build configs (all events/captions are computed from the data and asserted)
Every number shown on screen is computed here from the series; a wrong statement fails an assert instead of reaching a video.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
SER = os.path.join(ROOT, "data", "series")
OUT = os.path.join(ROOT, "data", "race")
SHORT = {"Russian Federation": "Russia", "United States": "USA", "United Kingdom": "UK", "Korea, Rep.": "South Korea",
         "Egypt, Arab Rep.": "Egypt", "Iran, Islamic Rep.": "Iran", "Turkiye": "Turkey", "Congo, Dem. Rep.": "DR Congo",
         "Netherlands": "Netherlands", "Hong Kong SAR, China": "Hong Kong", "Macao SAR, China": "Macao", "Somalia, Fed. Rep.": "Somalia",
         "Korea, Dem. People's Rep.": "North Korea", "Venezuela, RB": "Venezuela", "Bahamas, The": "Bahamas", "Gambia, The": "Gambia",
         "Yemen, Rep.": "Yemen", "Micronesia, Fed. Sts.": "Micronesia", "Lao PDR": "Laos", "Syrian Arab Republic": "Syria",
         "Kyrgyz Republic": "Kyrgyzstan", "Slovak Republic": "Slovakia", "Congo, Rep.": "Congo", "Sint Maarten (Dutch part)": "Sint Maarten",
         "Virgin Islands (U.S.)": "US Virgin Islands", "Naoero": "Nauru", "Cote d'Ivoire": "Ivory Coast", "Czechia": "Czechia",
         "St. Kitts and Nevis": "St Kitts", "St. Lucia": "St Lucia", "Trinidad and Tobago": "Trinidad"}
# Channel Islands (no flag) + tiny territories / micro-states: "richest per person" or "longest life" topped by Monaco, Cayman Islands or
# Isle of Man is not what viewers expect (they expect Switzerland, Norway, Singapore) and nobody relates to them
EXCLUDE = {"CHI", "MCO", "SMR", "LIE", "CYM", "IMN", "BMU", "PYF", "AND", "FRO", "GRL", "ABW", "CUW", "SXM", "MAF", "VGB", "TCA", "NCL",
           "GUM", "ASM", "MNP", "VIR", "KNA", "DMA", "PLW", "MHL", "FSM", "TUV", "NRU", "GIB", "SYC", "ATG", "GRD", "LCA", "VCT", "BRB",
           "BHS"}
INDICATORS = {"pop": "SP.POP.TOTL", "gdp": "NY.GDP.MKTP.CD"}


def fetch(slug):
    import wb_fetch as W
    meta = W.get("https://api.worldbank.org/v2/country?format=json&per_page=400")[1]
    real = {c["id"]: c["name"] for c in meta if c["region"]["value"] != "Aggregates"}
    out, page = {}, 1
    while True:
        r = W.get(f"https://api.worldbank.org/v2/country/all/indicator/{INDICATORS[slug]}?format=json&date=1960:2026&per_page=20000&page={page}")
        for x in r[1]:
            if x["countryiso3code"] in real and x["value"] is not None:
                out.setdefault(x["countryiso3code"], {})[int(x["date"])] = x["value"]
        if page >= r[0]["pages"]:
            break
        page += 1
    json.dump({"indicator": INDICATORS[slug], "updated": r[0].get("lastupdated"), "names": real, "data": out},
              open(os.path.join(SER, slug + ".json"), "w"))


class DS:
    def __init__(self, slug):
        J = json.load(open(os.path.join(SER, slug + ".json")))
        self.d = {k: {int(y): v for y, v in s.items()} for k, s in J["data"].items() if k not in EXCLUDE}
        self.names = J["names"]
        self.updated = J["updated"]
        self.y0 = 1960
        self.y1 = max(max(s) for k, s in self.d.items() if k in ("USA", "IND", "CHN"))
        self.years = list(range(self.y0, self.y1 + 1))
        self.cache = {}

    def v(self, k, y):
        return self.d.get(k, {}).get(y, 0.0)

    def order(self, y):
        if y not in self.cache:
            self.cache[y] = [c for _, c in sorted(((s.get(y, 0.0), c) for c, s in self.d.items()), reverse=True)]
        return self.cache[y]

    def rank(self, k, y):
        return self.order(y).index(k) + 1 if self.v(k, y) else None

    def top10_years(self, k):
        return [y for y in self.years if (self.rank(k, y) or 99) <= 10]

    def ever(self):
        e = []
        for y in self.years:
            for c in self.order(y)[:10]:
                if c not in e:
                    e.append(c)
        return e

    def name(self, k):
        n = self.names[k]
        return SHORT.get(n, n)

    def passes_for_good(self, a, b, lo=None):
        """first year after which a stays above b to the end"""
        ys = [y for y in self.years if (lo is None or y >= lo)]
        good = None
        for y in reversed(ys):
            if self.v(a, y) > self.v(b, y):
                good = y
            else:
                break
        return good


def pace(events, y0, y1, fast=0.85, slow=0.55, lead=1.0, lag=2.0):
    """segments [[m_end, years_per_second]] with slow zones around the event years"""
    zones = sorted((max(y0, e - lead), min(y1, e + lag)) for e in events)
    merged = []
    for a, b in zones:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    segs, cur = [], float(y0)
    # a slow start so the first caption can be read
    first_slow = min(y0 + 2.0, merged[0][0]) if merged and merged[0][0] > y0 + 2.0 else None
    if first_slow:
        segs.append([first_slow - y0, slow])
        cur = first_slow
    for a, b in merged:
        if a > cur:
            segs.append([a - y0, fast])
            cur = a
        if b > cur:
            segs.append([b - y0, slow])
            cur = b
    if cur < y1:
        segs.append([y1 - y0, fast])
    return segs


def assemble(slug, ds, title1, title2, fmt, captions, pills, hits, source, highlight, tail_caps):
    ents = ds.ever()
    cfg = dict(slug=slug, years=ds.years, entities=[{"iso": c, "name": ds.name(c)} for c in ents],
               series={c: [ds.v(c, y) for y in ds.years] for c in ents}, title1=title1, title2=title2, fmt=fmt,
               captions=captions + tail_caps, pills=pills, hits=hits, source=source, highlight=highlight, tail=9.0,
               seg=pace([p["at"] for p in pills if not isinstance(p["at"], dict)], ds.years[0], ds.years[-1]),
               built=time.strftime("%Y-%m-%d %H:%M"), updated=ds.updated)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, slug + ".json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False)
    return cfg


def M(x):
    return f"{x / 1e6:,.0f} million"


def Bn(x):
    return f"{x / 1e9:,.2f} billion"


def usd(x):
    return f"${x / 1e12:,.1f} trillion" if x >= 1e12 else f"${x / 1e9:,.0f} billion"


# ------------------------------------------------------------------ topics
def passes(ds, a, b, y):
    """a overtakes b in year y (adjacent top-10 ranks)"""
    return ds.rank(a, y - 1) == ds.rank(b, y - 1) + 1 and ds.rank(a, y) == ds.rank(b, y) - 1 and ds.v(a, y) > ds.v(b, y)


def pop():
    ds = DS("pop")
    y0, y1 = ds.y0, ds.y1
    assert y1 == 2025
    r = lambda k, y: ds.rank(k, y)  # noqa
    n = ds.name
    assert [ds.order(y0)[i] for i in range(3)] == ["CHN", "IND", "USA"]
    pak_in = ds.top10_years("PAK")[0]
    assert r("PAK", pak_in) == 10 and (r("PAK", pak_in - 1) or 99) == 11
    assert passes(ds, "IDN", "RUS", 1976) and r("IDN", 1976) == 4
    ng = ds.top10_years("NGA")[0]
    assert "DEU" in ds.order(ng - 1)[:10] and "DEU" not in ds.order(ng)[:10]
    c1b = next(y for y in ds.years if ds.v("CHN", y) >= 1e9)
    i1b = next(y for y in ds.years if ds.v("IND", y) >= 1e9)
    assert c1b == ng and i1b == 1998
    assert passes(ds, "PAK", "JPN", 1993) and r("PAK", 1993) == 7
    assert r("PAK", 2003) == 6 and r("BRA", 2003) == 5
    p5 = next(y for y in ds.years if r("PAK", y) == 5)
    passed = ds.order(p5 - 1)[4]
    assert r("PAK", p5 - 1) == 6 and ds.v("PAK", p5) > ds.v(passed, p5) and all(r("PAK", y) <= 5 for y in range(p5, y1 + 1))
    ind = ds.passes_for_good("IND", "CHN")
    assert ind and r("IND", ind) == 1 and r("CHN", ind) == 2
    grow = ds.v("PAK", y1) / ds.v("PAK", y0)
    assert r("PAK", y1) == 5 and r("IND", y1) == 1 and r("CHN", y1) == 2 and r("USA", y1) == 3
    caps = [
        dict(at=y0, text=f"In 1960, China is number one with {M(ds.v('CHN', y0))} people. India is second."),
        dict(at=1962.3, text="Each bar is a country. A longer bar means more people."),
        dict(at=pak_in, text=f"{pak_in}: Pakistan enters the Top 10 for the first time, with {M(ds.v('PAK', pak_in))} people."),
        dict(at=1976, text="1976: Indonesia passes Russia and becomes #4."),
        dict(at=ng, text=f"{ng}: China reaches 1 billion people. Nigeria enters the Top 10."),
        dict(at=1993, text="1993: Pakistan passes Japan and becomes #7."),
        dict(at=i1b, text=f"{i1b}: India reaches 1 billion people."),
        dict(at=2003, text=f"2003: Pakistan is #6 with {M(ds.v('PAK', 2003))}. Brazil is just ahead with {M(ds.v('BRA', 2003))}."),
        dict(at=p5, text=f"{p5}: Pakistan passes {n(passed)} and becomes #5!"),
        dict(at=2015, text=f"Since 1960, Pakistan's people grew {grow:.1f} times: from {M(ds.v('PAK', y0))} to {M(ds.v('PAK', y1))}."),
        dict(at=ind, text=f"{ind}: India passes China. India is now number one!"),
    ]
    tail = [dict(at={"tail": 0.4}, text=f"2025: India {Bn(ds.v('IND', y1))}, China {Bn(ds.v('CHN', y1))}. Pakistan is #5 with {M(ds.v('PAK', y1))}."),
            dict(at={"tail": 5.0}, text="Which country do you live in? Tell us in the comments.")]
    pills = [dict(at=pak_in, dur=3.6, text="PAKISTAN ENTERS TOP 10"), dict(at=ng, dur=3.6, text="CHINA: 1 BILLION PEOPLE"),
             dict(at=p5, dur=3.8, text=f"PAKISTAN PASSES {n(passed).upper()}: #5!"),
             dict(at=ind, dur=4.2, text="INDIA PASSES CHINA!"), dict(at={"tail": 0.4}, dur=4.2, text="PAKISTAN IS #5 TODAY")]
    return assemble("pop", ds, "TOP 10 MOST POPULATED", "COUNTRIES 1960-2025", "count", caps, pills, [pak_in, ng, p5, ind],
                    "Data: World Bank, total population, 1960 to 2025 (updated %s). Flags: Wikipedia." % ds.updated, "PAK", tail)


def gdp():
    ds = DS("gdp")
    y0, y1 = ds.y0, ds.y1
    assert y1 == 2025
    r = lambda k, y: ds.rank(k, y)  # noqa
    assert ds.order(y0)[:2] == ["USA", "DEU"]
    jd = next(y for y in ds.years if ds.v("JPN", y) > ds.v("DEU", y) and r("JPN", y) == 2 and r("JPN", y - 1) == 3)
    assert all(r("JPN", y) == 2 for y in range(1972, 2010))
    cd = ds.passes_for_good("CHN", "DEU", 2000)
    cj = ds.passes_for_good("CHN", "JPN", 2000)
    assert r("CHN", cd) == 3 and r("CHN", cj) == 2, (cd, cj)
    ratio = {y: ds.v("JPN", y) / ds.v("USA", y) for y in ds.years}
    jp = max(ratio, key=ratio.get)
    cr = {y: ds.v("CHN", y) / ds.v("USA", y) for y in ds.years}
    cp = max(cr, key=cr.get)
    gc = ds.v("CHN", y1) / ds.v("CHN", y0)
    pak_rank = r("PAK", y1)
    assert r("USA", y1) == 1 and r("CHN", y1) == 2 and 30 < pak_rank < 50
    assert 1980 < jp < 2000 and r("CHN", 1980) > 8 and cp > 2015
    assert ds.v("USA", y0) > sum(ds.v(c, y0) for c in ds.order(y0)[1:5])
    assert ds.order(1975)[:2] == ["USA", "JPN"] and ds.top10_years("RUS")[0] == 1988 and r("CHN", 2000) == 6
    assert ds.order(2015)[:3] == ["USA", "CHN", "JPN"]
    caps = [
        dict(at=y0, text=f"In 1960, the USA is number one with {usd(ds.v('USA', y0))}. Germany is second."),
        dict(at=1962.3, text="Each bar is a country. A longer bar means a bigger economy."),
        dict(at=jd, text=f"{jd}: Japan passes Germany and becomes #2!"),
        dict(at=1975, text=f"1975: The USA is still #1 with {usd(ds.v('USA', 1975))}. Japan is #2 with {usd(ds.v('JPN', 1975))}."),
        dict(at=1980, text=f"In 1980, China is only #{r('CHN', 1980)} with {usd(ds.v('CHN', 1980))}. Watch it grow."),
        dict(at=1988, text="1988: Russia's data starts. It enters the Top 10."),
        dict(at=jp, text=f"{jp}: Japan reaches {ratio[jp] * 100:.0f}% of the US economy. Its best year."),
        dict(at=2000, text=f"In 2000, China is #6 with {usd(ds.v('CHN', 2000))}."),
        dict(at=cd, text=f"{cd}: China passes Germany and becomes #3."),
        dict(at=cj, text=f"{cj}: China passes Japan. China is now #2!"),
        dict(at=2015, text=f"2015: China has {usd(ds.v('CHN', 2015))}. That is {ds.v('CHN', 2015) / ds.v('JPN', 2015):.1f} times bigger than #3 Japan."),
        dict(at=cp, text=f"{cp}: China reaches {cr[cp] * 100:.0f}% of the US economy. Its best year."),
    ]
    tail = [dict(at={"tail": 0.4}, text=f"2025: USA {usd(ds.v('USA', y1))}, China {usd(ds.v('CHN', y1))}. China grew {gc:,.0f} times since 1960."),
            dict(at={"tail": 5.0}, text=f"Pakistan is #{pak_rank} with {usd(ds.v('PAK', y1))}. Which country is next? Comment below.")]
    pills = [dict(at=jd, dur=3.6, text="JAPAN PASSES GERMANY: #2"), dict(at=cd, dur=3.4, text="CHINA PASSES GERMANY: #3"),
             dict(at=cj, dur=4.0, text="CHINA PASSES JAPAN: #2!"), dict(at=cp, dur=3.8, text=f"CHINA: {cr[cp] * 100:.0f}% OF THE USA"),
             dict(at={"tail": 0.4}, dur=4.2, text=f"PAKISTAN IS #{pak_rank} TODAY")]
    return assemble("gdp", ds, "TOP 10 BIGGEST", "ECONOMIES 1960-2025", "usd", caps, pills, [jd, 1988, cd, cp],
                    "Data: World Bank, GDP in current US dollars, 1960 to 2025 (updated %s). Russia starts 1988. Flags: Wikipedia." % ds.updated,
                    "PAK", tail)


BUILD = {"pop": pop, "gdp": gdp}

if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "fetch":
        for s in args[1:] or INDICATORS:
            fetch(s)
            print("fetched", s)
        sys.exit()
    for s in args or BUILD:
        c = BUILD[s]()
        print("==", s, len(c["years"]), "years,", len(c["entities"]), "entities")
        for x in c["captions"]:
            print("   cap", x["at"], x["text"])
        for x in c["pills"]:
            print("   pill", x["at"], x["text"])
        print("   hits", c["hits"], "seg", [[round(a, 1), b] for a, b in c["seg"]])
