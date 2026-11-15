"""Topic chooser: decides WHICH Top-10 race to make next, so the videos are the ones viewers are most likely to enjoy.

No AI is involved - it is a transparent score:
  1. interest(slug)   story quality computed from the data itself (lead changes, big rivalries, Pakistan's place, Pakistan climbing)
  2. PRIOR[category]  what kinds of rankings usually catch attention (money, power, people ...)
  3. feedback()       what YOUR audience actually watched: average Instagram views per category from the last 60 days
                      (read from Buffer; ignored until enough posts have numbers)
  4. variety          not the same category as the last 3 posts, and the two daily slots always use different categories
Every topic is used once only (see daily.used_topics).
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import race_auto as RA  # noqa

CAT = {
    "gdp": "economy", "gdppc": "economy", "gdpppp": "economy", "gni": "economy", "exports": "trade", "imports": "trade", "merchexp": "trade",
    "merchimp": "trade", "hightech": "trade", "reserves": "finance", "fdiin": "finance", "fdiout": "finance", "remittances": "finance",
    "remitpaid": "finance", "ictexp": "tech", "royalty": "tech", "servexp": "trade", "servimp": "trade", "water": "climate", "bigcity": "people",
    "mcity": "people", "airfreight": "transport", "gdppc": "economy", "milpct": "military", "extdebt": "finance", "debtservice": "finance", "savings": "finance", "capital": "finance", "mktcap": "finance",
    "manufacturing": "industry", "services": "industry", "industry": "industry", "agriculture": "food", "cereal": "food", "fish": "food",
    "aqua": "food", "fishcatch": "food", "agland": "food", "arable": "food", "cerealyield": "food", "consumption": "economy", "govspend": "economy",
    "military": "military", "arms": "military", "armsimp": "military", "milpersonnel": "military",
    "pop": "people", "urban": "people", "rural": "people", "labor": "people", "kids": "people", "pop65": "people", "workage": "people",
    "female": "people", "male": "people", "tertiary": "people", "primary": "people", "secondary": "people",
    "mobile": "tech", "broadband": "tech", "landlines": "tech", "patents": "tech", "nonresid": "tech", "journals": "tech", "internet": "tech",
    "co2": "climate", "ghg": "climate", "methane": "climate", "n2o": "climate", "co2pc": "climate", "forest": "climate", "elecpc": "climate",
    "energypc": "climate", "renewelec": "climate",
    "airpax": "transport", "airdep": "transport", "containers": "transport", "railkm": "transport",
    "healthpc": "health", "lifeexp": "health", "beds": "health", "doctors": "health", "fertility": "health", "aid": "finance",
}
PRIOR = {"military": 1.25, "economy": 1.2, "finance": 1.15, "trade": 1.1, "people": 1.15, "tech": 1.1, "climate": 1.0, "industry": 1.0,
         "transport": 0.95, "health": 1.0, "food": 0.9}
BIG = {"USA", "China", "India", "Russia", "Japan", "Germany", "UK", "Pakistan", "Brazil", "Saudi Arabia"}


def cat(slug):
    return CAT.get(slug, "other")


def interest(slug):
    """(score, reasons) from the cached data; offline"""
    refresh = os.environ.pop("REFRESH", None)
    try:
        cfg = RA.build(slug)
    finally:
        if refresh is not None:
            os.environ["REFRESH"] = refresh
    pills = [p["text"] for p in cfg["pills"] if isinstance(p.get("text"), str)]
    why, s = [], 0.0
    n1 = sum(1 for t in pills if "PASSES" in t and t.endswith("#1!"))
    n2 = sum(1 for t in pills if "PASSES" in t) - n1
    if n1:
        s += 12 * min(n1, 3)
        why.append(f"{n1} change(s) at number one")
    if n2:
        s += 4 * min(n2, 4)
        why.append(f"{n2} other overtake(s)")
    hl = cfg["meta"].get("hl_rank")
    if hl:
        b = 30 if hl <= 3 else 25 if hl <= 10 else 15 if hl <= 20 else 8 if hl <= 40 else 3
        s += b
        why.append(f"Pakistan #{hl} today")
    if any("PAKISTAN ENTERS" in t or t.startswith("PAKISTAN: #") for t in pills):
        s += 10
        why.append("Pakistan climbs in the race")
    top2 = {cfg["meta"].get("leader"), cfg["meta"].get("second")}
    if {"USA", "China"} <= top2:
        s += 10
        why.append("USA vs China at the top")
    elif len(top2 & BIG) == 2:
        s += 5
        why.append("two big names at the top")
    g = cfg["meta"].get("y1", 0)
    if g and g >= __import__("datetime").date.today().year - 2:
        s += 5
        why.append("fresh data")
    return s, why


def rank_topics(unused, recent, fb=None):
    """-> list of (slug, final_score, explanation), best first. recent = slugs posted lately, newest first."""
    fb = fb or {}
    recent_cats = {cat(r) for r in recent[:3]}
    rows = []
    for slug in unused:
        try:
            base, why = interest(slug)
        except Exception as e:  # noqa  no cached data -> neutral score
            base, why = 20.0, [f"no cached data ({e.__class__.__name__})"]
        c = cat(slug)
        mult = PRIOR.get(c, 1.0)
        score = (30 + base) * mult
        if c in fb:
            score *= fb[c]
            why.append(f"audience x{fb[c]:.2f}")
        if c in recent_cats:
            score *= 0.6
            why.append("same category as a recent post (x0.6)")
        rows.append((slug, round(score, 1), f"{c}: " + ", ".join(why)))
    rows.sort(key=lambda r: -r[1])
    return rows


def split_slots(rows):
    """deterministic: slot1 gets the best topic, slot2 the best topic of ANOTHER category; the rest alternate as fall-backs"""
    slugs = [r[0] for r in rows]
    if not slugs:
        return [], []
    p1 = slugs[0]
    p2 = next((s for s in slugs[1:] if cat(s) != cat(p1)), slugs[1] if len(slugs) > 1 else None)
    rest = [s for s in slugs if s not in (p1, p2)]
    return [p1] + rest[0::2], ([p2] if p2 else []) + rest[1::2]


# ---------------------------------------------------------------- audience feedback (best effort, never fatal)
def feedback(gql, org, ch, log=print):
    """{category: multiplier 0.7..1.4} from Buffer post metrics of the last 60 days; {} when unavailable or too little data"""
    try:
        q = ('query { posts(first: 100, input: {organizationId: "%s", filter: {status: [sent], channelIds: ["%s"]}}) '
             '{ edges { node { id sentAt assets { ... on VideoAsset { source } } metrics { name type value } } } } }' % (org, ch))
        edges = gql(q)["data"]["posts"]["edges"]
        per = {}
        for e in edges:
            n = e["node"]
            src = next((a.get("source") for a in n.get("assets") or [] if a.get("source")), "")
            mm = re.search(r"/download/(.+?)-race-", src or "")
            ms = {(str(m.get("type")) + " " + str(m.get("name"))).lower(): m.get("value") for m in n.get("metrics") or []}
            val = None
            for want in ("view", "play", "impression", "reach", "like"):  # best available engagement number
                hit = [v for k, v in ms.items() if want in k and v is not None]
                if hit:
                    val = float(hit[0])
                    break
            if val is None or not mm:
                continue
            per.setdefault(cat(mm.group(1)), []).append(val)
        allv = [x for v in per.values() for x in v]
        if len(allv) < 8:
            log(f"feedback: only {len(allv)} posts with numbers - not enough yet")
            return {}
        mean = sum(allv) / len(allv) or 1.0
        out = {}
        for c, v in per.items():
            shrink = len(v) / (len(v) + 3.0)  # few posts -> stay near 1.0
            out[c] = max(0.7, min(1.4, 1 + shrink * ((sum(v) / len(v)) / mean - 1)))
        log("feedback multipliers", {c: round(x, 2) for c, x in out.items()})
        return out
    except Exception as e:  # noqa
        log("feedback unavailable:", repr(e)[:200])
        return {}


if __name__ == "__main__":
    for a in sys.argv[1:] or list(RA.BANK):
        if RA.BANK[a].get("done"):
            continue
        try:
            print(a, interest(a))
        except Exception as e:  # noqa
            print(a, "ERR", e)
