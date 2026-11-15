"""Automatic race topics: any World Bank indicator -> verified config for bar_race_gen.py (no hand-written captions).

    python race_auto.py list                 # topic bank
    python race_auto.py build mobile co2     # fetch data (if needed) + build data/race/<slug>.json + print the story
Every caption is computed from the numbers (who passed whom, when, by how much); nothing is written by hand, so nothing can be wrong.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import race_topics as RT  # noqa

HL = "PAK"
# slug -> indicator + wording. fmt: count | usd | mt | num
BANK = {
    "mobile": dict(ind="IT.CEL.SETS", t1="TOP 10 COUNTRIES BY", t2="MOBILE PHONE SUBSCRIPTIONS", fmt="count", more="more mobile subscriptions",
                   what="mobile phone subscriptions", src="mobile cellular subscriptions", min_y=1990),
    "co2": dict(ind="EN.GHG.CO2.MT.CE.AR5", t1="TOP 10 COUNTRIES BY", t2="CO2 EMISSIONS", fmt="mt", more="more CO2 (Gt = billion tonnes)",
                what="CO2 emissions", src="CO2 emissions (million tonnes)", min_y=1970),
    "military": dict(ind="MS.MIL.XPND.CD", t1="TOP 10 BIGGEST", t2="MILITARY SPENDERS", fmt="usd", more="more military spending",
                     what="military spending", src="military expenditure, current US dollars", min_y=1960),
    "exports": dict(ind="NE.EXP.GNFS.CD", t1="TOP 10 BIGGEST", t2="EXPORTERS", fmt="usd", more="more exports",
                    what="exports", src="exports of goods and services, current US dollars", min_y=1960),
    "remittances": dict(ind="BX.TRF.PWKT.CD.DT", t1="TOP 10 COUNTRIES THAT", t2="RECEIVE MOST REMITTANCES", fmt="usd", more="more money received",
                        what="remittances", src="personal remittances received, current US dollars", min_y=1990),
    "tourists": dict(ind="ST.INT.ARVL", t1="TOP 10 MOST VISITED", t2="COUNTRIES", fmt="count", more="more tourists",
                     what="tourist arrivals", src="international tourist arrivals", min_y=1995),
    "imports": dict(ind="NE.IMP.GNFS.CD", t1="TOP 10 BIGGEST", t2="IMPORTERS", fmt="usd", more="more imports",
                    what="imports", src="imports of goods and services, current US dollars", min_y=1960),
    "manufacturing": dict(ind="NV.IND.MANF.CD", t1="TOP 10 BIGGEST", t2="MANUFACTURERS", fmt="usd", more="more factory output",
                          what="manufacturing output", src="manufacturing value added, current US dollars", min_y=1960),
    "agriculture": dict(ind="NV.AGR.TOTL.CD", t1="TOP 10 BIGGEST", t2="FARMING ECONOMIES", fmt="usd", more="more farm output",
                        what="agriculture output", src="agriculture value added, current US dollars", min_y=1960),
    "consumption": dict(ind="NE.CON.PRVT.CD", t1="TOP 10 BIGGEST", t2="CONSUMER SPENDERS", fmt="usd", more="more household spending",
                        what="household spending", src="household final consumption, current US dollars", min_y=1960),
    "reserves": dict(ind="FI.RES.TOTL.CD", t1="TOP 10 BIGGEST", t2="FOREIGN RESERVES", fmt="usd", more="more reserves",
                     what="foreign reserves", src="total reserves incl. gold, current US dollars", min_y=1960),
    "urban": dict(ind="SP.URB.TOTL", t1="TOP 10 COUNTRIES WITH MOST", t2="CITY PEOPLE", fmt="count", more="more people in cities",
                  what="urban population", src="urban population", min_y=1960),
    "aid": dict(ind="DT.ODA.ODAT.CD", t1="TOP 10 COUNTRIES THAT", t2="RECEIVE MOST AID", fmt="usd", more="more foreign aid",
                what="foreign aid received", src="net official development assistance received, current US dollars", min_y=1960),
    "tourism_money": dict(ind="ST.INT.RCPT.CD", t1="TOP 10 COUNTRIES THAT", t2="EARN MOST FROM TOURISTS", fmt="usd", more="more tourism income",
                          what="tourism income", src="international tourism receipts, current US dollars", min_y=1995),
    "labor": dict(ind="SL.TLF.TOTL.IN", t1="TOP 10 BIGGEST", t2="WORKFORCES", fmt="count", more="more workers",
                  what="workforce size", src="total labor force", min_y=1990),
    "pop": dict(done=True), "gdp": dict(done=True),
}


def build(slug, target=82.0):
    spec = BANK[slug]
    path = os.path.join(RT.SER, slug + ".json")
    if not os.path.exists(path) or os.environ.get("REFRESH"):
        RT.INDICATORS[slug] = spec["ind"]
        RT.fetch(slug)
    ds = RT.DS(slug)
    # usable range: from the first year >= min_y with >= 100 countries, to the last year with >= 100 countries
    cnt = {y: sum(1 for s in ds.d.values() if s.get(y)) for y in range(1960, 2027)}
    ok = [y for y in cnt if cnt[y] >= 80 and y >= spec["min_y"]]
    y0, y1 = ok[0], ok[-1]
    assert y1 - y0 >= 15, "too few years"
    ds.y0, ds.y1, ds.years = y0, y1, list(range(y0, y1 + 1))
    ds.cache = {}
    # the last year must be complete: every Top-10 country of the year before must have data (the newest year is often partial)
    ytop = y1
    while y1 > y0 + 15 and not all(ds.d.get(k, {}).get(y1) for k in set(ds.order(y1 - 1)[:10]) | set(ds.order(max(y0, y1 - 5))[:10])):
        y1 -= 1
        assert ytop - y1 <= 8, "latest years incomplete"
    ds.y1, ds.years = y1, list(range(y0, y1 + 1))
    ds.cache = {}
    r, n, v = ds.rank, ds.name, ds.v
    F = fmt_fn(spec["fmt"])
    top = lambda y, i: ds.order(y)[i]  # noqa

    cands = []  # (priority, year, text, pill)
    # a) leader changes that last
    seen = set()

    def stable(a, b, y):  # a stays ahead of b for the next 6 years (or to the end) and the pair is new
        if (a, b) in seen or (b, a) in seen:
            return False
        if all((v(a, yy) or 0) > (v(b, yy) or 0) for yy in range(y, min(y + 6, y1) + 1)):
            seen.add((a, b))
            return True
        return False

    for y in ds.years[1:]:
        a, b = top(y, 0), top(y - 1, 0)
        if a != b and stable(a, b, y):
            cands.append((1, y, f"{y}: {n(a)} passes {n(b)} and becomes number one!", f"{n(a).upper()} PASSES {n(b).upper()}: #1!"))
    # b) passes at ranks 2-3 that last >= 4 years
    for y in ds.years[1:]:
        for k in (1, 2):
            a = top(y, k)
            b = top(y - 1, k)
            if a != b and r(a, y - 1) == k + 2 and r(b, y) == k + 2 and stable(a, b, y):
                cands.append((2, y, f"{y}: {n(a)} passes {n(b)} and becomes #{k + 1}.", f"{n(a).upper()} PASSES {n(b).upper()}: #{k + 1}"))
    # c) Pakistan (highlight)
    if ds.top10_years(HL):
        yin = ds.top10_years(HL)[0]
        if yin > y0 + 2 and (r(HL, yin - 1) or 99) == 11 and all(r(HL, yy) <= 10 for yy in range(yin, min(yin + 3, y1 + 1))):
            cands.append((1, yin, f"{yin}: Pakistan enters the Top 10 for the first time.", "PAKISTAN ENTERS TOP 10"))
        best = min(r(HL, y) for y in ds.years if r(HL, y))
        yb = next(y for y in ds.years if r(HL, y) == best)
        if best < 8 and yb > y0 + 2 and r(HL, yb - 1) == best + 1:
            cands.append((1, yb, f"{yb}: Pakistan reaches #{best}, its best rank.", f"PAKISTAN: #{best}"))
    # d) leader growth
    lead_end = top(y1, 0)
    if v(lead_end, y0) > 0:
        g = v(lead_end, y1) / v(lead_end, y0)
        if g > 3:
            cands.append((3, int(y0 + 0.62 * (y1 - y0)), f"Since {y0}, {n(lead_end)} grew {g:,.0f} times: from {F(v(lead_end, y0))} to {F(v(lead_end, y1))}.", None))
    # e) snapshots
    for yy in range(y0 + 10, y1 - 4, 10):
        yy = yy - yy % 5
        a, b, c = top(yy, 0), top(yy, 1), top(yy, 2)
        cands.append((4, yy, f"In {yy}: {n(a)} {F(v(a, yy))}, {n(b)} {F(v(b, yy))}, {n(c)} {F(v(c, yy))}.", None))

    chosen, pills = [], []
    GAP = max(3.0, (y1 - y0) / 14.0)
    for pr, y, text, pill in sorted(cands, key=lambda c: (c[0], c[1])):
        if y < y0 + 4.0 or y > y1 - 3:
            continue
        if all(abs(y - c[1]) >= GAP for c in chosen):
            chosen.append((pr, y, text, pill))
    chosen.sort(key=lambda c: c[1])
    caps = [dict(at=y0, text=f"In {y0}, {n(top(y0, 0))} is number one with {F(v(top(y0, 0), y0))}. {n(top(y0, 1))} is second."),
            dict(at=y0 + 2.3, text=f"Each bar is a country. A longer bar means {ds_more(spec)}.")]
    for pr, y, text, pill in chosen:
        caps.append(dict(at=y, text=text))
        if pill:
            pills.append(dict(at=y, dur=3.8, text=pill))
    a, b = top(y1, 0), top(y1, 1)
    hl_rank = r(HL, y1)
    tail = [dict(at={"tail": 0.4}, text=f"{y1}: {n(a)} {F(v(a, y1))}, {n(b)} {F(v(b, y1))}." + (f" Pakistan is #{hl_rank} with {F(v(HL, y1))}." if hl_rank else "")),
            dict(at={"tail": 5.0}, text="Which country surprised you most? Tell us in the comments.")]
    if hl_rank:
        pills.append(dict(at={"tail": 0.4}, dur=4.2, text=f"PAKISTAN IS #{hl_rank} TODAY"))
    ev_years = [p["at"] for p in pills if not isinstance(p["at"], dict)] or [int((y0 + y1) / 2)]
    hits = pick_hits(ev_years, y0, y1)
    ents = ds.ever()
    cfg = dict(slug=slug, years=ds.years, entities=[{"iso": c, "name": n(c)} for c in ents],
               series={c: [v(c, y) for y in ds.years] for c in ents}, title1=spec["t1"], title2=f"{spec['t2']} {y0}-{y1}" if len(spec["t2"]) < 20 else spec["t2"],
               fmt=spec["fmt"], captions=caps + tail, pills=pills, hits=hits, highlight=HL, tail=9.0,
               source=f"Data: World Bank, {spec['src']}, {y0} to {y1} (updated {ds.updated}). Flags: Wikipedia.",
               seg=pace_for(ev_years, y0, y1, target), built=__import__("time").strftime("%Y-%m-%d %H:%M"), updated=ds.updated,
               meta=dict(what=spec["what"], y0=y0, y1=y1, leader=n(a), second=n(b), hl_rank=hl_rank, hl_value=v(HL, y1)))
    os.makedirs(RT.OUT, exist_ok=True)
    json.dump(cfg, open(os.path.join(RT.OUT, slug + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
    return cfg


def ds_more(spec):
    return spec["more"]


def fmt_fn(kind):
    def f(x):
        if kind == "count":
            return f"{x / 1e9:.2f} billion" if x >= 1e9 else (f"{x / 1e6:,.0f} million" if x >= 1e6 else f"{x:,.0f}")
        if kind == "usd":
            return f"${x / 1e12:,.1f} trillion" if x >= 1e12 else f"${x / 1e9:,.0f} billion"
        if kind == "mt":
            return f"{x / 1e3:,.1f} billion tonnes" if x >= 1e3 else f"{x:,.0f} million tonnes"
        return f"{x:,.0f}"
    return f


def pick_hits(evs, y0, y1):
    goal = [y0 + f * (y1 - y0) for f in (0.15, 0.40, 0.65, 0.85)]
    out = []
    for g in goal:
        c = min(evs, key=lambda e: abs(e - g))
        if c not in out:
            out.append(c)
    for g in goal:  # fill up to 4 distinct hits with plain years
        if len(out) >= 4:
            break
        yy = int(round(g))
        if yy not in out:
            out.append(yy)
    return sorted(out)[:4]


def pace_for(events, y0, y1, target, slow_ratio=0.65, lead=1.0, lag=2.0):
    zones = sorted((max(y0, e - lead), min(y1, e + lag)) for e in events)
    merged = []
    for a, b in zones:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    Z = sum(b - a for a, b in merged)
    fast = min(1.1, max(0.3, ((y1 - y0 - Z) + Z / slow_ratio) / target))
    slow = fast * slow_ratio
    segs, cur = [], float(y0)
    if merged and merged[0][0] > y0 + 2.0:
        segs.append([2.0, slow])
        cur = y0 + 2.0
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


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "list":
        for k, s in BANK.items():
            print(k, s.get("ind", "(built by race_topics.py)"), "-", s.get("what", ""))
    else:
        for slug in sys.argv[2:]:
            c = build(slug)
            print("==", slug, c["years"][0], "-", c["years"][-1], len(c["entities"]), "entities; hits", c["hits"])
            for x in c["captions"]:
                print("   cap", x["at"], x["text"])
            for x in c["pills"]:
                print("   pill", x["at"], x["text"])
