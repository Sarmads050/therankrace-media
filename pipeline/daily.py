"""Claude-free daily job (runs on GitHub Actions, no AI, no PC needed).

    python pipeline/daily.py --slot 1|2 [--dry-run] [--date YYYY-MM-DD]

topic (rotation) -> World Bank data -> computed captions -> flags -> render 1080x1920 60fps -> GitHub release -> Buffer schedule.
Env: GH_TOKEN (gh CLI), BUFFER_API_KEY, BUFFER_CHANNEL_ID (optional; else the first Instagram channel is used).
"""
import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa
        pass
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.chdir(ROOT)
import race_auto as RA  # noqa
import race_topics as RT  # noqa
import flags as FL  # noqa
import pick as PK  # noqa
from common import FFMPEG  # noqa

REPO = os.environ.get("GITHUB_REPOSITORY", "Sarmads050/therankrace-media")
PKT = dt.timezone(dt.timedelta(hours=5))
EPOCH = dt.date(2026, 11, 16)  # first cloud day
ORDER = [s for s in RA.BANK if s not in ("tourists", "tourism_money")]  # every topic; each is used exactly once (see used_topics)
PHRASE = {"pop": "population", "gdp": "economy size"}
TAGS = {"pop": "#population #worldpopulation", "gdp": "#gdp #economy #worldeconomy", "mobile": "#mobile #smartphones #technology",
        "co2": "#climate #co2 #environment", "military": "#military #defense #geopolitics", "exports": "#exports #trade #economy",
        "imports": "#imports #trade #economy", "remittances": "#remittances #money #economy", 
        "tourism_money": "#travel #tourism #money", "manufacturing": "#manufacturing #industry #economy", "agriculture": "#agriculture #farming #food",
        "consumption": "#spending #consumers #economy", "reserves": "#finance #reserves #economy", "urban": "#cities #urbanization #population",
        "aid": "#aid #development #economy", "labor": "#jobs #workforce #economy"}
CAT_TAGS = {"military": "#military #defense", "economy": "#economy #gdp", "finance": "#finance #money", "people": "#population #demographics",
            "trade": "#trade #exports", "tech": "#technology #tech", "climate": "#climate #environment", "industry": "#industry #business",
            "health": "#health #healthcare", "transport": "#transport #travel", "food": "#food #agriculture"}
SLOT_HOUR = {1: 18, 2: 21}
# "countdown" = #10 -> #1 reveal (~27 s, big readable cards); "race" = the old 43 s bar race. Switch with the env var REEL_FORMAT.
REEL_FORMAT = os.environ.get("REEL_FORMAT", "race")  # owner (2026-11-21) wants the bar-race format back
META = {}


def log(*a):
    print(*a, flush=True)


def run(cmd, **kw):
    log("$", " ".join(cmd) if isinstance(cmd, list) else cmd)
    return subprocess.run(cmd, check=True, **kw)


def ranks(cfg, year_idx=-1):
    vals = {e["iso"]: cfg["series"][e["iso"]][year_idx] for e in cfg["entities"] if cfg["series"][e["iso"]][year_idx] is not None}
    return sorted(vals, key=lambda k: -vals[k]), vals


def make_cfg(slug):
    os.environ["REFRESH"] = "1" if os.environ.get("CI") else os.environ.get("REFRESH", "")
    if slug in RT.BUILD:
        RT.fetch(slug) if os.environ.get("CI") else None
        cfg = RT.BUILD[slug]()
        cfg.setdefault("meta", {})["what"] = PHRASE[slug]
    else:
        cfg = RA.build(slug)
    if cfg["years"][-1] < dt.datetime.now(PKT).year - 3:
        raise RuntimeError(f"data too old: ends {cfg['years'][-1]}")
    return cfg


def ensure_flags(cfg, slug):
    names = json.load(open(os.path.join(RT.SER, slug + ".json")))["names"]
    missing = []
    for e in cfg["entities"]:
        iso = e["iso"]
        if os.path.exists(os.path.join(FL.DIR, iso + ".png")):
            continue
        r = FL.fetch(iso, names.get(iso))
        log("flag", iso, r)
        if not os.path.exists(os.path.join(FL.DIR, iso + ".png")):
            missing.append(iso)
    # a missing flag only matters if the country spends a long time in the Top 10
    bad = []
    for iso in missing:
        ser = cfg["series"][iso]
        n = 0
        for i in range(len(cfg["years"])):
            row = sorted((cfg["series"][e["iso"]][i] or 0 for e in cfg["entities"]), reverse=True)
            if (ser[i] or 0) >= row[9] and ser[i]:
                n += 1
        if n > 3:
            bad.append((iso, n))
    if bad:
        raise RuntimeError(f"flags missing for {bad}")


def render(slug, cfg, date, out):
    cfgp = os.path.join("data", "race", slug + ".json")
    assert os.path.exists(cfgp)
    env = dict(os.environ, RACE_CFG=cfgp)
    os.makedirs("work", exist_ok=True)
    script = "countdown_gen.py" if REEL_FORMAT == "countdown" else "bar_race_gen.py"
    run([sys.executable, "-I", os.path.join("pipeline", script), out, os.path.join("work", slug + "_race"), "--fps", "60"], env=env)
    try:
        META[slug] = json.load(open(out + ".meta.json"))
    except Exception:  # noqa
        META[slug] = {}
    p = subprocess.run([FFMPEG, "-hide_banner", "-i", out, "-af", "volumedetect", "-f", "null", "-"], capture_output=True, text=True)
    t = p.stderr
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", t)
    dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    mv = re.search(r"mean_volume: (-?[\d.]+) dB", t)
    assert "1080x1920" in t and " 60 fps" in t, "bad video format"
    assert 90 <= dur <= 119.5, f"bad duration {dur}"  # owner's rule: every video 90-119 s, f"bad duration {dur}"
    assert mv and -24 <= float(mv.group(1)) <= -9, f"bad audio level {mv and mv.group(1)}"
    log(f"verified {out}: {dur:.1f}s mean {mv.group(1)} dB")
    return dur


def caption(slug, cfg):
    spec = RA.BANK.get(slug, {})
    what = cfg.get("meta", {}).get("what") or spec.get("what") or PHRASE.get(slug, "ranking")
    order, vals = ranks(cfg)
    nm = {e["iso"]: e["name"] for e in cfg["entities"]}
    y0, y1 = cfg["years"][0], cfg["years"][-1]
    first, second = nm[order[0]], nm[order[1]]
    hl = cfg.get("highlight", "PAK")
    pr = order.index(hl) + 1 if hl in order else None
    hooks = []
    if pr and pr <= 10:
        hooks.append(f"Pakistan is number {pr} in the world for {what}! 🇵🇰")
    elif pr:
        hooks.append(f"Where is Pakistan in the world for {what}? Watch {y0} to {y1} 🇵🇰")
    else:
        hooks.append(f"Can you guess the No.1 country for {what}? Watch {y0} to {y1} 👀")
    hook = hooks[0]
    title = (cfg["title1"] + " " + cfg["title2"]).title().replace("Co2", "CO2").replace("Gdp", "GDP")
    # Instagram SEO: the first line (what people see before "more") is the search keyword + a hook; 5-6 focused hashtags (not 10);
    # a pointed question (comments), a save/share ask and a follow ask (the signals Instagram uses to suggest a Reel to new people)
    qs = [f"Which country will be No.1 in {y1 + 15}? Comment your guess 👇",
          f"Which country surprised you most? Tell us in the comments 👇",
          f"Is this what you expected? Comment your country and its rank 👇"]
    q = qs[(int(time.time() // 86400) + len(slug)) % len(qs)]
    ur = "\nPakistan ka number kya hai? Comment mein batayein 🇵🇰" if pr else ""
    topic_tags = TAGS.get(slug) or CAT_TAGS.get(PK.cat(slug), "#economy #data")
    ctags = ["#pakistan", "#countries"]  # no leader hashtag: it would spoil the ending
    tags = ["#top10"] + topic_tags.split()[:2] + ctags + ["#barchartrace", "#worldranking"]
    src = cfg["source"].split(" (updated")[0]
    if REEL_FORMAT == "countdown":
        # a countdown must not give away the answer: no #1 in the caption and no leader hashtag
        pk = cfg.get("meta", {}).get("hl_rank")
        if pk and pk <= 10:
            hook = f"Pakistan is in the Top 10 for {what}! Can you guess its rank? 🇵🇰"
        elif pk:
            hook = f"Top 10 countries by {what}: where is Pakistan? Can you guess #1? 👀"
        else:
            hook = f"Top 10 countries by {what}: can you guess #1? 👀"
        body = re.sub(r"\s*\d{4}-\d{4}\s*$", "", title).strip()
        qs = ["Comment your guess for #1 before the end 👇", "Was Pakistan higher or lower than you thought? Tell us 👇",
              "Which country surprised you most? Comment below 👇"]
        q = qs[(int(time.time() // 86400) + len(slug)) % len(qs)]
        tags = ["#top10"] + topic_tags.split()[:2] + ["#pakistan", "#countries", "#worldranking", "#facts"]
        tags = " ".join(dict.fromkeys(tags))
        return (f"{hook}\n\n{body}: the {y1} ranking, countdown from #10 to #1.\n\n{q}{ur}\n\n"
                f"🔖 Save this and send it to a friend.\nFollow @therankrace for a new Top 10 world ranking every day.\n\n{src}\n\n{tags}")
    tags = " ".join(dict.fromkeys(tags))
    return (f"{hook}\n\n{title}: the full race, {y0} to {y1}. Watch till the end.\n\n{q}{ur}\n\n"
            f"🔖 Save this and send it to a friend.\nFollow @therankrace for a new Top 10 world ranking every day.\n\n{src}\n\n{tags}")


def upload(slug, date, video):
    tag = f"{slug}-race-{date}"
    fn = f"{slug}-{date}.mp4"
    tmp = os.path.join("work", fn)
    if os.path.abspath(video) != os.path.abspath(tmp):
        import shutil
        shutil.copyfile(video, tmp)
    r = subprocess.run(["gh", "release", "view", tag, "-R", REPO], capture_output=True)
    if r.returncode != 0:
        run(["gh", "release", "create", tag, tmp, "-R", REPO, "-t", f"{slug} race {date}", "-n", "Auto-generated Top 10 race"])
    url = f"https://github.com/{REPO}/releases/download/{tag}/{fn}"
    for _ in range(20):
        c = subprocess.run(["curl", "-sIL", "-o", os.devnull, "-w", "%{http_code}", url], capture_output=True, text=True).stdout.strip()
        log("url", c)
        if c == "200":
            return url
        time.sleep(15)
    raise RuntimeError("release url never went live")


def gql(query):
    req = urllib.request.Request("https://api.buffer.com", data=json.dumps({"query": query}).encode(),
                                 headers={"Authorization": "Bearer " + os.environ["BUFFER_API_KEY"], "Content-Type": "application/json",
                                          "User-Agent": "rankrace-daily/1.0"})
    for i in range(4):
        try:
            return json.loads(urllib.request.urlopen(req, timeout=120).read())
        except Exception as e:  # noqa
            if i == 3:
                raise
            log("buffer retry", e)
            time.sleep(10)


def channel_id():
    o = gql("query { account { organizations { id name } } }")["data"]["account"]["organizations"]
    if os.environ.get("BUFFER_CHANNEL_ID"):
        return os.environ["BUFFER_CHANNEL_ID"], o[0]["id"]
    for org in o:
        ch = gql('query { channels(input: {organizationId: "%s"}) { id service name } }' % org["id"])["data"]["channels"]
        for c in ch:
            if c["service"] == "instagram":
                return c["id"], org["id"]
    raise RuntimeError("no Instagram channel in Buffer")


def schedule(text, url, due, thumb_ms=None, title=None):
    ch, org = channel_id()
    iso = due.isoformat()
    try:  # re-run guard: never schedule two posts at the same minute
        r = gql('query { posts(first: 50, input: {organizationId: "%s", filter: {status: [scheduled], channelIds: ["%s"]}}) '
                '{ edges { node { id dueAt } } } }' % (org, ch))
        for e in r["data"]["posts"]["edges"]:
            d0 = dt.datetime.fromisoformat(e["node"]["dueAt"].replace("Z", "+00:00"))
            if abs((d0 - due).total_seconds()) < 90:
                log("already scheduled at this time, skipping", e["node"]["id"])
                return {"id": e["node"]["id"], "skipped": True}
    except Exception as ex:  # noqa
        log("duplicate check failed (continuing):", repr(ex))
    # first try with a chosen cover frame (the finished ranking, not the empty first frame) and a video title; if Buffer rejects
    # that, schedule the plain post so a cosmetic option can never cost us the day's Reel
    metas = []
    if thumb_ms is not None:
        metas.append(", metadata: { thumbnailOffset: %d%s }" % (thumb_ms, (", title: %s" % json.dumps(title)) if title else ""))
    metas.append("")
    r, d = None, {}
    for m in metas:
        q = ('mutation { createPost(input: { text: %s, channelId: "%s", schedulingType: automatic, mode: customScheduled, dueAt: "%s", '
             'assets: [{ video: { url: "%s"%s } }], metadata: { instagram: { type: reel, shouldShareToFeed: true } } }) '
             '{ ... on PostActionSuccess { post { id status dueAt } } ... on MutationError { message } } }') % (json.dumps(text), ch, iso, url, m)
        r = gql(q)
        log("buffer:", json.dumps(r)[:600])
        d = (r.get("data") or {}).get("createPost") or {}
        if "post" in d:
            break
        log("createPost refused with options, retrying plain:", m)
    if "post" not in d:
        raise RuntimeError(f"Buffer refused: {r}")
    return d["post"]


def used_topics():
    """A topic is used once a release '<slug>-race-<date>' exists (the release is deleted again if scheduling fails)."""
    out = subprocess.run(["gh", "release", "list", "-R", REPO, "-L", "1000", "--json", "tagName", "--jq", ".[].tagName"],
                         capture_output=True, text=True, check=True).stdout.split()  # newest first
    recent = []
    for t in out:
        m = re.match(r"^(.+)-race-\d{4}-\d{2}-\d{2}$", t)
        if m and m.group(1) not in recent:
            recent.append(m.group(1))
    return recent


def post_near(due):
    """id of a scheduled Buffer post within 90 s of `due`, else None"""
    ch, org = channel_id()
    r = gql('query { posts(first: 100, input: {organizationId: "%s", filter: {status: [scheduled], channelIds: ["%s"]}}) '
            '{ edges { node { id dueAt } } } }' % (org, ch))
    for e in r["data"]["posts"]["edges"]:
        d0 = dt.datetime.fromisoformat(e["node"]["dueAt"].replace("Z", "+00:00"))
        if abs((d0 - due).total_seconds()) < 90:
            return e["node"]["id"]
    return None


def due_for(today, slot):
    now = dt.datetime.now(PKT)
    due = dt.datetime.combine(today, dt.time(SLOT_HOUR[slot], 0), PKT)
    if due < now + dt.timedelta(minutes=20):  # late run: next day, same hour
        due += dt.timedelta(days=1)
    return due


def summary(md):
    p = os.environ.get("GITHUB_STEP_SUMMARY")
    if p:
        with open(p, "a", encoding="utf-8") as f:
            f.write(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", type=int, required=True, choices=[1, 2])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--date")
    ap.add_argument("--topic", help="force a topic slug (testing)")
    a = ap.parse_args()
    if a.topic == "__probe__":  # print the Buffer API schema pieces the feedback loop needs (debug only)
        if a.slot == 1:
            def named(t):
                while t and not t.get("name"):
                    t = t.get("ofType")
                return t and t["name"]
            fs = gql('{ __type(name: "Post") { fields { name type { name kind ofType { name kind ofType { name kind ofType { name kind } } } } } } }')["data"]["__type"]["fields"]
            mname = named(next(f for f in fs if f["name"] == "metrics")["type"])
            log("PROBE metrics element type:", mname)
            log("PROBE", json.dumps(gql('{ __type(name: "%s") { kind fields { name type { name kind ofType { name kind ofType { name kind } } } } enumValues { name } } }' % mname))[:3500])
            for q in ('{ aggregatedPostMetrics: __type(name: "AggregatedPostMetrics") { fields { name } } }',):
                log("PROBE", json.dumps(gql(q))[:1500])
        return
    today = dt.datetime.now(PKT).date() if not a.date else dt.date.fromisoformat(a.date)
    due = due_for(today, a.slot)
    if not a.dry_run and not a.topic:  # this slot is already filled (e.g. a manual test run): nothing to do, nothing burnt
        pid = post_near(due)
        if pid:
            log(f"slot {a.slot}: a post already exists at {due} ({pid}); nothing to do")
            summary(f"### Slot {a.slot}: already scheduled for {due:%Y-%m-%d %H:%M}, skipped\n\n")
            return
    # NEVER REPEAT: every topic is used exactly once. The two slots take alternate topics of the unused list.
    recent = used_topics()
    unused = [t for t in ORDER if t not in recent]
    log(f"unused topics left: {len(unused)}")
    if not unused and not a.topic:
        raise SystemExit("NO TOPICS LEFT: every topic has been used once. Add new topics to ORDER/BANK.")
    if len(unused) < 20:
        msg = f"only {len(unused)} unused topics left (about {len(unused) // 2} days) - add new topics soon"
        print(f"::warning::{msg}")
        summary(f"**Warning: {msg}**\n\n")
    if a.topic:
        lst = a.topic.split(",")  # "a,b": slot 1 makes a, slot 2 makes b
        tries = [lst[(a.slot - 1) % len(lst)]]
    else:  # CHOOSE: score every unused topic for viewer interest, best two (different categories) go out today
        fb = {}
        if os.environ.get("BUFFER_API_KEY"):
            try:
                ch, org = channel_id()
                fb = PK.feedback(gql, org, ch, log)
            except Exception as e:  # noqa
                log("feedback skipped:", repr(e)[:150])
        rows = PK.rank_topics(unused, recent, fb)
        l1, l2 = PK.split_slots(rows)
        tries = (l1 if a.slot == 1 else l2)[:8]
        table = "\n".join(f"| {i + 1} | {s} | {sc} | {why} |" for i, (s, sc, why) in enumerate(rows[:12]))
        log("topic ranking (best first):\n" + "\n".join(f"  {s:14s} {sc:6.1f}  {why}" for s, sc, why in rows[:12]))
        if a.slot == 1:
            summary(f"#### Why these topics? ranking of the {len(rows)} unused topics\n\n| # | topic | score | reasons |\n|---|---|---|---|\n{table}\n\n")
    errors = []
    for slug in tries:
        try:
            log(f"=== slot {a.slot} topic {slug} date {today}")
            cfg = make_cfg(slug)
            ensure_flags(cfg, slug)
            out = os.path.join("videos", f"{slug}_{today}.mp4")
            os.makedirs("videos", exist_ok=True)
            dur = render(slug, cfg, str(today), out)
            break
        except Exception as e:  # noqa
            log("TOPIC FAILED", slug, repr(e))
            errors.append(f"{slug}: {e!r}")
            if a.topic:
                raise
    else:
        raise SystemExit("all topics failed: " + "; ".join(errors))
    text = caption(slug, cfg)
    log("--- caption ---\n" + text)
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a", encoding="utf-8") as f:
            f.write(f"### Slot {a.slot}: {slug} ({dur:.1f}s)\n\n```\n{text}\n```\n\nfailed topics before it: {errors or 'none'}\n")
    open(os.path.join("work", f"caption_slot{a.slot}.txt"), "w", encoding="utf-8").write(text)
    if a.dry_run:
        log("dry run: not uploading/scheduling")
        return
    url = upload(slug, str(today), out)
    try:
        thumb = META.get(slug, {}).get("cover_ms") or max(0, int((dur - 5.3) * 1000))
        post = schedule(text, url, due, thumb_ms=thumb, title=(cfg["title1"] + " " + re.sub(r"\s*\d{4}-\d{4}$", "", cfg["title2"])).title()[:90])
        if post.get("skipped"):
            raise RuntimeError("slot taken meanwhile")
    except Exception:
        # not scheduled -> release the topic again so it is not burnt
        subprocess.run(["gh", "release", "delete", f"{slug}-race-{today}", "-R", REPO, "--cleanup-tag", "-y"])
        raise
    log("SCHEDULED", post)


if __name__ == "__main__":
    main()
