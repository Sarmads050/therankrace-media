"""RACE_CFG=cfg.json python _preview_gen.py OUT.png t1 t2 ...  (contact sheet of race frames)"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bar_race_gen as G
from PIL import Image
t0 = time.time(); G.init_worker(60); print("simulate", round(time.time() - t0, 1), "s")
ts = [float(a) for a in sys.argv[2:]]
tiles = [Image.frombytes("RGB", (G.VW, G.VH), G.render_frame(int(round(t * 60)))).resize((540, 960), Image.LANCZOS) for t in ts]
cols = min(4, len(tiles)); rows = (len(tiles) + cols - 1) // cols
sheet = Image.new("RGB", (540 * cols, 960 * rows))
for i, im in enumerate(tiles): sheet.paste(im, ((i % cols) * 540, (i // cols) * 960))
sheet.save(sys.argv[1]); print("saved", len(tiles))
