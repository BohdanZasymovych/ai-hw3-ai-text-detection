"""Render the report tables (assets/tables/*.csv) as styled images, like in the notebooks: python scripts/report_tables.py"""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageChops

TABLES = Path(__file__).resolve().parent.parent / "assets" / "tables"
CHROME = "google-chrome"
CSS = """<style>body{margin:0;padding:8px;background:#fff;font-family:DejaVu Sans,Arial,sans-serif;font-size:13px}
table{border-collapse:collapse}th,td{padding:4px 10px;border:1px solid #ddd;text-align:right}
th{background:#f5f5f5}th[scope=row],th.row_heading{text-align:left}</style>"""


def read(name):
    return pd.read_csv(next(TABLES.glob(f"{name}_*.csv")), dtype=str, keep_default_na=False)


def num(s):
    return pd.to_numeric(s.str.replace(r"[%,]", "", regex=True), errors="coerce")


def styled(n):
    df = read(f"table{n}")
    if n in (1, 2):
        vals = df.set_index(["lang", "domain"]).apply(num)
        if n == 1:
            return vals.style.background_gradient(cmap="Reds", axis=None).format("{:.0f}%")
        return vals.style.background_gradient(cmap="Blues", axis=None, gmap=np.log10(vals)).format("{:,.0f}")
    if n == 3:
        df = df.assign(C=num(df["C"]), **{"macro-F1": num(df["macro-F1"])})
        return df.style.hide(axis="index").set_properties(subset=["features", "model"], **{"text-align": "left"}).background_gradient(subset="macro-F1", cmap="Blues").format(
            {"C": "{:g}", "macro-F1": "{:.3f}"})
    df = df.set_index(df.columns[0])
    if n == 7:
        df = df.apply(num)
        best = df["macro-F1"].idxmax()
        return (df.style.background_gradient(subset="macro-F1", cmap="Blues")
                .format({"n features": "{:.0f}", **{c: "{:.4f}" for c in df.columns[1:]}})
                .apply(lambda r: ["font-weight:bold" if r.name == best else "" for _ in r], axis=1))
    if n == 4:  # support column is a count, not scored
        scores = df.drop(columns="support").apply(num)
        return scores.assign(support=num(df["support"])).style.background_gradient(
            subset=list(scores), cmap="Blues", axis=None).format({**{c: "{:.3f}" for c in scores}, "support": "{:,.0f}"})
    scores = [c for c in df.columns if c != "public LB"]
    df = df.apply(num) if n != 8 else df.assign(**{c: num(df[c]) for c in scores})
    sty = df.style.background_gradient(subset=scores, cmap="Blues", axis=None).format("{:.3f}", subset=scores)
    return sty if n != 8 else sty.format(lambda v: v if v == "-" else f"{float(v):.3f}", subset=["public LB"])


def render(n):
    sty = styled(n)
    sty.index.names = [None] * sty.index.nlevels  # no extra index-name header row
    html = CSS + sty.to_html()
    with tempfile.TemporaryDirectory() as tmp:
        page, shot = Path(tmp) / "t.html", Path(tmp) / "t.png"
        page.write_text(html)
        subprocess.run([CHROME, "--headless", "--no-sandbox", "--hide-scrollbars", "--force-device-scale-factor=2",
                        f"--screenshot={shot}", "--window-size=1200,1400", f"file://{page}"], check=True,
                       capture_output=True)
        img = Image.open(shot).convert("RGB")
    box = ImageChops.difference(img, Image.new("RGB", img.size, "white")).getbbox()
    pad = 8
    img.crop((max(box[0] - pad, 0), max(box[1] - pad, 0), box[2] + pad, box[3] + pad)).save(
        next(TABLES.glob(f"table{n}_*.csv")).with_suffix(".png"))


if __name__ == "__main__":
    for n in range(1, 9):
        render(n)
