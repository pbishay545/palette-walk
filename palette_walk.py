"""Make a printable color-observation card using local, open-source clustering."""
import argparse
import colorsys
import hashlib
import html
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 20_000_000


def palette(source):
    if source.stat().st_size > MAX_BYTES:
        raise ValueError("Choose an image smaller than 20 MiB.")
    with Image.open(source) as image:
        if image.format not in {"PNG", "JPEG"}:
            raise ValueError("Use a PNG or JPEG image.")
        if image.width * image.height > MAX_PIXELS:
            raise ValueError("Choose an image with at most 20 million pixels.")
        image = ImageOps.exif_transpose(image).convert("RGBA")
        # Sample source colors without creating blended boundary pixels.
        image.thumbnail((64, 64), resample=Image.Resampling.NEAREST)
        rgba = np.asarray(image).reshape(-1, 4)
        pixels = rgba[rgba[:, 3] >= 240, :3].astype(float)
    if not len(pixels):
        raise ValueError("The image needs visible, nearly opaque pixels.")
    clusters = min(3, len(np.unique(pixels, axis=0)))
    with threadpool_limits(limits=1):
        model = KMeans(n_clusters=clusters, random_state=17, n_init=10).fit(pixels)
    counts = np.bincount(model.labels_, minlength=clusters)
    order = sorted(range(clusters), key=lambda i: (-int(counts[i]), tuple(model.cluster_centers_[i])))
    return [{"rgb": [int(x) for x in np.clip(np.rint(model.cluster_centers_[i]), 0, 255)],
             "share": round(int(counts[i]) / len(pixels), 4)} for i in order]


def color_words(rgb):
    hue, saturation, value = colorsys.rgb_to_hsv(*(v / 255 for v in rgb))
    if value < .22:
        return "a dark shade"
    if saturation < .16:
        return "a pale shade" if value > .72 else "a muted shade"
    return ("a reddish shade", "a golden shade", "a green shade", "a blue-green shade",
            "a blue shade", "a purple shade")[int(hue * 6 + .5) % 6]


def card(colors, demo):
    prompts = ["Notice a texture close to this color. Sketch its pattern.",
               "Notice how this color changes between light and shade.",
               "Find a small detail you would usually walk past. Sketch its shape."]
    panels = []
    for number, color in enumerate(colors):
        hex_color = "#" + "".join(f"{v:02x}" for v in color["rgb"])
        panels.append(f'''<section><div class="swatch" style="background:{hex_color}"></div>
        <p class="meta">{html.escape(color_words(color['rgb']))} · {hex_color}</p>
        <h2>{number + 1}. Look closer</h2><p>{prompts[number]}</p>
        <div class="sketch">Sketch or write here</div><label><input type="checkbox"> I noticed something new</label></section>''')
    badge = "Synthetic illustration demo" if demo else "Your local color card"
    return '''<!doctype html><html lang="en"><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1"><title>Palette Walk</title>
    <style>body{margin:0;background:#f6f3e9;color:#24392f;font:17px/1.5 system-ui,sans-serif}
    main{max-width:1080px;margin:auto;padding:36px}h1{font-size:52px;line-height:1.1;margin:12px 0}
    .badge,.meta{font-size:13px;letter-spacing:.04em}.intro{max-width:710px;font-size:20px}
    .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin:30px 0}
    section{background:#fff;border:1px solid #d4d9cf;border-radius:16px;padding:20px}
    .swatch{height:80px;border-radius:8px;border:1px solid #ddd}.sketch{height:150px;border:1px dashed #a7b5a4;
    border-radius:8px;margin:22px 0;color:#73816f;padding:12px;font-size:14px}h2{font-size:23px}
    button{background:#244c36;color:white;border:0;border-radius:8px;padding:12px 18px;font:inherit;cursor:pointer}
    footer{font-size:13px;max-width:900px;border-top:1px solid #ccd3c6;padding-top:18px}
    @media(max-width:700px){main{padding:20px}.grid{grid-template-columns:1fr}h1{font-size:40px}}
    @media print{body{background:white}main{padding:0}button{display:none}section{break-inside:avoid}.grid{gap:10px}}
    </style><main><div class="badge">PALETTE WALK · ''' + badge + '''</div>
    <h1>A little color. A closer look.</h1><p class="intro">Print this card, put your screen away,
    and take a short observation break in an outdoor place you already know.</p>
    <p>Use the colors as loose prompts. Leave plants and wildlife undisturbed; stay on permitted paths.</p>
    <button onclick="window.print()">Print my observation card</button><div class="grid">''' + "".join(panels) + '''</div>
    <footer>Colors are grouped locally with scikit-learn K-means, an open-source machine-learning algorithm.
    The observation prompts are fixed text. This is color clustering, not object, plant or species identification.
    Lighting and the chosen photo change the palette. No map, route or weather assessment is provided.
    This card contains no source photo, filename, EXIF or location. No upload or network call occurs during analysis.
    Demo tested on generated illustrations only; no outdoor user test or customer adoption is claimed.
    Created with AI assistance.</footer></main></html>'''


def demo_image(path):
    # Original geometric fixture, not a photograph or an outdoor field trial.
    image = Image.new("RGB", (384, 256), "#8fbacb")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 128, 383, 255), fill="#507349")
    draw.polygon([(160, 255), (230, 255), (206, 128), (188, 128)], fill="#d7c29a")
    draw.ellipse((40, 28, 82, 70), fill="#f2d26a")
    image.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", type=Path)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--out", type=Path, required=True, help="A new output directory")
    args = parser.parse_args()
    if args.demo == bool(args.image):
        parser.error("Choose either --demo or one local image path.")
    if args.out.exists():
        parser.error("Output directory already exists; choose a new one.")
    args.out.mkdir(parents=True)
    source = args.image
    if args.demo:
        source = args.out / "synthetic-scene.png"
        demo_image(source)
    if source.stat().st_size > MAX_BYTES:
        raise ValueError("Choose an image smaller than 20 MiB.")
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    colors = palette(source)
    (args.out / "card.html").write_text(card(colors, args.demo), encoding="utf-8")
    report = {"algorithm": "scikit-learn KMeans", "synthetic_demo": args.demo,
              "colors": colors, "source_unchanged": before == hashlib.sha256(source.read_bytes()).hexdigest(),
              "analysis_pixels_max": 4096, "network_requests": 0}
    (args.out / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
