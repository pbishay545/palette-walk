# Palette Walk

An original local prototype: group a picture's colors with open-source machine learning, then make a printable outdoor observation card. The prompts invite noticing texture, light and small details; put the screen away after printing.

The browser demo is a **precomputed card from an original synthetic illustration**, not browser inference or a field trial. Run the Python tool to analyze a PNG/JPEG you are authorized to use. It makes no network requests during analysis. Initial dependency installation needs internet access. No source photo, filename, EXIF or location is included in the generated card or JSON.

## Run locally

Tested on Python 3.12 and Windows. Create an isolated environment and install the pinned dependencies:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m unittest -v test_palette_walk.py
.venv/Scripts/python.exe palette_walk.py --demo --out first-card
```

On macOS/Linux, use `.venv/bin/python` instead. Open `first-card/card.html` in a browser. The print button opens your browser's print dialog; no printer or PDF output has been tested.

For a local picture you have permission to process:

```text
python palette_walk.py your-picture.png --out another-new-folder
```

Use the environment's Python executable. Output folders must be new; existing output is never overwritten by the CLI. Inputs remain unchanged. Keep real-photo outputs private unless you choose to share them; generated palettes can still reflect the picture's content.

## What the machine learning does

scikit-learn K-means fits up to three clusters to at most 4,096 sampled opaque RGB pixels. Nearest-neighbor sampling avoids manufacturing blended boundary colors; a fixed seed and ten initializations make this bounded example reproducible. Monochrome inputs produce one swatch rather than three invented colors. Fixed observation prompts are arranged beside the clusters.

This is unsupervised machine learning, **not a generative model**, object detector or plant/species identifier. It does not assess routes, weather or outdoor safety. RGB distance is not perceptually uniform; sampling can miss small details, and lighting changes the palette. Color words are approximate; use the swatches as loose prompts. PNG/JPEG only; 20 MiB and 20-million-pixel limits, no fully transparent inputs. No serialized model loading.

Five bounded checks pass: three known colors and original preservation/privacy of the card, monochrome input, transparent input rejection, pre-decoder byte limit, and green/gold labels. Only generated fixtures were used. Real photos, printing, mobile layout and outdoor usability remain untested. No adoption, certification or earnings claimed.

## Provenance and proposed license

Created with AI assistance on October 11, 2026 for a bounded “Touch Grass” challenge experiment. Original code, fixture and card design. Original-code license: MIT; see LICENSE. Libraries are installed separately, not vendored: scikit-learn BSD-3-Clause, NumPy/SciPy BSD, Pillow MIT-CMU, and the licenses of the pinned dependencies remain applicable.

References: [scikit-learn clustering documentation](https://scikit-learn.org/stable/modules/clustering.html#k-means), [scikit-learn license](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING), and [DEV challenge](https://dev.to/challenges/hacktoberfest-week1-2026-10-05). The official color-quantization example motivated the reusable clustering procedure; its page was not retrievable in this session, so no source code or image was copied. This is a separate original implementation.

