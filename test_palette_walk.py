"""Bounded correctness/privacy checks using original generated fixtures only."""
import hashlib
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from palette_walk import palette, card, color_words


class PaletteChecks(unittest.TestCase):
    def test_observation_labels_match_green_and_gold_swatches(self):
        self.assertEqual(color_words([80, 115, 73]), "a green shade")
        self.assertEqual(color_words([220, 197, 144]), "a golden shade")

    def test_three_known_colors_and_preserved_source(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "private-name.png"
            colors = np.array([[20, 100, 40], [60, 130, 210], [230, 190, 80]], dtype=np.uint8)
            image = np.empty((96, 96, 3), dtype=np.uint8)
            for number, rgb in enumerate(colors):
                image[:, number * 32:(number + 1) * 32] = rgb
            Image.fromarray(image).save(path)
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            observed = palette(path)
            self.assertEqual(len(observed), 3)
            for rgb in colors:
                self.assertLess(min(np.linalg.norm(rgb.astype(float) - x['rgb']) for x in observed), 4)
            self.assertTrue(all(abs(x['share'] - 1/3) < .025 for x in observed))
            self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())
            report = card(observed, False)
            for forbidden in [path.name, str(path.parent), '<img', 'fetch(', 'XMLHttpRequest', 'https://']:
                self.assertNotIn(forbidden, report)

    def test_monochrome_does_not_invent_three_colors(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "one.png"
            Image.new("RGB", (20, 20), (23, 42, 61)).save(path)
            self.assertEqual(palette(path), [{"rgb": [23, 42, 61], "share": 1.0}])

    def test_transparent_pixels_do_not_form_a_palette(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "transparent.png"
            Image.new("RGBA", (20, 20), (255, 0, 0, 0)).save(path)
            with self.assertRaisesRegex(ValueError, "visible"):
                palette(path)

    def test_oversize_rejected_before_decoder(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "large.png"
            with path.open("wb") as stream:
                stream.truncate(20 * 1024 * 1024 + 1)
            with self.assertRaisesRegex(ValueError, "20 MiB"):
                palette(path)


if __name__ == "__main__":
    unittest.main()
