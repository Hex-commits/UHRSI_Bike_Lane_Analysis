"""Preview: connecting the coarse CNN detections before edge tracing.

The coarse scan fires reliably on lane *interior* (99% of windows centred on
one, measured over a 319 m run) and correctly refuses windows that straddle a
lane edge, which are half asphalt. Because the scan grid is anchored to the
image and a lane drifts across it, the paint fill of the nearest window
oscillates along the lane -- so detections switch on and off in bands even
where the lane is continuous and the detector has confirmed it either side.

Those gaps are an alignment artifact, not evidence of absence, which is what
makes closing them defensible. This renders the mask before and after a
directional closing at two bridging lengths so the trade can be seen rather
than argued about:

    uv run python -m scripts.diagnostics.preview_connection
"""

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from scipy.ndimage import label

from pipeline.config import COARSE_BRIDGE_M, INPUT_CHUNK_RES_M, PROJECT_ROOT
from scripts.detection.edge_trace import connect_coarse
from scripts.detection.texture_detector import bike_lane_detector
from scripts.diagnostics.generate_pipeline_report import OUTPUT_TILE_PATH, WINDOW
from scripts.measurement.measure_bikelane_gap import LANE_COLOR

OUT_PATH = PROJECT_ROOT / "connection_preview.png"

# Bridging lengths to compare, in metres of ground.
BRIDGE_M = (2.0, COARSE_BRIDGE_M)



def _panel(rgb: np.ndarray, mask: np.ndarray, title: str) -> np.ndarray:
    out = rgb.astype(np.float32) * 0.55
    colour = np.array([int(LANE_COLOR[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float32)
    out[mask] = out[mask] * 0.35 + colour * 0.65
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> None:
    with rasterio.open(OUTPUT_TILE_PATH) as src:
        rgb = np.transpose(src.read([1, 2, 3], window=WINDOW), (1, 2, 0))

    print("running the coarse scan over the report window...", flush=True)
    detections = bike_lane_detector().predict(rgb)
    coarse = detections[0].mask if detections else np.zeros(rgb.shape[:2], bool)
    print(f"  coarse mask: {int(coarse.sum()):,} px, {label(coarse)[1]} components")

    panels = [(_panel(rgb, np.zeros_like(coarse), "imagery"), "prefiltered imagery"),
              (_panel(rgb, coarse, "coarse"),
               f"coarse CNN mask -- {label(coarse)[1]} components")]
    for metres in BRIDGE_M:
        px = max(3, int(round(metres / INPUT_CHUNK_RES_M)))
        closed = connect_coarse(coarse, px)
        n = label(closed)[1]
        grew = (closed.sum() / max(coarse.sum(), 1) - 1) * 100
        print(f"  bridge {metres:.0f} m ({px} px): {int(closed.sum()):,} px "
              f"(+{grew:.0f}%), {n} components")
        panels.append((_panel(rgb, closed, "closed"),
                       f"connected, bridge {metres:.0f} m -- {n} components (+{grew:.0f}% area)"))

    h, w = rgb.shape[:2]
    scale = 0.5
    ph, pw = int(h * scale), int(w * scale)
    label_h = 30
    sheet = Image.new("RGB", (pw * 2 + 12, (ph + label_h) * 2 + 12), "black")
    d = ImageDraw.Draw(sheet)
    for i, (img, caption) in enumerate(panels):
        r, c = divmod(i, 2)
        x, y = c * (pw + 12), r * (ph + label_h + 12)
        sheet.paste(Image.fromarray(img).resize((pw, ph), Image.LANCZOS), (x, y))
        d.text((x + 4, y + ph + 8), caption, fill="white")
    sheet.save(OUT_PATH)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
