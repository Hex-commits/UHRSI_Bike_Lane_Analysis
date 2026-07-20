"""Identify bike-lane texture via a frozen, pretrained-on-aerial-imagery CNN's embeddings.

Uses TorchGeo's Swin V2-B backbone pretrained on NAIP aerial RGB
(SatlasPretrain) as a frozen feature extractor -- no training at all. NAIP is
the closest domain match among TorchGeo's options: both are high-res top-down
RGB aerial capture, unlike Sentinel-2 (10 m/px) or Landsat (30 m/px) which
would see a whole bike lane as a sub-pixel smudge.

Plain nearest-neighbor cosine similarity to individual references (an earlier
version) didn't work: any two pavement-like patches stayed similar (0.75-0.95)
almost regardless of class -- two different bike-lane crops came out *less*
similar (0.85) than one was to a road crop (0.87). The embedding's dominant
variance is "generic top-down pavement texture", shared by everything, and
the signal we want (reddish paint vs gray asphalt) is a small component
swamped by raw similarity. The fix: `discriminant_score` projects a candidate
onto the *difference* between the mean bikelane and mean negative embedding,
isolating the component that separates the classes. Still zero training --
arithmetic on frozen embeddings, not gradient descent.

Reference embeddings are extracted once from data/input/textures/<label>/ (one
per file, grouped by subfolder). Folders are named per *surface* ("bikelane",
"road", "sidewalk", "rooftop") not "positive"/"negative", because which counts
as positive depends on the detector: road is a negative for the bike-lane
detector and the positive for the road detector. Each detector names its own
positive and negative labels (config.py's BIKE_LANE_TEXTURE_LABELS /
ROAD_TEXTURE_LABELS).
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchgeo.models import Swin_V2_B_Weights, swin_v2_b

INPUT_SIZE_PX = 256

_model: torch.nn.Module | None = None
_device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def _load_model() -> torch.nn.Module:
    global _model
    if _model is None:
        model = swin_v2_b(weights=Swin_V2_B_Weights.NAIP_RGB_SI_SATLAS)
        model.head = torch.nn.Identity()
        model.eval()
        model = model.to(_device)
        _model = model
    return _model


def _preprocess(image: np.ndarray) -> torch.Tensor:
    """Resize an (H, W, 3) uint8 RGB array to the model's input size and normalize to [0, 1]."""
    resized = np.array(Image.fromarray(image).resize((INPUT_SIZE_PX, INPUT_SIZE_PX), Image.BILINEAR))
    tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0)
    return tensor / 255.0


def embed(image: np.ndarray) -> np.ndarray:
    """Return the frozen backbone's embedding for an (H, W, 3) uint8 RGB image."""
    model = _load_model()
    with torch.no_grad():
        out = model(_preprocess(image).to(_device))
    return out.squeeze(0).cpu().numpy()


def embed_batch(images: list[np.ndarray]) -> np.ndarray:
    """Return the frozen backbone's embeddings for a batch of (H, W, 3) uint8 RGB images.

    One forward pass for the whole batch rather than one per image -- still
    ~28 ms/image on this machine's MPS backend (batching didn't meaningfully
    beat that per-image cost for this model), but avoids Python-level
    per-call overhead when scanning many windows.
    """
    model = _load_model()
    batch = torch.cat([_preprocess(image) for image in images], dim=0).to(_device)
    with torch.no_grad():
        out = model(batch)
    return out.cpu().numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def load_references(textures_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    """Embed every image under `textures_dir`/<label>/*.png, grouped by label.

    `textures_dir` must contain one subfolder per label (e.g. "bikelane",
    "negative"); each subfolder's images become that label's reference
    embeddings, keyed by filename stem.
    """
    references: dict[str, dict[str, np.ndarray]] = {}
    for label_dir in sorted(p for p in textures_dir.iterdir() if p.is_dir()):
        embeddings = {}
        for path in sorted(label_dir.glob("*.png")):
            image = np.array(Image.open(path).convert("RGB"))
            embeddings[path.stem] = embed(image)
        if embeddings:
            references[label_dir.name] = embeddings
    return references


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / (np.linalg.norm(vector) + 1e-8)


def _class_mean(references: dict[str, dict[str, np.ndarray]], labels: Sequence[str]) -> np.ndarray:
    """Mean of the (unit-normalized) embeddings pooled across `labels`.

    Pooled, not an average of per-label means: every crop carries equal
    weight, so a label's influence stays proportional to how many examples it
    has -- rooftops need four crops to cover their color range (see README),
    and per-label means would collapse those to the weight of one sidewalk crop.
    """
    return np.mean([_unit(e) for label in labels for e in references[label].values()], axis=0)


def discriminant_direction(
    references: dict[str, dict[str, np.ndarray]],
    positive_label: str,
    negative_labels: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (direction, midpoint) separating `positive_label` from `negative_labels`.

    `direction` points from the pooled negative mean towards the positive
    class's; `midpoint` is halfway between them. Exposed separately from
    `discriminant_score` so a caller scoring many embeddings (a sliding-window
    scan) can compute it once. `negative_labels` is a set because the same
    folders serve more than one detector: road is a negative when looking for
    bike-lane paint and a positive when looking for road (config.py's
    *_TEXTURE_LABELS).
    """
    positive_mean = _class_mean(references, [positive_label])
    negative_mean = _class_mean(references, negative_labels)
    return positive_mean - negative_mean, (positive_mean + negative_mean) / 2


def discriminant_score(
    embedding: np.ndarray,
    references: dict[str, dict[str, np.ndarray]],
    positive_label: str,
    negative_labels: Sequence[str],
) -> float:
    """Project `embedding` onto the direction separating `positive_label` from `negative_labels`.

    Positive means `embedding` sits on the `positive_label` side of the
    midpoint between the class means; a magnitude of ~1 is about as far from
    the midpoint as the class means themselves. Unlike raw cosine similarity,
    this isolates the component that distinguishes the classes (see module
    docstring).
    """
    direction, midpoint = discriminant_direction(references, positive_label, negative_labels)
    return float(np.dot(_unit(embedding) - midpoint, direction) / (np.linalg.norm(direction) + 1e-8))
