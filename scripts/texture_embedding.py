"""Identify bike-lane texture via a frozen, pretrained-on-aerial-imagery CNN's embeddings.

Uses TorchGeo's Swin V2-B backbone pretrained on NAIP aerial RGB imagery
(SatlasPretrain) as a frozen feature extractor -- no training happens here
at all. NAIP is the closest available domain match to this project's own
aerial orthophoto imagery among TorchGeo's pretrained options: both are
high-resolution top-down RGB aerial capture, unlike the Sentinel-2 (10 m/px)
or Landsat (30 m/px) alternatives, which would see an entire bike lane as at
most a sub-pixel smudge rather than a resolved texture.

There's no classifier here, learned or otherwise, in the gradient-descent
sense -- but plain nearest-neighbor cosine similarity against individual
reference embeddings (an earlier version of this module) turned out not to
work: pairwise similarity between *any* two pavement-like patches stayed
high (0.75-0.95) almost regardless of class, e.g. two genuinely different
bike-lane paint crops came out *less* similar to each other (0.85) than one
of them was to a plain road crop (0.87). That means this embedding's
dominant variance is something like "generic top-down aerial pavement
texture", shared by everything we feed it here, and the actual signal we
care about (reddish paint vs. gray asphalt) is a comparatively small
component riding on top of that -- swamped by raw cosine similarity to a
single reference, no matter how the reference crops are chosen.

The fix: `discriminant_score` projects a candidate embedding onto the
*difference* between the mean "bikelane" embedding and the mean "negative"
embedding, rather than comparing to either individually. That isolates
specifically the component that separates the two classes. Still zero
training -- it's arithmetic on frozen embeddings already extracted, not
anything fit by gradient descent -- but it rescues the signal that raw
similarity to a single reference was drowning out.

Reference embeddings are extracted once from the example images in
data/input/textures/<label>/ (one embedding per file, grouped by the
subfolder they're in -- "bikelane" and "negative").
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchgeo.models import Swin_V2_B_Weights, swin_v2_b

# Matches the resolution SatlasPretrain trained this backbone at.
INPUT_SIZE_PX = 256

_model: torch.nn.Module | None = None
_device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def _load_model() -> torch.nn.Module:
    global _model
    if _model is None:
        model = swin_v2_b(weights=Swin_V2_B_Weights.NAIP_RGB_SI_SATLAS)
        model.head = torch.nn.Identity()  # drop the 1000-class head, keep the pooled embedding
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


def _class_mean(embeddings: dict[str, np.ndarray]) -> np.ndarray:
    """Mean of the (unit-normalized) embeddings for one class."""
    return np.mean([_unit(e) for e in embeddings.values()], axis=0)


def discriminant_direction(
    references: dict[str, dict[str, np.ndarray]],
    positive_label: str = "bikelane",
    negative_label: str = "negative",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (direction, midpoint) separating `positive_label` from `negative_label`.

    `direction` points from the negative class's mean embedding towards the
    positive class's; `midpoint` is halfway between them. See
    `discriminant_score`, which uses this internally -- exposed separately
    so callers scoring many embeddings against the same reference set (e.g.
    a sliding-window scan) can compute it once rather than per window.
    """
    positive_mean = _class_mean(references[positive_label])
    negative_mean = _class_mean(references[negative_label])
    return positive_mean - negative_mean, (positive_mean + negative_mean) / 2


def discriminant_score(
    embedding: np.ndarray,
    references: dict[str, dict[str, np.ndarray]],
    positive_label: str = "bikelane",
    negative_label: str = "negative",
) -> float:
    """Project `embedding` onto the direction separating `positive_label` from `negative_label`.

    Positive values mean `embedding` sits on the `positive_label` side of
    the midpoint between the two classes' mean embeddings; negative values
    mean it sits on the `negative_label` side. A magnitude of ~1 means about
    as far from the midpoint as the class means themselves are -- unlike raw
    cosine similarity to individual references, this isolates specifically
    the component that distinguishes the two classes (see module
    docstring).
    """
    direction, midpoint = discriminant_direction(references, positive_label, negative_label)
    return float(np.dot(_unit(embedding) - midpoint, direction) / (np.linalg.norm(direction) + 1e-8))
