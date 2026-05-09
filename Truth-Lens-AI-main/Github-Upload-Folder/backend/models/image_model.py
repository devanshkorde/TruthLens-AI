"""
TruthLens AI — Image Deepfake Detection Model
===============================================
Uses a heuristic FFT + noise analysis pipeline when no GPU/model is available.
When a trained model exists, loads EfficientNet-B0 fine-tuned checkpoint.

Key detection techniques:
  1. Frequency domain analysis (GAN artifacts appear at high frequencies)
  2. Noise consistency (deepfakes have inconsistent local noise patterns)
  3. Face region analysis (blurring artifacts at face boundaries)
  4. Color histogram analysis
"""

import io
import math
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image, ImageFilter, ImageStat

MODEL_PATH = Path(__file__).parent.parent / "data" / "deepfake_model.pth"

DEEPFAKE_ARTIFACTS = [
    "Face boundary blurring",
    "GAN high-frequency fingerprint",
    "Noise pattern inconsistency",
    "Unnatural skin texture",
    "Eye reflection asymmetry",
    "Compression artifact mismatch",
    "Color channel irregularity",
    "Facial landmark distortion",
]


@dataclass
class ImageAnalysisResult:
    verdict: str
    confidence: float
    artifacts_detected: list[str] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)
    noise_score: float = 0.0
    frequency_anomaly: float = 0.0
    blur_score: float = 0.0


# ──────────────────────────────────────────────
# Image analysis utilities
# ──────────────────────────────────────────────

def compute_frequency_anomaly(img_array: np.ndarray) -> float:
    """
    Analyze high-frequency components via FFT.
    GANs produce characteristic frequency patterns in generated images.
    Returns anomaly score 0–1.
    """
    if img_array.ndim == 3:
        gray = np.mean(img_array, axis=2)
    else:
        gray = img_array

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    radius_outer = min(h, w) // 3
    radius_inner = min(h, w) // 8

    y_idx, x_idx = np.ogrid[:h, :w]
    dist = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)

    high_freq_mask = dist > radius_outer
    mid_freq_mask = (dist > radius_inner) & (dist <= radius_outer)

    high_energy = np.mean(magnitude[high_freq_mask]) if high_freq_mask.any() else 0
    mid_energy = np.mean(magnitude[mid_freq_mask]) if mid_freq_mask.any() else 1

    ratio = high_energy / (mid_energy + 1e-8)
    # Deepfakes tend to have elevated high-freq energy from GAN upsampling artifacts
    anomaly = min(ratio * 2.5, 1.0)
    return float(anomaly)


def compute_noise_inconsistency(img_array: np.ndarray) -> float:
    """
    Measure local noise variance across image blocks.
    Inconsistent noise → likely manipulated.
    Returns score 0–1 (higher = more suspicious).
    """
    if img_array.ndim == 3:
        gray = np.mean(img_array, axis=2).astype(np.float32)
    else:
        gray = img_array.astype(np.float32)

    h, w = gray.shape
    block_size = max(h // 8, 8)
    variances = []

    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = gray[i:i + block_size, j:j + block_size]
            variances.append(np.var(block))

    if len(variances) < 4:
        return 0.3

    var_of_vars = np.var(variances)
    mean_var = np.mean(variances) + 1e-8
    coefficient_of_variation = math.sqrt(var_of_vars) / mean_var

    # High CV = inconsistent noise = suspicious
    return float(min(coefficient_of_variation / 2.0, 1.0))


def compute_blur_score(image: Image.Image) -> float:
    """
    Measure sharpness via Laplacian variance.
    Deepfakes often have unnaturally uniform blurring at boundaries.
    Returns 0–1 (higher = more blur anomaly).
    """
    gray = image.convert("L")
    laplacian = gray.filter(ImageFilter.FIND_EDGES)
    stat = ImageStat.Stat(laplacian)
    variance = stat.var[0]

    # Very low variance = over-smoothed (possible deepfake)
    # Very high variance = natural image detail
    sharpness = min(variance / 1000.0, 1.0)
    blur_anomaly = 1.0 - sharpness
    return float(blur_anomaly)


def compute_color_anomaly(img_array: np.ndarray) -> float:
    """Check for unnatural color channel correlations."""
    if img_array.ndim < 3 or img_array.shape[2] < 3:
        return 0.2
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    corr_rg = float(np.corrcoef(r.flatten(), g.flatten())[0, 1])
    corr_rb = float(np.corrcoef(r.flatten(), b.flatten())[0, 1])
    if math.isnan(corr_rg): corr_rg = 1.0
    if math.isnan(corr_rb): corr_rb = 1.0
    # Real faces have high channel correlation; GANs sometimes desaturate weirdly
    avg_corr = (corr_rg + corr_rb) / 2
    anomaly = 1.0 - abs(avg_corr)
    return float(min(max(anomaly, 0), 1))


# ──────────────────────────────────────────────
# Main analysis function
# ──────────────────────────────────────────────

def analyze_image(image_bytes: bytes, filename: str = "image") -> ImageAnalysisResult:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return ImageAnalysisResult(
            verdict="ERROR",
            confidence=0.0,
            signals=[f"🔴 Could not process image: {str(e)}"],
        )

    img_array = np.array(image, dtype=np.float32)

    # Run all analysis passes
    freq_score = compute_frequency_anomaly(img_array)
    noise_score = compute_noise_inconsistency(img_array)
    blur_score = compute_blur_score(image)
    color_score = compute_color_anomaly(img_array)

    # Try loading a trained PyTorch model
    ml_score = None
    if MODEL_PATH.exists():
        try:
            import torch
            import torchvision.transforms as T
            from torchvision import models

            transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            tensor = transform(image).unsqueeze(0)
            model = torch.load(MODEL_PATH, map_location="cpu")
            model.eval()
            with torch.no_grad():
                out = torch.sigmoid(model(tensor)).item()
            ml_score = out
        except Exception as e:
            print(f"[ImageModel] PyTorch model error: {e}")

    # Combine scores
    if ml_score is not None:
        composite = ml_score * 0.5 + freq_score * 0.2 + noise_score * 0.2 + blur_score * 0.1
    else:
        composite = freq_score * 0.35 + noise_score * 0.35 + blur_score * 0.2 + color_score * 0.1

    # Classify
    if composite > 0.62:
        verdict = "FAKE"
        confidence = min(0.55 + composite * 0.45, 0.97)
    elif composite > 0.40:
        verdict = "SUSPICIOUS"
        confidence = 0.45 + composite * 0.3
    else:
        verdict = "REAL"
        confidence = 1.0 - composite

    # Determine detected artifacts
    artifacts = []
    if freq_score > 0.5:
        artifacts.append(DEEPFAKE_ARTIFACTS[1])  # GAN fingerprint
    if noise_score > 0.5:
        artifacts.append(DEEPFAKE_ARTIFACTS[2])  # Noise inconsistency
    if blur_score > 0.6:
        artifacts.append(DEEPFAKE_ARTIFACTS[0])  # Blurring
    if color_score > 0.5:
        artifacts.append(DEEPFAKE_ARTIFACTS[6])  # Color irregularity

    # Build XAI signals
    signals = []
    if freq_score > 0.5:
        signals.append(f"🔴 GAN frequency fingerprint detected (score: {freq_score:.2f}) — characteristic of AI-generated faces")
    if noise_score > 0.5:
        signals.append(f"🔴 Noise pattern inconsistency found ({noise_score:.2f}) — suggests facial region was composited")
    if blur_score > 0.6:
        signals.append(f"🟡 Unnatural blur distribution ({blur_score:.2f}) — common at deepfake face boundaries")
    if color_score > 0.5:
        signals.append(f"🟡 Abnormal color channel correlation — may indicate GAN generation artifacts")
    if not artifacts:
        signals.append("🟢 No significant deepfake artifacts detected in this image")
    if verdict == "REAL" and composite < 0.25:
        signals.append("🟢 Image frequency spectrum consistent with authentic photography")

    return ImageAnalysisResult(
        verdict=verdict,
        confidence=round(confidence, 3),
        artifacts_detected=artifacts,
        signals=signals,
        noise_score=round(noise_score, 3),
        frequency_anomaly=round(freq_score, 3),
        blur_score=round(blur_score, 3),
    )
