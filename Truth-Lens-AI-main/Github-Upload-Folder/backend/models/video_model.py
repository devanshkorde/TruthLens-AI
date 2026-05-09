"""
TruthLens AI — Video Deepfake Detection Model
===============================================
Extracts frames from video and applies the image model's heuristics.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from models.image_model import analyze_image

@dataclass
class VideoAnalysisResult:
    verdict: str
    confidence: float
    artifacts_detected: list[str] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)
    frames_analyzed: int = 0
    avg_noise_score: float = 0.0
    avg_frequency_anomaly: float = 0.0
    avg_blur_score: float = 0.0

def analyze_video(video_path: str) -> VideoAnalysisResult:
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return VideoAnalysisResult(verdict="ERROR", confidence=0.0, signals=["🔴 Could not open video file"])
            
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            return VideoAnalysisResult(verdict="ERROR", confidence=0.0, signals=["🔴 Video has no frames"])

        num_frames_to_extract = 5
        if frame_count < num_frames_to_extract:
            frame_indices = list(range(frame_count))
        else:
            frame_indices = np.linspace(0, frame_count - 1, num_frames_to_extract, dtype=int)

        results = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            is_success, buffer = cv2.imencode(".jpg", frame)
            if not is_success:
                continue
            
            image_bytes = buffer.tobytes()
            res = analyze_image(image_bytes, filename=f"frame_{idx}")
            if res.verdict != "ERROR":
                results.append(res)
                
        cap.release()

        if not results:
            return VideoAnalysisResult(verdict="ERROR", confidence=0.0, signals=["🔴 Failed to extract and analyze frames"])

        avg_noise = sum(r.noise_score for r in results) / len(results)
        avg_freq = sum(r.frequency_anomaly for r in results) / len(results)
        avg_blur = sum(r.blur_score for r in results) / len(results)
        
        fake_count = sum(1 for r in results if r.verdict == "FAKE")
        suspicious_count = sum(1 for r in results if r.verdict == "SUSPICIOUS")
        
        all_artifacts = set()
        for r in results:
            all_artifacts.update(r.artifacts_detected)
            
        if fake_count >= 2:
            verdict = "FAKE"
            confidence = sum(r.confidence for r in results if r.verdict == "FAKE") / fake_count
        elif fake_count == 1 or suspicious_count >= 2:
            verdict = "SUSPICIOUS"
            confidence = max(r.confidence for r in results if r.verdict in ("FAKE", "SUSPICIOUS"))
        else:
            verdict = "REAL"
            confidence = sum(r.confidence for r in results if r.verdict == "REAL") / len(results) if len(results) > 0 else 0.99

        signals = []
        if fake_count > 0:
            signals.append(f"🔴 Deepfake artifacts detected in {fake_count} out of {len(results)} analyzed frames")
        if avg_freq > 0.5:
            signals.append(f"🔴 Persistent GAN frequency fingerprint detected across frames (avg score: {avg_freq:.2f})")
        if avg_noise > 0.5:
            signals.append(f"🔴 Temporal noise inconsistency found ({avg_noise:.2f}) — suggests facial region was composited")
        if avg_blur > 0.6:
            signals.append(f"🟡 Unnatural blur distribution ({avg_blur:.2f}) — common at deepfake face boundaries")
        if not all_artifacts:
            signals.append(f"🟢 Temporal consistency is natural across {len(results)} frames")

        return VideoAnalysisResult(
            verdict=verdict,
            confidence=round(confidence, 3),
            artifacts_detected=list(all_artifacts),
            signals=signals,
            frames_analyzed=len(results),
            avg_noise_score=round(avg_noise, 3),
            avg_frequency_anomaly=round(avg_freq, 3),
            avg_blur_score=round(avg_blur, 3),
        )

    except Exception as e:
        return VideoAnalysisResult(verdict="ERROR", confidence=0.0, signals=[f"🔴 Video processing failed: {str(e)}"])
