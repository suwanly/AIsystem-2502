from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import HTTPException
from insightface.app import FaceAnalysis
from PIL import Image
import io


# ---------------------------------------------------------------------------
# Global model handle & constants
# ---------------------------------------------------------------------------

_face_app: Optional[FaceAnalysis] = None


# ArcFace / RetinaFace 5-point template (112x112)
ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


class FacePipelineError(Exception):
    """Internal exception used to signal predictable pipeline failures."""
    pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_face_app() -> FaceAnalysis:
    """
    Lazy-initialize the InsightFace FaceAnalysis object.

    - name='buffalo_l' : includes RetinaFace detector + ArcFace recognizer
    - providers=['CPUExecutionProvider'] : CPU-only for reproducible Docker runs
    - ctx_id=-1 : force CPU
    """
    global _face_app
    if _face_app is None:
        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=-1, det_size=(640, 640))
        _face_app = app
    return _face_app


def _bytes_to_image(image_bytes: bytes) -> np.ndarray:
    """
    Decode raw image bytes into a BGR numpy array understood by OpenCV.

    1. Try PIL (handles many formats/modes).
    2. Fallback to cv2.imdecode if needed.
    """
    # First, try decoding via PIL
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        # Normalize to RGB first, then convert to BGR for OpenCV
        if pil_img.mode not in ("RGB", "L", "RGBA"):
            pil_img = pil_img.convert("RGB")

        np_img = np.array(pil_img)

        if len(np_img.shape) == 2:  # grayscale
            bgr = cv2.cvtColor(np_img, cv2.COLOR_GRAY2BGR)
        elif np_img.shape[2] == 4:  # RGBA
            bgr = cv2.cvtColor(np_img, cv2.COLOR_RGBA2BGR)
        else:  # RGB
            bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

        return bgr
    except Exception:
        # Fallback: use OpenCV's imdecode directly
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise FacePipelineError("Failed to decode image bytes.")
        return img


def _ensure_bgr(image: Any) -> np.ndarray:
    """
    Ensure input is a BGR numpy array. Accepts raw bytes or numpy arrays.
    """
    if isinstance(image, (bytes, bytearray)):
        return _bytes_to_image(image)
    if isinstance(image, np.ndarray):
        return image
    raise FacePipelineError("Unsupported image type; expected bytes or numpy array.")


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        raise FacePipelineError("Invalid embedding: zero norm.")

    sim = float(np.dot(a, b) / (norm_a * norm_b + 1e-6))
    # Clamp to [-1, 1] for safety
    return max(-1.0, min(1.0, sim))


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------

def detect_faces(image: Any) -> List[Dict[str, Any]]:
    """
    Detect faces in the input image.

    Parameters
    ----------
    image : bytes or np.ndarray (BGR)

    Returns
    -------
    List[Dict[str, Any]]
        Each dict has:
        - "bbox": [x1, y1, x2, y2]
        - "score": detection confidence
        - "landmarks": np.ndarray shape (5, 2) of 5-point facial landmarks
    """
    img_bgr = _ensure_bgr(image)
    app = _get_face_app()
    faces = app.get(img_bgr)

    results: List[Dict[str, Any]] = []
    for face in faces:
        bbox = face.bbox.astype(int)  # [x1, y1, x2, y2]
        kps = face.kps.astype(np.float32)  # (5, 2)

        results.append(
            {
                "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                "score": float(face.det_score),
                "landmarks": kps,
            }
        )

    # Sort by area (largest face first); if tie, by score
    def _area(entry: Dict[str, Any]) -> float:
        x1, y1, x2, y2 = entry["bbox"]
        return float(max(0, x2 - x1) * max(0, y2 - y1))

    results.sort(key=lambda d: (_area(d), d["score"]), reverse=True)
    return results


def detect_face_keypoints(face_image: Any) -> np.ndarray:
    """
    Detect 5-point facial landmarks for the most prominent face.

    Returns
    -------
    np.ndarray
        (5, 2) array: left eye, right eye, nose, left mouth corner, right mouth corner.
    """
    faces = detect_faces(face_image)
    if not faces:
        raise FacePipelineError("No face detected for keypoint extraction.")
    return faces[0]["landmarks"]  # (5, 2)


def warp_face(image: Any, homography_matrix: Any) -> np.ndarray:
    """
    Warp the provided face image using the supplied homography/affine matrix.

    - If matrix is 2x3: cv2.warpAffine
    - If matrix is 3x3: cv2.warpPerspective

    Output size is fixed to 112x112.
    """
    img_bgr = _ensure_bgr(image)
    M = np.asarray(homography_matrix, dtype=np.float32)

    if M.shape == (2, 3):
        aligned = cv2.warpAffine(
            img_bgr,
            M,
            (112, 112),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    elif M.shape == (3, 3):
        aligned = cv2.warpPerspective(
            img_bgr,
            M,
            (112, 112),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    else:
        raise FacePipelineError(
            f"Unexpected transformation matrix shape {M.shape}; expected (2,3) or (3,3)."
        )

    return aligned


def antispoof_check(face_image: Any) -> float:
    """
    Very simple anti-spoofing heuristic.

    Uses a combination of:
    - Image sharpness (Laplacian variance)
    - Overall color standard deviation

    Returns
    -------
    float
        Score in [0, 1]; higher = more likely to be a real, live face.
    """
    img = _ensure_bgr(face_image)

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Sharpness via Laplacian variance
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(lap_var / 500.0, 1.0)

    # Color spread;very flat colors may indicate a printed/photo spoof
    if img.ndim == 3:
        color_std = float(np.std(img))
        color_score = min(color_std / 50.0, 1.0)
    else:
        color_score = 0.5

    # Weighted combination
    confidence = 0.6 * sharpness_score + 0.4 * color_score
    # Clamp to [0, 1]
    confidence = max(0.0, min(1.0, confidence))

    return float(confidence)


def compute_face_embedding(face_image: Any) -> np.ndarray:
    """
    Compute a 512-dimensional face embedding for an aligned 112x112 BGR face.

    Parameters
    ----------
    face_image : np.ndarray
        Expected shape: (112, 112, 3) or (112, 112) grayscale.

    Returns
    -------
    np.ndarray
        L2-normalized embedding vector of shape (512,).
    """
    if isinstance(face_image, (bytes, bytearray)):
        face = _bytes_to_image(face_image)
    elif isinstance(face_image, np.ndarray):
        face = face_image
    else:
        raise FacePipelineError("Embedding input must be bytes or numpy array.")

    if face.shape[0] != 112 or face.shape[1] != 112:
        raise FacePipelineError(f"Expected aligned 112x112 face, got shape {face.shape[:2]}.")

    if face.ndim == 2:
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)

    app = _get_face_app()
    rec_model = app.models.get("recognition")
    if rec_model is None:
        raise FacePipelineError("Face recognition model not loaded.")

    embedding = rec_model.get_feat(face)
    embedding = np.asarray(embedding, dtype=np.float32).flatten()

    # L2-normalize (safe even if model already returns normalized vector)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


def calculate_face_similarity(image_a: Any, image_b: Any) -> float:
    """
    End-to-end face similarity pipeline.

    Steps
    -----
    1. Decode raw image bytes (if needed) to BGR numpy arrays.
    2. Detect faces and 5-point landmarks in both images.
    3. Estimate affine transforms that map landmarks to the ArcFace template.
    4. Warp faces to 112x112 aligned crops.
    5. Run anti-spoofing checks; optionally reject if too low.
    6. Compute face embeddings for both aligned faces.
    7. Return cosine similarity between embeddings.

    Returns
    -------
    float
        Cosine similarity in [-1, 1].
    """
    try:
        img_a = _ensure_bgr(image_a)
        img_b = _ensure_bgr(image_b)

        # 1) Landmark detection
        kps_a = detect_face_keypoints(img_a)  # (5, 2)
        kps_b = detect_face_keypoints(img_b)  # (5, 2)

        # 2) Estimate affine transforms to ArcFace template
        M_a, _ = cv2.estimateAffinePartial2D(
            kps_a, ARCFACE_DST, method=cv2.LMEDS
        )
        M_b, _ = cv2.estimateAffinePartial2D(
            kps_b, ARCFACE_DST, method=cv2.LMEDS
        )

        if M_a is None or M_b is None:
            raise FacePipelineError("Failed to compute affine transform for alignment.")

        # 3) Warp to 112x112 aligned faces
        aligned_a = warp_face(img_a, M_a)
        aligned_b = warp_face(img_b, M_b)

        # 4) Simple anti-spoof checking
        spoof_a = antispoof_check(aligned_a)
        spoof_b = antispoof_check(aligned_b)

        SPOOF_THRESHOLD = 0.2  # heuristic; adjust if needed
        if spoof_a < SPOOF_THRESHOLD or spoof_b < SPOOF_THRESHOLD:
            raise FacePipelineError(
                f"Low liveness score (A={spoof_a:.2f}, B={spoof_b:.2f}); "
                "cannot reliably compute similarity."
            )

        # 5) Embeddings
        emb_a = compute_face_embedding(aligned_a)
        emb_b = compute_face_embedding(aligned_b)

        # 6) Cosine similarity
        similarity = _cosine_similarity(emb_a, emb_b)
        return float(similarity)

    except FacePipelineError as exc:
        msg = str(exc)
        # Map "no face" errors to a user-friendly 422
        if "No face detected" in msg:
            raise HTTPException(
                status_code=422,
                detail="사람 얼굴이 인식되지 않았습니다.",
            )
        raise HTTPException(
            status_code=422,
            detail=f"이미지 처리 오류: {msg}",
        )
    except HTTPException:
        # Re-raise FastAPI HTTPExceptions untouched
        raise
    except Exception as exc:
        # Any other unexpected error → 500
        raise HTTPException(
            status_code=500,
            detail=f"서버 내부 오류 발생: {exc}",
        )