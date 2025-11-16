from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import HTTPException
from insightface.app import FaceAnalysis
from PIL import Image
import io


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
    """Exception type for predictable face-pipeline errors."""
    pass



def _get_face_app() -> FaceAnalysis:
    """
    Return a singleton instance of InsightFace FaceAnalysis.

    Uses:
    - model pack: 'buffalo_l'
    - CPU only execution (ctx_id = -1)
    """
    global _face_app

    # 이미 초기화된 경우 바로 반환
    if isinstance(_face_app, FaceAnalysis):
        return _face_app

    # 최초 1회 초기화
    analyzer = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
    )
    # ctx_id = -1 : CPU 강제 사용
    analyzer.prepare(ctx_id=-1, det_size=(640, 640))

    _face_app = analyzer
    return _face_app


def _bytes_to_image(image_bytes: bytes) -> np.ndarray:
    """
    Decode image bytes into a BGR numpy array.

    1) 먼저 OpenCV imdecode로 시도
    2) 실패하면 PIL을 사용해서 다시 시도
    """
    # OpenCV로 먼저 한 번 시도
    np_buf = np.frombuffer(image_bytes, dtype=np.uint8)
    opencv_img = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
    if opencv_img is not None:
        return opencv_img

    # imdecode가 실패한 경우 PIL 경로
    try:
        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            # 대부분의 경우 RGB로 통일
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            rgb_arr = np.asarray(pil_img)
            bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
            return bgr_arr
    except Exception as exc:  # pragma: no cover - 예외 경로
        raise FacePipelineError(f"Image decoding failed: {exc}")


def _ensure_bgr(image: Any) -> np.ndarray:
    """
    Normalize input to a BGR image (numpy array).

    허용 타입:
    - bytes / bytearray (raw image bytes)
    - np.ndarray (H,W) or (H,W,C)
    """
    if isinstance(image, (bytes, bytearray)):
        return _bytes_to_image(image)

    if isinstance(image, np.ndarray):
        arr = image
        # 그레이스케일 → BGR
        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        # 컬러 이미지
        if arr.ndim == 3:
            # BGR 3채널
            if arr.shape[2] == 3:
                return arr
            # BGRA 4채널 → BGR
            if arr.shape[2] == 4:
                return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

        raise FacePipelineError(
            f"Unsupported ndarray shape for image: {arr.shape}"
        )

    raise FacePipelineError(
        f"Unsupported image type: {type(image)} (expected bytes or numpy.ndarray)"
    )


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1D vectors.

    결과 범위는 [-1, 1]로 클램핑.
    """
    vec_a = np.asarray(a, dtype=np.float32).ravel()
    vec_b = np.asarray(b, dtype=np.float32).ravel()

    denom_a = float(np.linalg.norm(vec_a))
    denom_b = float(np.linalg.norm(vec_b))
    if denom_a == 0.0 or denom_b == 0.0:
        raise FacePipelineError("Embedding with zero norm encountered.")

    sim_val = float(np.dot(vec_a, vec_b) / (denom_a * denom_b + 1e-6))
    # 수치 안정성을 위해 clipping
    return max(-1.0, min(1.0, sim_val))



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
    frame_bgr = _ensure_bgr(image)
    face_engine = _get_face_app()
    raw_faces = face_engine.get(frame_bgr)

    detections: List[Dict[str, Any]] = []
    for f in raw_faces:
        box = f.bbox.astype(int)  # [x1, y1, x2, y2]
        kps = f.kps.astype(np.float32)  # (5, 2)

        detections.append(
            {
                "bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                "score": float(f.det_score),
                "landmarks": kps,
            }
        )

    # 가장 큰 얼굴 + 높은 score 순으로 정렬
    def _area(entry: Dict[str, Any]) -> float:
        x1, y1, x2, y2 = entry["bbox"]
        return float(max(0, x2 - x1) * max(0, y2 - y1))

    detections.sort(
        key=lambda item: (_area(item), item["score"]),
        reverse=True,
    )
    return detections


def detect_face_keypoints(face_image: Any) -> np.ndarray:
    """
    Detect 5-point facial landmarks for the most prominent face.

    Returns
    -------
    np.ndarray
        (5, 2) array: left eye, right eye, nose, left mouth corner, right mouth corner.
    """
    face_list = detect_faces(face_image)
    if not face_list:
        raise FacePipelineError("No face detected for keypoint extraction.")
    # 가장 우선순위 높은 얼굴의 keypoints 반환
    keypoints = face_list[0]["landmarks"]
    return keypoints


def warp_face(image: Any, homography_matrix: Any) -> np.ndarray:
    """
    Warp the provided face image using the supplied homography/affine matrix.

    - If matrix is 2x3: cv2.warpAffine
    - If matrix is 3x3: cv2.warpPerspective

    Output size is fixed to 112x112.
    """
    src_img = _ensure_bgr(image)
    mat = np.asarray(homography_matrix, dtype=np.float32)

    if mat.shape == (2, 3):
        aligned_img = cv2.warpAffine(
            src_img,
            mat,
            (112, 112),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return aligned_img

    if mat.shape == (3, 3):
        aligned_img = cv2.warpPerspective(
            src_img,
            mat,
            (112, 112),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return aligned_img

    raise FacePipelineError(
        f"Unexpected transformation matrix shape {mat.shape}; expected (2,3) or (3,3)."
    )


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
    bgr_face = _ensure_bgr(face_image)

    # 명암 성분
    gray_face = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2GRAY)

    # Laplacian으로 대략적인 초점(샤프니스) 측정
    focus_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
    focus_score = focus_var / 500.0
    focus_score = float(np.clip(focus_score, 0.0, 1.0))

    # 색 분산이 너무 낮으면 인쇄물/화면 캡쳐일 가능성
    color_dispersion = float(np.std(bgr_face))
    color_score = color_dispersion / 50.0
    color_score = float(np.clip(color_score, 0.0, 1.0))

    # 두 지표를 단순 가중합
    raw_conf = 0.6 * focus_score + 0.4 * color_score
    final_conf = float(np.clip(raw_conf, 0.0, 1.0))
    return final_conf


def compute_face_embedding(face_image: Any) -> np.ndarray:
    """
    Compute a 512-dimensional face embedding for an aligned 112x112 BGR face.

    Parameters
    ----------
    face_image : np.ndarray or bytes
        Expected shape: (112, 112, 3) or (112, 112) grayscale.

    Returns
    -------
    np.ndarray
        L2-normalized embedding vector of shape (512,).
    """
    # 입력 타입 유연하게 처리
    if isinstance(face_image, (bytes, bytearray)):
        aligned = _bytes_to_image(face_image)
    elif isinstance(face_image, np.ndarray):
        aligned = face_image
    else:
        raise FacePipelineError("Embedding input must be bytes or numpy array.")

    # 해상도 체크
    if aligned.shape[0] != 112 or aligned.shape[1] != 112:
        raise FacePipelineError(
            f"Expected aligned 112x112 face, got shape {aligned.shape[:2]}."
        )

    # 그레이스케일일 경우 BGR 변환
    if aligned.ndim == 2:
        aligned = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR)

    face_app = _get_face_app()
    rec_model = face_app.models.get("recognition")
    if rec_model is None:
        raise FacePipelineError("Face recognition model not loaded.")

    raw_feat = rec_model.get_feat(aligned)
    emb = np.asarray(raw_feat, dtype=np.float32).flatten()

    # L2 정규화
    norm_val = float(np.linalg.norm(emb))
    if norm_val > 0.0:
        emb = emb / norm_val

    return emb


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
        # 1) 디코딩 → BGR
        img_a = _ensure_bgr(image_a)
        img_b = _ensure_bgr(image_b)

        # 2) 랜드마크 검출
        kps_a = detect_face_keypoints(img_a)  # (5, 2)
        kps_b = detect_face_keypoints(img_b)  # (5, 2)

        # 3) ArcFace 템플릿으로 보정할 affine transform 계산
        aff_a, _ = cv2.estimateAffinePartial2D(
            kps_a, ARCFACE_DST, method=cv2.LMEDS
        )
        aff_b, _ = cv2.estimateAffinePartial2D(
            kps_b, ARCFACE_DST, method=cv2.LMEDS
        )

        if aff_a is None or aff_b is None:
            raise FacePipelineError("Failed to compute affine transform for alignment.")

        # 4) 112x112 정렬 얼굴 생성
        aligned_a = warp_face(img_a, aff_a)
        aligned_b = warp_face(img_b, aff_b)

        # 5) 간단한 anti-spoof 검사
        live_score_a = antispoof_check(aligned_a)
        live_score_b = antispoof_check(aligned_b)

        SPOOF_THRESHOLD = 0.2  # 필요시 조정 가능
        if live_score_a < SPOOF_THRESHOLD or live_score_b < SPOOF_THRESHOLD:
            raise FacePipelineError(
                f"Low liveness score (A={live_score_a:.2f}, B={live_score_b:.2f}); "
                "similarity cannot be trusted."
            )

        # 6) 임베딩 계산
        emb_a = compute_face_embedding(aligned_a)
        emb_b = compute_face_embedding(aligned_b)

        # 7) 코사인 유사도
        sim = _cosine_similarity(emb_a, emb_b)
        return float(sim)

    except FacePipelineError as exc:
        err_msg = str(exc)
        # 얼굴이 아예 안 잡힌 경우 → 422 + 한글 메시지
        if "No face detected" in err_msg:
            raise HTTPException(
                status_code=422,
                detail="사람 얼굴이 인식되지 않았습니다.",
            )
        # 기타 파이프라인 오류 → 422
        raise HTTPException(
            status_code=422,
            detail=f"이미지 처리 오류: {err_msg}",
        )

    except HTTPException:
        # 이미 HTTPException인 경우 그대로 전달
        raise

    except Exception as exc:  # 예기치 못한 오류 → 500
        raise HTTPException(
            status_code=500,
            detail=f"서버 내부 오류 발생: {exc}",
        )
