###############################################################################
#  SoulX-FlashHead avatar adapter for LiveTalking
#  Runs SoulX in a separate subprocess (different venv / different transformers)
#  and bridges audio/frames over stdin/stdout pipes.
###############################################################################

import os
import sys
import pickle
import struct
import subprocess
import queue
import time
import cv2
import numpy as np

from threading import Thread
from avatars.audio_features.soulx_asr import SoulXASR
from avatars.base_avatar import BaseAvatar, AudioFrameData
from utils.image import mirror_index
from utils.logger import logger
from utils.device import initialize_device
from registry import register


def _detect_face_bbox(photo_bgr):
    """Return (x1, y1, x2, y2) bounding box of the largest face in the photo."""
    gray = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    cascade = cv2.CascadeClassifier(cascade_path)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    h, w = photo_bgr.shape[:2]
    if len(faces) == 0:
        # Fallback: centre half of the image
        return (w // 4, h // 4, 3 * w // 4, 3 * h // 4)
    # Pick largest face
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    # Expand 10 % to include forehead / chin
    pad_x = int(fw * 0.10)
    pad_y = int(fh * 0.10)
    return (
        max(0, x - pad_x),
        max(0, y - pad_y),
        min(w, x + fw + pad_x),
        min(h, y + fh + pad_y),
    )


# ── module-level functions expected by app.py ──────────────────────────────

def load_model(opt):
    """SoulX model lives in the subprocess; return opt so __init__ can read paths."""
    return opt


def load_avatar(avatar_id):
    """Load reference photo and detect face bbox.

    Returns (photo_bgr, photo_path, face_bbox).
    The photo must exist at data/avatars/<avatar_id>/source.jpg (or .png).
    """
    avatar_path = f"./data/avatars/{avatar_id}"
    # Look for source image
    for ext in ('source.jpg', 'source.jpeg', 'source.png', 'source.webp'):
        candidate = os.path.join(avatar_path, ext)
        if os.path.exists(candidate):
            photo_path = candidate
            break
    else:
        # Fall back to any image in the directory
        import glob
        imgs = glob.glob(os.path.join(avatar_path, '*.[jpJP][pnPN]*[gG]'))
        imgs += glob.glob(os.path.join(avatar_path, '*.webp'))
        if not imgs:
            raise FileNotFoundError(
                f"No reference image found in {avatar_path}. "
                "Place a portrait photo as data/avatars/<id>/source.jpg"
            )
        photo_path = imgs[0]

    photo_bgr = cv2.imread(photo_path)
    if photo_bgr is None:
        raise IOError(f"Could not read image: {photo_path}")

    face_bbox = _detect_face_bbox(photo_bgr)
    logger.info(f"[SoulX] reference photo: {photo_path}, face bbox: {face_bbox}")
    return (photo_bgr, os.path.abspath(photo_path), face_bbox)


def warm_up(batch_size, avatar):
    """No-op: SoulX warms up inside the worker subprocess on first call."""
    pass


# ── avatar class ────────────────────────────────────────────────────────────

@register("avatar", "soulx")
class SoulXAvatar(BaseAvatar):

    def __init__(self, opt, model, avatar):
        super().__init__(opt)

        photo_bgr, photo_path, face_bbox = avatar
        self.frame_list_cycle = [photo_bgr]  # used for silent frames
        self.face_bbox = face_bbox

        # Paths from opt (with defaults for RunPod)
        ckpt_dir    = getattr(opt, 'soulx_ckpt_dir',    '/workspace/models/SoulX-FlashHead-1_3B')
        wav2vec_dir = getattr(opt, 'soulx_wav2vec_dir', '/workspace/models/wav2vec2-base-960h')
        venv_python = getattr(opt, 'soulx_venv_python', '/workspace/soulx_venv/bin/python')

        worker_script = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'soulx_worker.py')
        )

        logger.info('[SoulX] spawning worker subprocess…')
        self._worker = subprocess.Popen(
            [venv_python, worker_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
        )

        self._send({'type': 'init', 'ckpt_dir': ckpt_dir,
                    'wav2vec_dir': wav2vec_dir, 'photo_path': photo_path})
        resp = self._recv()
        if resp is None or resp.get('status') != 'ready':
            err = resp.get('error', 'unknown') if resp else 'no response'
            raise RuntimeError(f"SoulX worker init failed: {err}")

        self._infer_params = resp
        logger.info(f"[SoulX] worker ready — params: {resp}")

        self.asr = SoulXASR(opt, self)
        self.asr.warm_up()

    # ── IPC helpers ────────────────────────────────────────────────────────

    def _send(self, obj):
        data = pickle.dumps(obj, protocol=4)
        self._worker.stdin.write(struct.pack('>I', len(data)))
        self._worker.stdin.write(data)
        self._worker.stdin.flush()

    def _recv(self):
        header = self._worker.stdout.read(4)
        if len(header) < 4:
            return None
        size = struct.unpack('>I', header)[0]
        payload = self._worker.stdout.read(size)
        return pickle.loads(payload)

    def _worker_buffer(self, raw_pcm: np.ndarray):
        """Forward silent audio to the worker to keep its buffer in sync."""
        self._send({'type': 'buffer', 'audio': raw_pcm})
        self._recv()  # consume ACK

    def _worker_infer(self, raw_pcm: np.ndarray, n_frames: int):
        """Forward audio and request inference; returns (N, H, W, 3) uint8 or None."""
        self._send({'type': 'infer', 'audio': raw_pcm, 'n_frames': n_frames})
        resp = self._recv()
        if resp and resp.get('status') == 'ok' and resp.get('frames') is not None:
            return resp['frames']
        if resp and resp.get('status') == 'error':
            logger.warning(f"[SoulX] worker infer error: {resp.get('error')}")
        return None

    # ── inference loop (overrides BaseAvatar.inference) ───────────────────

    def inference(self, quit_event):
        """Override to always forward audio (silent or not) to the worker."""
        length = len(self.frame_list_cycle)
        index = 0

        logger.info('[SoulX] inference thread started')
        while not quit_event.is_set():
            try:
                raw_pcm = self.asr.feat_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            is_all_silence = True
            audio_frames = []
            for _ in range(self.batch_size * 2):
                af: AudioFrameData = self.asr.output_queue.get()
                if af.type == 0:
                    is_all_silence = False
                audio_frames.append(af)

            if is_all_silence:
                self._worker_buffer(raw_pcm)
                for i in range(self.batch_size):
                    idx = mirror_index(length, index)
                    self.res_frame_queue.put((None, audio_frames[i * 2:i * 2 + 2], idx))
                    index += 1
            else:
                frames = self._worker_infer(raw_pcm, self.batch_size)
                if frames is not None and len(frames) > 0:
                    for i in range(self.batch_size):
                        frame_i = min(i, len(frames) - 1)
                        idx = mirror_index(length, index)
                        self.res_frame_queue.put((frames[frame_i], audio_frames[i * 2:i * 2 + 2], idx))
                        index += 1
                else:
                    # Worker failed — fall back to static frame
                    for i in range(self.batch_size):
                        idx = mirror_index(length, index)
                        self.res_frame_queue.put((None, audio_frames[i * 2:i * 2 + 2], idx))
                        index += 1

        logger.info('[SoulX] inference thread stopped')

    # ── required by process_frames in BaseAvatar ──────────────────────────

    def inference_batch(self, index, audiofeat_batch):
        """Not called — inference() is overridden. Kept to satisfy the interface."""
        raise NotImplementedError('SoulXAvatar overrides inference() directly')

    def paste_back_frame(self, pred_frame, idx: int):
        """Paste SoulX output face onto the original background photo."""
        x1, y1, x2, y2 = self.face_bbox
        bg = self.frame_list_cycle[0].copy()
        face_w, face_h = x2 - x1, y2 - y1
        resized = cv2.resize(pred_frame.astype(np.uint8), (face_w, face_h))
        bg[y1:y2, x1:x2] = resized
        return bg

    def __del__(self):
        try:
            self._send({'type': 'exit'})
        except Exception:
            pass
        try:
            self._worker.terminate()
        except Exception:
            pass
