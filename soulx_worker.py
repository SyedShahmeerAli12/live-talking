#!/usr/bin/env python
"""
SoulX-FlashHead worker process.
Must be run with the SoulX venv Python:
    /workspace/soulx_venv/bin/python soulx_worker.py

Parent process communicates via stdin/stdout using
4-byte big-endian length prefix + pickle payload.
All diagnostic output goes to stderr so stdout stays clean.
"""

import sys
import pickle
import struct
import numpy as np


def _send(obj):
    data = pickle.dumps(obj, protocol=4)
    sys.stdout.buffer.write(struct.pack('>I', len(data)))
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def _recv():
    header = sys.stdin.buffer.read(4)
    if len(header) < 4:
        return None
    size = struct.unpack('>I', header)[0]
    payload = sys.stdin.buffer.read(size)
    if len(payload) < size:
        return None
    return pickle.loads(payload)


def main():
    # ── init ──────────────────────────────────────────────────────────────
    cmd = _recv()
    if cmd is None or cmd.get('type') != 'init':
        _send({'status': 'error', 'error': 'expected init command'})
        sys.exit(1)

    ckpt_dir    = cmd['ckpt_dir']
    wav2vec_dir = cmd['wav2vec_dir']
    photo_path  = cmd['photo_path']

    try:
        import sys as _sys
        _sys.path.insert(0, '/workspace/SoulX-FlashHead')
        from flash_head.inference import (
            get_pipeline, get_base_data, get_infer_params,
            get_audio_embedding, run_pipeline,
        )
    except ImportError as exc:
        _send({'status': 'error', 'error': f'import failed: {exc}'})
        sys.exit(1)

    try:
        print('[soulx_worker] loading pipeline…', file=sys.stderr, flush=True)
        pipeline = get_pipeline(
            world_size=1,
            ckpt_dir=ckpt_dir,
            wav2vec_dir=wav2vec_dir,
            model_type='lite',
        )
        print('[soulx_worker] loading base data…', file=sys.stderr, flush=True)
        get_base_data(pipeline, cond_image_path_or_dir=photo_path, base_seed=42, use_face_crop=True)
        infer_params = get_infer_params()
    except Exception as exc:
        _send({'status': 'error', 'error': str(exc)})
        sys.exit(1)

    sample_rate          = int(infer_params.get('sample_rate', 16000))
    tgt_fps              = float(infer_params.get('tgt_fps', 25))
    cached_audio_duration = float(infer_params.get('cached_audio_duration', 10.0))
    frame_num            = int(infer_params.get('frame_num', 5))
    motion_frames_num    = int(infer_params.get('motion_frames_num', 2))

    _send({
        'status': 'ready',
        'sample_rate': sample_rate,
        'tgt_fps': tgt_fps,
        'cached_audio_duration': cached_audio_duration,
        'frame_num': frame_num,
        'motion_frames_num': motion_frames_num,
    })
    print('[soulx_worker] ready', file=sys.stderr, flush=True)

    # ── inference loop ─────────────────────────────────────────────────────
    audio_buffer      = np.zeros(0, dtype=np.float32)
    max_audio_samples = int(cached_audio_duration * sample_rate)
    frame_counter     = 0.0  # cumulative video frames worth of audio processed

    while True:
        msg = _recv()
        if msg is None:
            break

        msg_type   = msg.get('type')
        audio_chunk = msg.get('audio')  # numpy float32 or None

        if audio_chunk is not None:
            audio_buffer = np.concatenate([audio_buffer, audio_chunk])
            if len(audio_buffer) > max_audio_samples:
                audio_buffer = audio_buffer[-max_audio_samples:]

        if msg_type == 'exit':
            break

        elif msg_type == 'buffer':
            # Advance timing counter without generating frames
            if audio_chunk is not None:
                frame_counter += len(audio_chunk) / sample_rate * tgt_fps
            _send({'status': 'ok'})

        elif msg_type == 'infer':
            n_frames = int(msg.get('n_frames', frame_num))
            try:
                audio_end_idx   = frame_counter + n_frames
                audio_start_idx = audio_end_idx - frame_num

                emb   = get_audio_embedding(pipeline, audio_buffer, audio_start_idx, audio_end_idx)
                video = run_pipeline(pipeline, emb)
                video = video[motion_frames_num:]
                frames = video.cpu().numpy().astype(np.uint8)

                # Ensure (N, H, W, 3)
                if frames.ndim == 4 and frames.shape[1] == 3:
                    frames = frames.transpose(0, 2, 3, 1)

                if audio_chunk is not None:
                    frame_counter += len(audio_chunk) / sample_rate * tgt_fps
                else:
                    frame_counter = audio_end_idx

                _send({'status': 'ok', 'frames': frames})
            except Exception as exc:
                print(f'[soulx_worker] infer error: {exc}', file=sys.stderr, flush=True)
                _send({'status': 'error', 'error': str(exc), 'frames': None})

        else:
            _send({'status': 'error', 'error': f'unknown type: {msg_type}'})


if __name__ == '__main__':
    main()
