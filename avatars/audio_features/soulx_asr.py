import numpy as np
from avatars.audio_features.base_asr import BaseASR


class SoulXASR(BaseASR):
    """Raw-PCM accumulator for SoulX — no Whisper feature extraction.

    Puts a flat float32 numpy array (batch_size*2 chunks concatenated) into
    feat_queue so SoulXAvatar.inference() can forward it to the worker process.
    """

    def run_step(self):
        for _ in range(self.batch_size * 2):
            audio_frame = self.get_audio_frame()
            self.frames.append(audio_frame.data)
            self.output_queue.put(audio_frame)

        if len(self.frames) <= self.stride_left_size + self.stride_right_size:
            return

        raw_pcm = np.concatenate(self.frames[-self.batch_size * 2:])
        self.feat_queue.put(raw_pcm)

        self.frames = self.frames[-(self.stride_left_size + self.stride_right_size):]
