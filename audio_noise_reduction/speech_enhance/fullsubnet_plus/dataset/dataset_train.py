import random

import numpy as np
from audio_zen.acoustics.feature import norm_amplitude, tailor_dB_FS, is_clipped, load_wav, subsample
from audio_zen.dataset.base_dataset import BaseDataset
from audio_zen.utils import expand_path
from joblib import Parallel, delayed
from scipy import signal
from tqdm import tqdm


class Dataset(BaseDataset):
    def __init__(self,
                 clean_dataset,
                 clean_dataset_limit,
                 clean_dataset_offset,
                 noise_dataset,
                 noise_dataset_limit,
                 noise_dataset_offset,
                 rir_dataset,
                 rir_dataset_limit,
                 rir_dataset_offset,
                 snr_range,
                 reverb_proportion,
                 silence_length,
                 target_dB_FS,
                 target_dB_FS_floating_value,
                 sub_sample_length,
                 sr,
                 pre_load_clean_dataset,
                 pre_load_noise,
                 pre_load_rir,
                 num_workers
                 ):
        """
        Dynamic mixing for training

        Args:
            clean_dataset_limit:
            clean_dataset_offset:
            noise_dataset_limit:
            noise_dataset_offset:
            rir_dataset:
            rir_dataset_limit:
            rir_dataset_offset:
            snr_range:
            reverb_proportion:
            clean_dataset: scp file
            noise_dataset: scp file
            sub_sample_length:
            sr:
        """
        super().__init__()
        # acoustics args
        self.sr = sr

        # parallel args
        self.num_workers = num_workers

        clean_dataset_list = [line.rstrip('\n') for line in open(expand_path(clean_dataset), "r")]
        noise_dataset_list = [line.rstrip('\n') for line in open(expand_path(noise_dataset), "r")]
        rir_dataset_list = [line.rstrip('\n') for line in open(expand_path(rir_dataset), "r")]

        clean_dataset_list = self._offset_and_limit(clean_dataset_list, clean_dataset_offset, clean_dataset_limit)
        noise_dataset_list = self._offset_and_limit(noise_dataset_list, noise_dataset_offset, noise_dataset_limit)
        rir_dataset_list = self._offset_and_limit(rir_dataset_list, rir_dataset_offset, rir_dataset_limit)

        if pre_load_clean_dataset:
            clean_dataset_list = self._preload_dataset(clean_dataset_list, remark="Clean Dataset")

        if pre_load_noise:
            noise_dataset_list = self._preload_dataset(noise_dataset_list, remark="Noise Dataset")

        if pre_load_rir:
            rir_dataset_list = self._preload_dataset(rir_dataset_list, remark="RIR Dataset")

        self.clean_dataset_list = clean_dataset_list
        self.noise_dataset_list = noise_dataset_list
        self.rir_dataset_list = rir_dataset_list

        snr_list = self._parse_snr_range(snr_range)
        self.snr_list = snr_list

        assert 0 <= reverb_proportion <= 1, "reverberation proportion should be in [0, 1]"
        self.reverb_proportion = reverb_proportion
        self.silence_length = silence_length
        self.target_dB_FS = target_dB_FS
        self.target_dB_FS_floating_value = target_dB_FS_floating_value
        self.sub_sample_length = sub_sample_length

        self.length = len(self.clean_dataset_list)

    def __len__(self):
        return self.length

    def _preload_dataset(self, file_path_list, remark=""):
        waveform_list = Parallel(n_jobs=self.num_workers)(
            delayed(load_wav)(f_path) for f_path in tqdm(file_path_list, desc=remark)
        )
        return list(zip(file_path_list, waveform_list))

    @staticmethod
    def _random_select_from(dataset_list):
        return random.choice(dataset_list)

    def _select_noise_y(self, target_length):
        noise_y = np.zeros(0, dtype=np.float32)
        silence = np.zeros(int(self.sr * self.silence_length), dtype=np.float32)
        remaining_length = target_length

        while remaining_length > 0:
            noise_file = self._random_select_from(self.noise_dataset_list)
            noise_new_added = load_wav(noise_file, sr=self.sr)
            noise_y = np.append(noise_y, noise_new_added)
            remaining_length -= len(noise_new_added)

            # 如果还需要添加新的噪声，就插入一个小静音段
            if remaining_length > 0:
                silence_len = min(remaining_length, len(silence))
                noise_y = np.append(noise_y, silence[:silence_len])
                remaining_length -= silence_len

        if len(noise_y) > target_length:
            idx_start = np.random.randint(len(noise_y) - target_length)
            noise_y = noise_y[idx_start:idx_start + target_length]

        return noise_y

    @staticmethod
    def snr_mix(clean_y, noise_y, snr, target_dB_FS, target_dB_FS_floating_value, rir=None, eps=1e-6):
        """
        混合噪声与纯净语音，当 rir 参数不为空时，对纯净语音施加混响效果

        Args:
            clean_y: 纯净语音
            noise_y: 噪声
            snr (int): 信噪比
            target_dB_FS (int):
            target_dB_FS_floating_value (int):
            rir: room impulse response, None 或 np.array
            eps: eps

        Returns:
            (noisy_y，clean_y)
        """
        if rir is not None:
            if rir.ndim > 1:
                rir_idx = np.random.randint(0, rir.shape[0])
                rir = rir[rir_idx, :]

            clean_y = signal.fftconvolve(clean_y, rir)[:len(clean_y)]

        clean_y, _ = norm_amplitude(clean_y)
        clean_y, _, _ = tailor_dB_FS(clean_y, target_dB_FS)
        clean_rms = (clean_y ** 2).mean() ** 0.5

        noise_y, _ = norm_amplitude(noise_y)
        noise_y, _, _ = tailor_dB_FS(noise_y, target_dB_FS)
        noise_rms = (noise_y ** 2).mean() ** 0.5

        snr_scalar = clean_rms / (10 ** (snr / 20)) / (noise_rms + eps)
        noise_y *= snr_scalar
        noisy_y = clean_y + noise_y

        # Randomly select RMS value of dBFS between -15 dBFS and -35 dBFS and normalize noisy speech with that value
        noisy_target_dB_FS = np.random.randint(
            target_dB_FS - target_dB_FS_floating_value,
            target_dB_FS + target_dB_FS_floating_value
        )

        # 使用 noisy 的 rms 放缩音频
        noisy_y, _, noisy_scalar = tailor_dB_FS(noisy_y, noisy_target_dB_FS)
        clean_y *= noisy_scalar

        # 合成带噪语音的时候可能会 clipping，虽然极少
        # 对 noisy, clean_y, noise_y 稍微进行调整
        if is_clipped(noisy_y):
            noisy_y_scalar = np.max(np.abs(noisy_y)) / (0.99 - eps)  # 相当于除以 1
            noisy_y = noisy_y / noisy_y_scalar
            clean_y = clean_y / noisy_y_scalar

        return noisy_y, clean_y

    def __getitem__(self, item):
        clean_file = self.clean_dataset_list[item]
        clean_y = load_wav(clean_file, sr=self.sr)
        clean_y = subsample(clean_y, sub_sample_length=int(self.sub_sample_length * self.sr))

        noise_y = self._select_noise_y(target_length=len(clean_y))
        assert len(clean_y) == len(noise_y), f"Inequality: {len(clean_y)} {len(noise_y)}"

        snr = self._random_select_from(self.snr_list)
        use_reverb = bool(np.random.random(1) < self.reverb_proportion)

        noisy_y, clean_y = self.snr_mix(
            clean_y=clean_y,
            noise_y=noise_y,
            snr=snr,
            target_dB_FS=self.target_dB_FS,
            target_dB_FS_floating_value=self.target_dB_FS_floating_value,
            rir=load_wav(self._random_select_from(self.rir_dataset_list), sr=self.sr) if use_reverb else None
        )

        noisy_y = noisy_y.astype(np.float32)
        clean_y = clean_y.astype(np.float32)

        return noisy_y, clean_y
