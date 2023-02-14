import os

import librosa
import numpy as np
import torch

from audio_zen.constant import EPSILON


def build_ideal_ratio_mask(noisy_mag, clean_mag) -> torch.Tensor:
    """

    Args:
        noisy_mag: [B, F, T], noisy magnitude
        clean_mag: [B, F, T], clean magnitude

    Returns:
        [B, F, T, 1]
    """
    # noisy_mag_finetune = torch.sqrt(torch.square(noisy_mag) + EPSILON)
    # ratio_mask = clean_mag / noisy_mag_finetune
    ratio_mask = clean_mag / (noisy_mag +  EPSILON)
    ratio_mask = ratio_mask[..., None]
    return compress_cIRM(ratio_mask, K=10, C=0.1)


def build_complex_ideal_ratio_mask(noisy: torch.complex64, clean: torch.complex64) -> torch.Tensor:
    """

    Args:
        noisy: [B, F, T], noisy complex-valued stft coefficients
        clean: [B, F, T], clean complex-valued stft coefficients

    Returns:
        [B, F, T, 2]
    """
    denominator = torch.square(noisy.real) + torch.square(noisy.imag) + EPSILON

    mask_real = (noisy.real * clean.real + noisy.imag * clean.imag) / denominator
    mask_imag = (noisy.real * clean.imag - noisy.imag * clean.real) / denominator

    complex_ratio_mask = torch.stack((mask_real, mask_imag), dim=-1)

    return compress_cIRM(complex_ratio_mask, K=10, C=0.1)


def compress_cIRM(mask, K=10, C=0.1):
    """
        Compress from (-inf, +inf) to [-K ~ K]
    """
    if torch.is_tensor(mask):
        mask = -100 * (mask <= -100) + mask * (mask > -100)
        mask = K * (1 - torch.exp(-C * mask)) / (1 + torch.exp(-C * mask))
    else:
        mask = -100 * (mask <= -100) + mask * (mask > -100)
        mask = K * (1 - np.exp(-C * mask)) / (1 + np.exp(-C * mask))
    return mask


def decompress_cIRM(mask, K=10, limit=9.9):
    mask = limit * (mask >= limit) - limit * (mask <= -limit) + mask * (torch.abs(mask) < limit)
    mask = -K * torch.log((K - mask) / (K + mask))
    return mask


def complex_mul(noisy_r, noisy_i, mask_r, mask_i):
    r = noisy_r * mask_r - noisy_i * mask_i
    i = noisy_r * mask_i + noisy_i * mask_r
    return r, i
