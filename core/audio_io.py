import os

import numpy as np
import scipy.io.wavfile as wav

from . import sigproc


def get_default_wav_path():
    """
    返回当前项目默认音频 english.wav 的绝对路径。
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "english.wav"))


def select_mono_channel(sig, channel=0):
    """
    将输入波形统一为单通道。

    :param sig: 原始波形数组，可能是 (N,) 或 (N, C)
    :param channel: 多通道时选择的通道索引
    :returns: 单通道波形，shape (N,)
    """
    if sig.ndim == 1:
        return sig
    return sig[:, channel]


def read_wav(wav_path):
    """
    读取任意 wav 文件，并返回采样率与波形数组。

    :param wav_path: wav 文件路径（绝对路径或相对路径）
    :returns:
      - rate：采样率（samplerate），单位 Hz
      - sig：波形数组
    """
    rate, sig = wav.read(wav_path)
    sig = np.asarray(sig)
    return rate, sig


def read_english_wav():
    """
    读取当前目录下的 english.wav，并返回采样率与波形数组。

    返回值说明：
    - rate：采样率（samplerate），单位 Hz
    - sig：波形数组
      - 单声道时 shape 为 (N,)
      - 多通道时 shape 为 (N, C)，其中 C 为通道数
    """
    return read_wav(get_default_wav_path())


def read_wav_mono(wav_path, channel=0, dtype=np.float32):
    """
    统一读取入口：读取任意 wav 文件，并返回指定类型的单通道波形。

    :param wav_path: wav 文件路径（绝对路径或相对路径）
    :param channel: 多通道时选择的通道索引
    :param dtype: 输出单通道波形的数据类型，默认 float32
    :returns:
      - rate: 采样率
      - sig: 原始波形（可能多通道）
      - sig_mono: 单通道波形
    """
    rate, sig = read_wav(wav_path)
    sig_mono = select_mono_channel(sig, channel=channel).astype(dtype)
    return rate, sig, sig_mono


def read_english_wav_mono(channel=0, dtype=np.float32, wav_path=None):
    """
    统一读取入口：读取 english.wav，并返回指定类型的单通道波形。

    :param channel: 多通道时选择的通道索引
    :param dtype: 输出单通道波形的数据类型，默认 float32
    :param wav_path: 可选 wav 文件路径；不传时默认读取 english.wav
    :returns:
      - rate: 采样率
      - sig: 原始波形（可能多通道）
      - sig_mono: 单通道波形
    """
    target_path = wav_path or get_default_wav_path()
    return read_wav_mono(target_path, channel=channel, dtype=dtype)


def preemphasis_english_wav(coeff=0.97, channel=0, wav_path=None):
    """
    读取 english.wav 并执行预加重（Pre-emphasis）。

    预加重常用形式：
    y[0] = x[0]
    y[t] = x[t] - coeff * x[t-1]

    :param coeff: 预加重系数（pre-emphasis coefficient），常用 0.95~0.97
    :param channel: 若音频为多通道，选择第几个通道（从 0 开始）
    :param wav_path: 可选 wav 文件路径；不传时默认读取 english.wav
    :returns:
      - rate: 采样率
      - sig_mono: 选择后的单通道原始波形，shape (N,)
      - sig_pre: 预加重后的波形，shape (N,)
    """
    rate, _, sig_mono = read_english_wav_mono(channel=channel, dtype=np.float32, wav_path=wav_path)
    sig_pre = sigproc.preemphasis(sig_mono, coeff=coeff)
    return rate, sig_mono, sig_pre


if __name__ == "__main__":
    rate, sig = read_english_wav()
    print("采样率 samplerate (Hz):", rate)
    print("原始波形 数据类型（dtype）:", sig.dtype)
    print("原始波形 数组形状（shape）:", sig.shape)

    coeff = 0.97
    _, _, sig_pre = preemphasis_english_wav(coeff=coeff, channel=0)

    print("预加重系数（coeff）:", coeff)
    print("预加重后 数据类型（dtype）:", sig_pre.dtype)
    print("预加重后 数组形状（shape）:", sig_pre.shape)
