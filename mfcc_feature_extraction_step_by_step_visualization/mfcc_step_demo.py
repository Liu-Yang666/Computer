import argparse
import os
import sys

import matplotlib
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core import sigproc
from core.audio_io import read_english_wav_mono

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def hz2mel(hz):
    """
    将赫兹频率（Hz）转换为梅尔频率（Mel）。
    """
    return 2595 * np.log10(1 + hz / 700.0)


def mel2hz(mel):
    """
    将梅尔频率（Mel）转换为赫兹频率（Hz）。
    """
    return 700 * (10 ** (mel / 2595.0) - 1)


def calculate_nfft(samplerate, winlen):
    """
    计算不小于窗长采样点数的最小 2 的幂，作为 NFFT。
    """
    window_length_samples = winlen * samplerate
    nfft = 1
    while nfft < window_length_samples:
        nfft *= 2
    return nfft


def get_filterbanks(nfilt, nfft, samplerate, lowfreq=0, highfreq=None):
    """
    构造梅尔滤波器组（Mel Filter Bank）。

    返回矩阵形状为 (nfilt, nfft//2 + 1)。
    """
    highfreq = highfreq or samplerate / 2
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel, highmel, nfilt + 2)
    bins = np.floor((nfft + 1) * mel2hz(melpoints) / samplerate).astype(int)

    fbank = np.zeros((nfilt, nfft // 2 + 1))
    for j in range(nfilt):
        for i in range(bins[j], bins[j + 1]):
            fbank[j, i] = (i - bins[j]) / (bins[j + 1] - bins[j] + 1e-12)
        for i in range(bins[j + 1], bins[j + 2]):
            fbank[j, i] = (bins[j + 2] - i) / (bins[j + 2] - bins[j + 1] + 1e-12)
    return fbank


def save_plot(fig, output_dir, filename):
    """
    保存图像到输出目录，并释放图像对象。
    """
    path = os.path.join(output_dir, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"已保存图像: {path}")


def main(wav_path=None):
    """
    分步骤演示并绘制：
    1) 读取音频与原始波形
    2) 预加重（Pre-emphasis）
    3) 分帧（Framing）
    4) 加窗（Windowing）
    5) FFT
    6) 周期图（Periodogram）
    7) 梅尔滤波器组（Mel Filter Bank）与梅尔能量
    """
    base_dir = os.path.dirname(__file__)
    wav_path = wav_path or os.path.join(PROJECT_ROOT, "data", "english.wav")
    output_dir = os.path.join(base_dir, "mfcc_step_plots")
    os.makedirs(output_dir, exist_ok=True)

    winlen = 0.025
    winstep = 0.01
    preemph_coeff = 0.97
    nfilt = 26
    lowfreq = 20

    # 统一读取入口：读取 english.wav，并返回原始波形与单通道波形
    rate, sig, sig_mono = read_english_wav_mono(channel=0, dtype=np.float32, wav_path=wav_path)
    time_axis = np.arange(sig_mono.shape[0]) / rate

    print("\n========== 维度信息 ==========")
    print(f"原始波形 数据类型（dtype）: {sig.dtype}")
    print(f"原始波形 数组形状（shape）: {sig.shape}")
    print(f"单通道波形 数据类型（dtype）: {sig_mono.dtype}")
    print(f"单通道波形 数组形状（shape）: {sig_mono.shape}")

    fig = plt.figure(figsize=(10, 3))
    plt.plot(time_axis, sig_mono, linewidth=0.8)
    plt.title("Step1: Original Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    save_plot(fig, output_dir, "step1_waveform_原始音频时域波形.png")

    sig_pre = sigproc.preemphasis(sig_mono, coeff=preemph_coeff)
    print(f"预加重后 数据类型（dtype）: {sig_pre.dtype}")
    print(f"预加重后 数组形状（shape）: {sig_pre.shape}")

    fig = plt.figure(figsize=(10, 3))
    plt.plot(time_axis, sig_pre, linewidth=0.8)
    plt.title(f"Step2: Pre-emphasis Waveform (coeff={preemph_coeff})")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    save_plot(fig, output_dir, "step2_preemphasis_预加重后的时域波形.png")

    frame_len = int(round(winlen * rate))
    frame_step = int(round(winstep * rate))
    nfft = calculate_nfft(rate, winlen)
    print(f"每帧长度（frame_len）: {frame_len}")
    print(f"帧移（frame_step）: {frame_step}")
    print(f"FFT点数（NFFT）: {nfft}")

    frames = sigproc.framesig(
        sig_pre,
        frame_len=frame_len,
        frame_step=frame_step,
        winfunc=lambda x: np.ones((x,))
    )
    print(f"分帧后 数组形状（shape）: {frames.shape}")

    fig = plt.figure(figsize=(10, 4))
    plt.imshow(frames.T, origin="lower", aspect="auto", cmap="viridis")
    plt.title("Step3: Framing")
    plt.xlabel("Frame Index")
    plt.ylabel("Sample Index in Frame")
    plt.colorbar(label="Amplitude")
    save_plot(fig, output_dir, "step3_framing_分帧结果热力图.png")

    window = np.hamming(frame_len)
    windowed_frames = frames * window
    print(f"窗函数 数组形状（shape）: {window.shape}")
    print(f"加窗后帧 数据类型（dtype）: {windowed_frames.dtype}")
    print(f"加窗后帧 数组形状（shape）: {windowed_frames.shape}")

    fig = plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(frames[0], linewidth=0.8)
    plt.title("Step4: Frame-1 Before Windowing")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.subplot(2, 1, 2)
    plt.plot(windowed_frames[0], linewidth=0.8)
    plt.title("Step4: Frame-1 After Hamming Window")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    save_plot(fig, output_dir, "step4_windowing_第1帧加窗前后对比.png")

    fft_complex = np.fft.rfft(windowed_frames, n=nfft, axis=1)
    fft_magnitude = np.abs(fft_complex)
    freq_axis = np.linspace(0, rate / 2, fft_magnitude.shape[1])
    print(f"FFT复数谱 数据类型（dtype）: {fft_complex.dtype}")
    print(f"FFT复数谱 数组形状（shape）: {fft_complex.shape}")
    print(f"FFT幅度谱 数据类型（dtype）: {fft_magnitude.dtype}")
    print(f"FFT幅度谱 数组形状（shape）: {fft_magnitude.shape}")

    fig = plt.figure(figsize=(10, 3.5))
    plt.plot(freq_axis, fft_magnitude[0], linewidth=0.8)
    plt.title("Step5: Frame-1 FFT Magnitude Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    save_plot(fig, output_dir, "step5_fft_第1帧FFT幅度谱.png")

    periodogram = (1.0 / nfft) * (fft_magnitude ** 2)
    print(f"周期图 数据类型（dtype）: {periodogram.dtype}")
    print(f"周期图 数组形状（shape）: {periodogram.shape}")

    fig = plt.figure(figsize=(10, 3.5))
    plt.plot(freq_axis, periodogram[0], linewidth=0.8)
    plt.title("Step6: Frame-1 Periodogram")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    save_plot(fig, output_dir, "step6_periodogram_第1帧周期图功率谱.png")

    mel_filters = get_filterbanks(
        nfilt=nfilt,
        nfft=nfft,
        samplerate=rate,
        lowfreq=lowfreq,
        highfreq=rate / 2
    )
    mel_energies = np.dot(periodogram, mel_filters.T)
    mel_energies = np.where(mel_energies == 0, np.finfo(float).eps, mel_energies)
    print(f"梅尔滤波器组 数组形状（shape）: {mel_filters.shape}")
    print(f"梅尔滤波器组能量 数据类型（dtype）: {mel_energies.dtype}")
    print(f"梅尔滤波器组能量 数组形状（shape）: {mel_energies.shape}")
    print("========== 维度信息结束 ==========\n")

    fig = plt.figure(figsize=(10, 4))
    for i in range(mel_filters.shape[0]):
        plt.plot(freq_axis, mel_filters[i], linewidth=0.8)
    plt.title("Step7: Mel Filter Bank")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Filter Weight")
    save_plot(fig, output_dir, "step7_mel_filterbank_梅尔滤波器组曲线图.png")

    fig = plt.figure(figsize=(10, 4))
    plt.imshow(np.log(mel_energies).T, origin="lower", aspect="auto", cmap="magma")
    plt.title("Step7: Log Mel Filter-Bank Energy")
    plt.xlabel("Frame Index")
    plt.ylabel("Filter Index")
    plt.colorbar(label="Log Energy")
    save_plot(fig, output_dir, "step7_log_mel_energy_对数梅尔能量热力图.png")

    print(f"全部步骤绘图已完成，输出目录：{output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", dest="wav_path", default=None)
    args = parser.parse_args()
    main(wav_path=args.wav_path)
