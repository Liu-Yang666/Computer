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


def fft_db(frame, nfft, fs, normalize_peak=False):
    """
    对输入帧做 rFFT，并返回频率轴和 dB 幅度谱。

    normalize_peak=True 时，会把谱峰值归一化到 0 dB，
    便于只比较谱形状（主瓣宽度、旁瓣泄漏），弱化绝对幅值差异。
    """
    spec = np.fft.rfft(frame, n=nfft)
    mag = np.abs(spec)
    db = 20 * np.log10(mag + 1e-12)
    if normalize_peak:
        db = db - np.max(db)
    freq = np.fft.rfftfreq(nfft, d=1.0 / fs)
    return freq, db


def fft_magnitude(frame, nfft, fs):
    """
    对输入帧做 rFFT，并返回频率轴和线性幅度谱（不取对数）。
    """
    spec = np.fft.rfft(frame, n=nfft)
    mag = np.abs(spec)
    freq = np.fft.rfftfreq(nfft, d=1.0 / fs)
    return freq, mag


def apply_hamming_with_gain_compensation(frame):
    """
    对一帧信号加汉明窗，并做相干增益补偿。

    说明：
    - 直接乘汉明窗会降低谱峰绝对幅值；
    - 用窗均值做补偿后，可更公平地和“不加汉明窗”对比。
    """
    win_hamming = np.hamming(len(frame))
    coherent_gain = np.mean(win_hamming)
    return frame * win_hamming / coherent_gain


def main():
    # 输出目录：保存合成信号4图对比和语音帧2图对比
    base_dir = os.path.dirname(__file__)
    wav_path = os.path.join(PROJECT_ROOT, "data", "english.wav")
    output_dir = os.path.join(base_dir, "window_fft_compare_plots")
    os.makedirs(output_dir, exist_ok=True)

    # 合成信号参数：
    # - fs: 采样率
    # - frame_len: 帧长
    # - nfft: FFT点数
    fs = 8000
    nfft = 256
    frame_len = 256
    n = np.arange(frame_len)

    # 频率设计：
    # - f_integer: 整数周期频率（与 FFT bin 对齐）
    # - f_non_integer: 非整数周期频率（故意偏离 FFT bin）
    k = 20
    f_integer = k * fs / frame_len
    f_non_integer = (k + 0.35) * fs / frame_len

    # 单频正弦信号
    x_integer = np.sin(2 * np.pi * f_integer * n / fs)
    x_non_integer = np.sin(2 * np.pi * f_non_integer * n / fs)

    # “不加汉明窗”就是全 1 权重
    win_rect = np.ones(frame_len)

    # 为了让整数周期场景下的对比更符合理论直觉：
    # 1) 汉明窗做增益补偿（避免纯粹因窗均值导致峰值整体偏低）
    # 2) 两路都做峰值归一化（峰值归一到 0 dB，只比较谱形状）
    x_integer_ham = apply_hamming_with_gain_compensation(x_integer)
    x_non_integer_ham = apply_hamming_with_gain_compensation(x_non_integer)

    freq_int_rect, db_int_rect = fft_db(x_integer * win_rect, nfft, fs, normalize_peak=True)
    freq_int_ham, db_int_ham = fft_db(x_integer_ham, nfft, fs, normalize_peak=True)
    freq_non_rect, db_non_rect = fft_db(x_non_integer * win_rect, nfft, fs, normalize_peak=True)
    freq_non_ham, db_non_ham = fft_db(x_non_integer_ham, nfft, fs, normalize_peak=True)
    # 这三条用于“新增两张图”，按你的要求使用对数纵轴，但不做峰值归一化
    freq_int_rect_raw, db_int_rect_raw = fft_db(x_integer * win_rect, nfft, fs, normalize_peak=False)
    freq_non_rect_raw, db_non_rect_raw = fft_db(x_non_integer * win_rect, nfft, fs, normalize_peak=False)
    freq_non_ham_raw, db_non_ham_raw = fft_db(x_non_integer_ham, nfft, fs, normalize_peak=False)

    # 4图对比：整数/非整数周期 × 不加汉明窗/加汉明窗
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(freq_int_rect, db_int_rect)
    plt.title("Integer-cycle sine + No Hamming window")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(freq_int_ham, db_int_ham, color="orange")
    plt.title("Integer-cycle sine + Hamming window")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(freq_non_rect, db_non_rect)
    plt.title("Non-integer-cycle sine + No Hamming window")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot(freq_non_ham, db_non_ham, color="orange")
    plt.title("Non-integer-cycle sine + Hamming window")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    synthetic_path = os.path.join(output_dir, "synthetic_4_fft_compare.png")
    plt.savefig(synthetic_path, dpi=150)
    plt.close()

    # 额外图1：整数周期不加汉明窗 vs 非整数周期不加汉明窗
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(freq_int_rect_raw, db_int_rect_raw)
    plt.title("Integer-cycle sine + No Hamming window")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(freq_non_rect_raw, db_non_rect_raw)
    plt.title("Non-integer-cycle sine + No Hamming window")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    no_hamming_compare_path = os.path.join(output_dir, "compare_integer_vs_noninteger_no_hamming.png")
    plt.savefig(no_hamming_compare_path, dpi=150)
    plt.close()

    # 额外图2：非整数周期不加汉明窗 vs 非整数周期加汉明窗
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(freq_non_rect_raw, db_non_rect_raw)
    plt.title("Non-integer-cycle sine + No Hamming window")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(freq_non_ham_raw, db_non_ham_raw, color="orange")
    plt.title("Non-integer-cycle sine + Hamming window")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    noninteger_window_compare_path = os.path.join(output_dir, "compare_noninteger_nohamming_vs_hamming.png")
    plt.savefig(noninteger_window_compare_path, dpi=150)
    plt.close()

    # 统一读取入口：读取 english.wav 并转为单通道 float32
    rate, _, sig = read_english_wav_mono(channel=0, dtype=np.float32, wav_path=wav_path)

    # 语音帧参数：与合成实验一致使用 256 点帧长
    winlen = frame_len / rate
    winstep = 0.01

    # 分帧：
    # - 不加汉明窗：全1窗
    # - 加汉明窗：np.hamming
    frames_rect = sigproc.framesig(sig, frame_len=winlen * rate, frame_step=winstep * rate, winfunc=lambda x: np.ones((x,)))
    frames_ham = sigproc.framesig(sig, frame_len=winlen * rate, frame_step=winstep * rate, winfunc=np.hamming)

    # 取中间一帧做展示（避免开头/结尾静音段影响）
    frame_idx = len(frames_rect) // 2
    frame_rect = frames_rect[frame_idx]
    frame_ham = frames_ham[frame_idx]

    # 语音帧也做峰值归一化，方便看窗函数带来的谱形差异
    freq_s_rect, db_s_rect = fft_db(frame_rect, nfft, rate, normalize_peak=True)
    freq_s_ham, db_s_ham = fft_db(frame_ham, nfft, rate, normalize_peak=True)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(freq_s_rect, db_s_rect)
    plt.title(f"Speech frame {frame_idx} + No Hamming window")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(freq_s_ham, db_s_ham, color="orange")
    plt.title(f"Speech frame {frame_idx} + Hamming window")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    speech_path = os.path.join(output_dir, "speech_frame_fft_compare.png")
    plt.savefig(speech_path, dpi=150)
    plt.close()

    print("========== 参数与维度 ==========")
    print("合成信号采样率 fs:", fs)
    print("帧长 frame_len:", frame_len)
    print("NFFT:", nfft)
    print("整数周期频率 f_integer(Hz):", f_integer)
    print("非整数周期频率 f_non_integer(Hz):", f_non_integer)
    print("汉明窗相干增益（均值）:", np.mean(np.hamming(frame_len)))
    print("x_integer shape:", x_integer.shape)
    print("x_non_integer shape:", x_non_integer.shape)
    print("英语语音采样率 rate:", rate)
    print("语音一维波形 shape:", sig.shape)
    print("分帧后（不加窗）shape:", frames_rect.shape)
    print("分帧后（加窗）shape:", frames_ham.shape)
    print("选取语音帧索引 frame_idx:", frame_idx)
    print("语音帧（不加窗）shape:", frame_rect.shape)
    print("语音帧（加窗）shape:", frame_ham.shape)
    print("========== 输出图像 ==========")
    print(synthetic_path)
    print(no_hamming_compare_path)
    print(noninteger_window_compare_path)
    print(speech_path)


if __name__ == "__main__":
    main()
