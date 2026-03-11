# 本文件包含基础信号处理（signal processing）例程，包括分帧（framing）与功率谱（power spectra）计算等。
# 作者（Author）：James Lyons，2012
import decimal

import numpy
import math
import logging


def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))


def framesig(sig, frame_len, frame_step, winfunc=numpy.hamming):
    """将信号切分为重叠帧（overlapping frames）。

    :param sig: 需要分帧的音频信号（audio signal）。
    :param frame_len: 每帧长度（frame length），单位为采样点（samples）。
    :param frame_step: 帧移（frame step），即下一帧相对上一帧起点偏移的采样点数。
    :param winfunc: 对每帧应用的窗函数（analysis window）。默认使用 Hamming 窗。
    :returns: 帧数组（frames），大小为 NUMFRAMES * frame_len。
    """
    sig = numpy.asarray(sig)
    slen = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))

    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0 * slen - frame_len) / frame_step))

    padlen = int((numframes - 1) * frame_step + frame_len)
    # 先创建补零后的信号，长度为 padlen
    padsignal = numpy.zeros((padlen,), dtype=sig.dtype)
    # 把原始信号逐点拷贝到 padsignal 前半段
    # 这样写法更直观，逻辑与切片赋值一致
    for i in range(slen):
        padsignal[i] = sig[i]

    # frames 的每一行就是一帧
    frames = numpy.zeros((numframes, frame_len), dtype=padsignal.dtype)
    for i in range(numframes):
        start = i * frame_step
        end = start + frame_len
        # 取出从 start 到 end 的一段，作为第 i 帧
        frames[i, :] = padsignal[start:end]

    # 获取窗函数（例如全 1 窗、Hamming 窗）
    win = numpy.asarray(winfunc(frame_len))
    # 逐点把每一帧乘以窗函数，便于不熟悉广播语法的读者理解
    windowed_frames = numpy.zeros((numframes, frame_len), dtype=numpy.float64)
    for i in range(numframes):
        for j in range(frame_len):
            windowed_frames[i, j] = frames[i, j] * win[j]

    return windowed_frames


def deframesig(frames, siglen, frame_len, frame_step, winfunc=numpy.hamming):
    """通过 overlap-add（重叠相加）过程撤销 framesig 的分帧效果（近似还原原始信号）。

    :param frames: 帧数组（frames）。
    :param siglen: 目标信号长度（desired signal length）；未知可传 0。输出将截断到 siglen 个采样点。
    :param frame_len: 每帧长度（frame length），单位为采样点（samples）。
    :param frame_step: 帧移（frame step），即下一帧相对上一帧起点偏移的采样点数。
    :param winfunc: 对每帧应用的窗函数（analysis window）。默认使用 Hamming 窗。
    :returns: 一维信号（1-D signal）。
    """
    frame_len = round_half_up(frame_len)
    frame_step = round_half_up(frame_step)
    numframes = numpy.shape(frames)[0]
    assert numpy.shape(frames)[1] == frame_len, '"frames" matrix is wrong size, 2nd dim is not equal to frame_len'

    indices = numpy.tile(numpy.arange(0, frame_len), (numframes, 1)) + numpy.tile(
        numpy.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
    indices = numpy.array(indices, dtype=numpy.int32)
    padlen = (numframes - 1) * frame_step + frame_len

    if siglen <= 0: siglen = padlen

    rec_signal = numpy.zeros((padlen,))
    window_correction = numpy.zeros((padlen,))
    win = winfunc(frame_len)

    for i in range(0, numframes):
        window_correction[indices[i, :]] = window_correction[
                                               indices[i, :]] + win + 1e-15  # 加一个很小的值，避免出现 0
        rec_signal[indices[i, :]] = rec_signal[indices[i, :]] + frames[i, :]

    rec_signal = rec_signal / window_correction
    return rec_signal[0:siglen]


def magspec(frames, NFFT):
    """计算每一帧的幅度谱（magnitude spectrum）。

    若 frames 是 N×D 矩阵，则输出为 N×(NFFT/2+1)。

    :param frames: 帧数组（frames），每一行是一帧。
    :param NFFT: FFT（Fast Fourier Transform，快速傅里叶变换）长度（点数）。若 NFFT > frame_len，将对帧进行 zero-padding（零填充）。
    :returns: 若 frames 为 N×D，则输出为 N×(NFFT/2+1)，每一行对应一帧的幅度谱。
    """
    frames = numpy.asarray(frames)
    if numpy.shape(frames)[1] > NFFT:
        logging.warn(
            'frame length（帧长，%d）大于 FFT size（FFT 点数，%d），帧将被截断。可增大 NFFT 以避免截断。',
            numpy.shape(frames)[1], NFFT)

    numframes = numpy.shape(frames)[0]
    nfreq = NFFT // 2 + 1
    # 幅度谱输出矩阵：每一行对应一帧，每一列对应一个频率点
    magnitude = numpy.zeros((numframes, nfreq), dtype=numpy.float64)

    for i in range(numframes):
        # 先做一维 rFFT，把时域帧转为频域复数谱
        complex_spec = numpy.fft.rfft(frames[i], NFFT)
        for k in range(nfreq):
            # 复数取绝对值，得到该频率点的幅值
            magnitude[i, k] = abs(complex_spec[k])

    return magnitude


def powspec(frames, NFFT):
    """计算每一帧的功率谱（power spectrum）。

    若 frames 是 N×D 矩阵，则输出为 N×(NFFT/2+1)。

    :param frames: 帧数组（frames），每一行是一帧。
    :param NFFT: FFT（快速傅里叶变换）长度（点数）。若 NFFT > frame_len，将对帧进行 zero-padding（零填充）。
    :returns: 若 frames 为 N×D，则输出为 N×(NFFT/2+1)，每一行对应一帧的功率谱。
    """
    mag = magspec(frames, NFFT)
    numframes = numpy.shape(mag)[0]
    nfreq = numpy.shape(mag)[1]
    # 功率谱输出矩阵：和幅度谱维度相同
    power = numpy.zeros((numframes, nfreq), dtype=numpy.float64)

    for i in range(numframes):
        for k in range(nfreq):
            # 功率 = 幅值平方 / NFFT
            power[i, k] = (mag[i, k] * mag[i, k]) / NFFT

    return power


def logpowspec(frames, NFFT, norm=1):
    """计算每一帧的对数功率谱（log power spectrum）。

    若 frames 是 N×D 矩阵，则输出为 N×(NFFT/2+1)。

    :param frames: 帧数组（frames），每一行是一帧。
    :param NFFT: FFT（快速傅里叶变换）长度（点数）。若 NFFT > frame_len，将对帧进行 zero-padding（零填充）。
    :param norm: 若 norm=1，则对数功率谱会做归一化（normalised），使所有帧中的最大值为 0。
    :returns: 若 frames 为 N×D，则输出为 N×(NFFT/2+1)，每一行对应一帧的对数功率谱。
    """
    ps = powspec(frames, NFFT);
    ps[ps <= 1e-30] = 1e-30
    lps = 10 * numpy.log10(ps)
    if norm:
        return lps - numpy.max(lps)
    else:
        return lps


def remove_dc_offset(signal):
    """执行直流分量去除（DC offset removal）。

    :param signal: 输入一维波形。
    :returns: 去除直流分量后的波形。
    """
    signal = numpy.asarray(signal)
    if signal.size == 0:
        return signal
    return signal - numpy.mean(signal)


def add_dither(signal, dither_scale=1e-5, seed=0):
    """执行抖动（dither），为波形加入极小幅随机噪声。

    :param signal: 输入一维波形。
    :param dither_scale: 抖动强度，默认 1e-5。
    :param seed: 随机种子，默认 0，保证可复现。
    :returns: 加入抖动后的波形。
    """
    signal = numpy.asarray(signal)
    if signal.size == 0:
        return signal
    rng = numpy.random.RandomState(seed)
    noise = rng.normal(loc=0.0, scale=dither_scale, size=signal.shape)
    return signal + noise


def simple_energy_vad_mask(energy, threshold_db=40.0, min_energy=1e-8):
    """基于帧能量的简单 VAD，返回保留语音帧的布尔掩码。

    规则：保留能量高于“最大帧能量 - threshold_db”的帧。

    :param energy: 每帧能量数组。
    :param threshold_db: 相对阈值（dB）。
    :param min_energy: 数值下限，避免 log 计算异常。
    :returns: 布尔掩码，True 表示语音帧。
    """
    energy = numpy.asarray(energy, dtype=numpy.float64)
    if energy.size == 0:
        return numpy.zeros((0,), dtype=bool)
    safe_energy = numpy.maximum(energy, min_energy)
    log_energy_db = 10.0 * numpy.log10(safe_energy)
    threshold = numpy.max(log_energy_db) - threshold_db
    mask = log_energy_db >= threshold
    if not numpy.any(mask):
        mask[numpy.argmax(log_energy_db)] = True
    return mask


def cmvn(feat, eps=1e-8):
    """执行逐句 CMVN（倒谱均值方差归一化）。

    :param feat: 特征矩阵，形状为 (num_frames, feat_dim)。
    :param eps: 方差下限，避免除零。
    :returns: CMVN 后的特征矩阵。
    """
    feat = numpy.asarray(feat, dtype=numpy.float64)
    if feat.size == 0:
        return feat
    mean = numpy.mean(feat, axis=0, keepdims=True)
    std = numpy.std(feat, axis=0, keepdims=True)
    std = numpy.maximum(std, eps)
    return (feat - mean) / std


def preemphasis(signal, coeff=0.95):
    """对输入信号执行预加重（pre-emphasis）处理。

    :param signal: 需要滤波的信号（signal）。
    :param coeff: 预加重系数（preemphasis coefficient）。0 表示不滤波，默认 0.95。
    :returns: 滤波后的信号（filtered signal）。
    """

##############################################################################################
#简写


    #把 signal 统一转成 NumPy 数组 （如果本来就是数组，一般不会复制，直接返回同一个视图/对象）。
    signal = numpy.asarray(signal)
    # 为了尽量保持与原实现一致：
    # - 浮点输入保持原浮点类型（例如 float32 -> float32）
    # - 整型输入提升为 float64，避免小数被截断
    if numpy.issubdtype(signal.dtype, numpy.floating):
        out_dtype = signal.dtype
    else:
        out_dtype = numpy.float64
    filtered = numpy.zeros_like(signal, dtype=out_dtype)

    # 第一个采样点没有前一个点，直接保留原值
    filtered[0] = signal[0]

    # 从第二个采样点开始，按 y[t] = x[t] - coeff * x[t-1] 逐点计算
    for t in range(1, len(signal)):
        filtered[t] = signal[t] - coeff * signal[t - 1]

    return filtered
