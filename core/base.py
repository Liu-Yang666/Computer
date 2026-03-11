# 计算 filterbank（滤波器组）特征。提供例如 fbank 与 mfcc 特征，用于 ASR（Automatic Speech Recognition，自动语音识别）应用
# 作者（Author）：James Lyons，2012
from __future__ import division
import numpy
from . import sigproc
from scipy.fftpack import dct

def calculate_nfft(samplerate, winlen):
    """计算 FFT（Fast Fourier Transform，快速傅里叶变换）点数，取不小于单个窗口长度采样点数的 2 的幂。

    如果 FFT 点数小于窗口长度，会丢弃大量采样点从而降低精度；如果 FFT 点数大于窗口长度，则会对 FFT 缓冲区做 zero-padding（零填充），对频域变换结果通常是中性的。

    :param samplerate: 当前信号的采样率（sample rate，采样频率），单位 Hz（Hertz，赫兹）。
    :param winlen: 分析窗口长度（analysis window length），单位秒（s）。
    """
    window_length_samples = winlen * samplerate
    nfft = 1
    while nfft < window_length_samples:
        nfft *= 2
    return nfft

def mfcc(signal,samplerate=16000,winlen=0.025,winstep=0.01,numcep=13,
         nfilt=26,nfft=None,lowfreq=20,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True,
         winfunc=numpy.hamming):
    """从音频信号计算 MFCC（Mel-Frequency Cepstral Coefficients，梅尔频率倒谱系数）特征。

    :param signal: 用于计算特征的音频信号（audio signal），应为 N*1 的数组。
    :param samplerate: 当前信号的采样率（sample rate，采样频率），单位 Hz（Hertz，赫兹）。
    :param winlen: 分析窗口长度（analysis window length），单位秒（s）。默认 0.025s（25ms）。
    :param winstep: 相邻窗口之间的步长（step），单位秒（s）。默认 0.01s（10ms）。
    :param numcep: 返回的 cepstrum（倒谱）系数数量，默认 13。
    :param nfilt: filterbank（滤波器组）中滤波器数量，默认 26。
    :param nfft: FFT（快速傅里叶变换）点数。默认 None，使用 calculate_nfft 选择一个不会丢弃样本数据的最小点数。
    :param lowfreq: mel filters（梅尔滤波器）最低频带边界，单位 Hz（赫兹），默认 20。
    :param highfreq: mel filters（梅尔滤波器）最高频带边界，单位 Hz（赫兹），默认 samplerate/2。
    :param preemph: 预加重（pre-emphasis）滤波器系数。0 表示不做滤波。默认 0.97。
    :param ceplifter: 对最终 cepstral coefficients（倒谱系数）做 lifter（倒谱提升/升倒谱）。0 表示不使用。默认 22。
    :param appendEnergy: 若为 True，则用每帧总能量（frame energy）的对数替换第 0 维倒谱系数。
    :param winfunc: 对每帧应用的窗函数（analysis window）。默认使用 Hamming 窗。
    :returns: numpy 数组，大小为（NUMFRAMES * numcep），每一行是一帧的特征向量（feature vector）。
    """
    nfft = nfft or calculate_nfft(samplerate, winlen)
    feat,energy = fbank(signal,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph,winfunc)
    feat = numpy.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[:,:numcep]
    feat = lifter(feat,ceplifter)
    if appendEnergy: feat[:,0] = numpy.log(energy) # replace first cepstral coefficient with log of frame energy
    feat = sigproc.cmvn(feat)
    return feat

def fbank(signal,samplerate=16000,winlen=0.025,winstep=0.01,
          nfilt=26,nfft=512,lowfreq=20,highfreq=None,preemph=0.97,
          winfunc=numpy.hamming):
    """从音频信号计算 Mel-filterbank（梅尔滤波器组）能量特征。

    :param signal: 用于计算特征的音频信号（audio signal），应为 N*1 的数组。
    :param samplerate: 当前信号的采样率（sample rate，采样频率），单位 Hz（Hertz，赫兹）。
    :param winlen: 分析窗口长度（analysis window length），单位秒（s）。默认 0.025s（25ms）。
    :param winstep: 相邻窗口之间的步长（step），单位秒（s）。默认 0.01s（10ms）。
    :param nfilt: filterbank（滤波器组）中滤波器数量，默认 26。
    :param nfft: FFT（快速傅里叶变换）点数。默认 512。
    :param lowfreq: mel filters（梅尔滤波器）最低频带边界，单位 Hz（赫兹），默认 20。
    :param highfreq: mel filters（梅尔滤波器）最高频带边界，单位 Hz（赫兹），默认 samplerate/2。
    :param preemph: 预加重（pre-emphasis）滤波器系数。0 表示不做滤波。默认 0.97。
    :param winfunc: 对每帧应用的窗函数（analysis window）。默认使用 Hamming 窗。
    :returns: 返回两个值：第一个是 numpy 数组，大小为（NUMFRAMES * nfilt），每行是一帧的特征向量；第二个是每帧能量（energy），为总能量且不加窗（unwindowed）。
    """
    highfreq= highfreq or samplerate/2
    signal = sigproc.remove_dc_offset(signal)
    signal = sigproc.add_dither(signal)
    signal = sigproc.preemphasis(signal,preemph)
    frames = sigproc.framesig(signal, winlen*samplerate, winstep*samplerate, winfunc)
    pspec = sigproc.powspec(frames,nfft)
    energy = numpy.sum(pspec,1) # 存储每帧的总能量（total energy）
    energy = numpy.where(energy == 0,numpy.finfo(float).eps,energy) # 若能量为 0，取 log 时会出现问题
    vad_mask = sigproc.simple_energy_vad_mask(energy)
    pspec = pspec[vad_mask]
    energy = energy[vad_mask]

    fb = get_filterbanks(nfilt,nfft,samplerate,lowfreq,highfreq)
    feat = numpy.dot(pspec,fb.T) # 计算 filterbank（滤波器组）能量
    feat = numpy.where(feat == 0,numpy.finfo(float).eps,feat) # 若特征为 0，取 log 时会出现问题

    return feat,energy

def logfbank(signal,samplerate=16000,winlen=0.025,winstep=0.01,
             nfilt=26,nfft=512,lowfreq=20,highfreq=None,preemph=0.97,
             winfunc=numpy.hamming):
    """从音频信号计算 log Mel-filterbank（对数梅尔滤波器组）能量特征。

    :param signal: 用于计算特征的音频信号（audio signal），应为 N*1 的数组。
    :param samplerate: 当前信号的采样率（sample rate，采样频率），单位 Hz（Hertz，赫兹）。
    :param winlen: 分析窗口长度（analysis window length），单位秒（s）。默认 0.025s（25ms）。
    :param winstep: 相邻窗口之间的步长（step），单位秒（s）。默认 0.01s（10ms）。
    :param nfilt: filterbank（滤波器组）中滤波器数量，默认 26。
    :param nfft: FFT（快速傅里叶变换）点数。默认 512。
    :param lowfreq: mel filters（梅尔滤波器）最低频带边界，单位 Hz（赫兹），默认 20。
    :param highfreq: mel filters（梅尔滤波器）最高频带边界，单位 Hz（赫兹），默认 samplerate/2。
    :param preemph: 预加重（pre-emphasis）滤波器系数。0 表示不做滤波。默认 0.97。
    :param winfunc: 对每帧应用的窗函数（analysis window）。默认使用 Hamming 窗。
    :returns: numpy 数组，大小为（NUMFRAMES * nfilt），每行是一帧的特征向量。
    """
    feat,energy = fbank(signal,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph,winfunc)
    log_feat = numpy.log(feat)
    return sigproc.cmvn(log_feat)

def ssc(signal,samplerate=16000,winlen=0.025,winstep=0.01,
        nfilt=26,nfft=512,lowfreq=20,highfreq=None,preemph=0.97,
        winfunc=numpy.hamming):
    """从音频信号计算 SSC（Spectral Subband Centroid，谱子带质心）特征。

    :param signal: 用于计算特征的音频信号（audio signal），应为 N*1 的数组。
    :param samplerate: 当前信号的采样率（sample rate，采样频率），单位 Hz（Hertz，赫兹）。
    :param winlen: 分析窗口长度（analysis window length），单位秒（s）。默认 0.025s（25ms）。
    :param winstep: 相邻窗口之间的步长（step），单位秒（s）。默认 0.01s（10ms）。
    :param nfilt: filterbank（滤波器组）中滤波器数量，默认 26。
    :param nfft: FFT（快速傅里叶变换）点数。默认 512。
    :param lowfreq: mel filters（梅尔滤波器）最低频带边界，单位 Hz（赫兹），默认 20。
    :param highfreq: mel filters（梅尔滤波器）最高频带边界，单位 Hz（赫兹），默认 samplerate/2。
    :param preemph: 预加重（pre-emphasis）滤波器系数。0 表示不做滤波。默认 0.97。
    :param winfunc: 对每帧应用的窗函数（analysis window）。默认使用 Hamming 窗。
    :returns: numpy 数组，大小为（NUMFRAMES * nfilt），每行是一帧的特征向量。
    """
    highfreq= highfreq or samplerate/2
    signal = sigproc.remove_dc_offset(signal)
    signal = sigproc.add_dither(signal)
    signal = sigproc.preemphasis(signal,preemph)
    frames = sigproc.framesig(signal, winlen*samplerate, winstep*samplerate, winfunc)
    pspec = sigproc.powspec(frames,nfft)
    pspec = numpy.where(pspec == 0,numpy.finfo(float).eps,pspec) # 若全为 0，会导致后续计算出现问题
    energy = numpy.sum(pspec,1)
    vad_mask = sigproc.simple_energy_vad_mask(energy)
    pspec = pspec[vad_mask]

    fb = get_filterbanks(nfilt,nfft,samplerate,lowfreq,highfreq)
    feat = numpy.dot(pspec,fb.T) # 计算 filterbank（滤波器组）能量
    feat = numpy.where(feat == 0,numpy.finfo(float).eps,feat)
    R = numpy.tile(numpy.linspace(1,samplerate/2,numpy.size(pspec,1)),(numpy.size(pspec,0),1))

    ssc_feat = numpy.dot(pspec*R,fb.T) / feat
    return sigproc.cmvn(ssc_feat)

def hz2mel(hz):
    """将 Hertz（赫兹）频率转换为 Mels（梅尔）。

    :param hz: 频率值，单位 Hz（赫兹）。也可以传入 numpy 数组，将逐元素转换。
    :returns: 转换后的值，单位 Mels（梅尔）。若输入为数组，则返回同形状数组。
    """
    return 2595 * numpy.log10(1+hz/700.)

def mel2hz(mel):
    """将 Mels（梅尔）转换为 Hertz（赫兹）频率。

    :param mel: 值，单位 Mels（梅尔）。也可以传入 numpy 数组，将逐元素转换。
    :returns: 转换后的频率值，单位 Hz（Hertz，赫兹）。若输入为数组，则返回同形状数组。
    """
    return 700*(10**(mel/2595.0)-1)

def get_filterbanks(nfilt=20,nfft=512,samplerate=16000,lowfreq=20,highfreq=None):
    """计算 Mel-filterbank（梅尔滤波器组）。

    每一行存放一个滤波器（filter），列对应 FFT（快速傅里叶变换）的 bin（频点）。返回矩阵大小为 nfilt * (nfft/2 + 1)。

    :param nfilt: filterbank（滤波器组）中滤波器数量，默认 20。
    :param nfft: FFT（快速傅里叶变换）点数，默认 512。
    :param samplerate: 当前信号的采样率（sample rate，采样频率），单位 Hz（赫兹），会影响 mel 轴间隔（mel spacing）。
    :param lowfreq: mel filters（梅尔滤波器）最低频带边界，默认 20 Hz。
    :param highfreq: mel filters（梅尔滤波器）最高频带边界，默认 samplerate/2。
    :returns: numpy 数组，大小为 nfilt * (nfft/2 + 1)，每行对应一个滤波器（filter）。
    """
    highfreq= highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

    # 在 mel 轴上计算等间隔采样点（evenly spaced points）
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = numpy.linspace(lowmel,highmel,nfilt+2)
    # 当前点在 Hz 轴，但滤波器需要对齐 FFT bins，因此需要把 Hz 转为 FFT bin 索引
    bin = numpy.floor((nfft+1)*mel2hz(melpoints)/samplerate)

    fbank = numpy.zeros([nfilt,nfft//2+1])
    for j in range(0,nfilt):
        for i in range(int(bin[j]), int(bin[j+1])):
            fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
        for i in range(int(bin[j+1]), int(bin[j+2])):
            fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
    return fbank

def lifter(cepstra, L=22):
    """对 cepstra（倒谱矩阵）应用 cepstral lifter（倒谱提升/升倒谱）。

    其效果是增大高频 DCT（Discrete Cosine Transform，离散余弦变换）系数的幅度（magnitude）。

    :param cepstra: mel-cepstra（梅尔倒谱）矩阵，大小为 numframes * numcep。
    :param L: liftering 系数，默认 22。L <= 0 表示禁用 lifter。
    """
    if L > 0:
        nframes,ncoeff = numpy.shape(cepstra)
        n = numpy.arange(ncoeff)
        lift = 1 + (L/2.)*numpy.sin(numpy.pi*n/L)
        return lift*cepstra
    else:
        # L <= 0 时不做任何处理
        return cepstra

def delta(feat, N):
    """从特征序列（feature vector sequence）计算 delta features（Δ 差分特征）。

    :param feat: numpy 数组，大小为（NUMFRAMES * number of features），每行是一帧的特征向量。
    :param N: 对每一帧，基于前后各 N 帧计算 delta 特征。
    :returns: numpy 数组，大小为（NUMFRAMES * number of features），每行是一帧的 delta 特征向量。
    """
    if N < 1:
        raise ValueError('N must be an integer >= 1')
    NUMFRAMES = len(feat)
    denominator = 2 * sum([i**2 for i in range(1, N+1)])
    delta_feat = numpy.empty_like(feat)
    padded = numpy.pad(feat, ((N, N), (0, 0)), mode='edge')   # 对 feat 进行 padding（边界扩展）
    for t in range(NUMFRAMES):
        delta_feat[t] = numpy.dot(numpy.arange(-N, N+1), padded[t : t+2*N+1]) / denominator   # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
    return delta_feat
