# Computer 项目说明

本项目用于语音特征提取与可视化教学，核心实现为 MFCC 相关流程。

## 1. 项目结构

```text
Computer/
├─ .gitignore
├─ README.md
├─ core/
│  ├─ __init__.py
│  ├─ base.py
│  ├─ sigproc.py
│  └─ audio_io.py
├─ data/
│  └─ english.wav
├─ mfcc_feature_extraction_step_by_step_visualization/
│  └─ mfcc_step_demo.py
└─ window_function_and_spectral_leakage_comparative_experiments/
   └─ window_fft_compare.py
```

## 2. 各目录用途

- `core/`：MFCC 核心实现目录，训练与推理时真正依赖的代码都在这里。
- `data/`：样例音频数据目录，当前包含 `english.wav`。
- `mfcc_feature_extraction_step_by_step_visualization/`：教学可视化目录，逐步骤展示 MFCC 提取过程并输出图像。
- `window_function_and_spectral_leakage_comparative_experiments/`：原理实验目录，对比窗函数与频谱泄漏现象。

### 2.1 core/ 目录内文件功能详解

#### core/__init__.py

- 包入口文件，用于统一导出 `base.py` 中的核心接口。
- 作用是让上层代码可以通过 `from core import ...` 方式调用特征函数。
- 本文件本身不做数值计算，职责是“暴露接口、简化导入”。

#### core/base.py

本文件是“特征构建层”，主要负责从底层频谱结果组装成可用于建模的特征：

- `calculate_nfft`：根据 `samplerate` 与 `winlen` 自动计算合适的 FFT 点数（不小于窗长采样点数的最小 2 的幂）。
- `mfcc`：MFCC 主入口。
  - 先调用 `fbank` 得到 Mel 滤波器组能量；
  - 再做 `log`、`DCT`、`lifter`；
  - 可选用每帧 `log-energy` 替换第 0 维倒谱系数。
- `fbank`：计算 Mel 滤波器组能量特征（未取对数）。
- `logfbank`：在 `fbank` 基础上取对数，得到 log Mel 特征。
- `ssc`：计算谱子带质心特征（Spectral Subband Centroid）。
- `hz2mel / mel2hz`：Hz 与 Mel 频率刻度之间的双向转换。
- `get_filterbanks`：生成三角形 Mel 滤波器组矩阵（`nfilt x (nfft/2+1)`）。
- `lifter`：对倒谱系数进行升倒谱处理，调节高阶倒谱系数权重。
- `delta`：计算时序差分特征（Δ），可进一步得到 ΔΔ。

总结：`base.py` 负责“从频谱到特征向量”的整合，是建模前特征工程的核心控制层。

#### core/sigproc.py

本文件是“底层信号处理算子层”，提供特征提取的基础数学操作：

- `preemphasis`：预加重，增强高频分量，抑制语音高频衰减影响。
- `framesig`：分帧并加窗，把长语音变成短时平稳帧序列。
- `deframesig`：逆分帧（重叠相加），用于信号重建或调试验证。
- `magspec`：计算幅度谱（每帧 rFFT 后取绝对值）。
- `powspec`：计算功率谱（幅度平方并按 NFFT 归一）。
- `logpowspec`：计算对数功率谱（可选归一化）。
- `round_half_up`：用于帧长/帧移的四舍五入策略函数。

总结：`sigproc.py` 负责“从时域到频域”的底层处理，是 `base.py` 的计算基础。

#### core/audio_io.py

本文件是“音频输入与预处理入口层”，统一处理 wav 读取与通道整理：

- `get_default_wav_path`：返回默认样例音频路径（`data/english.wav`）。
- `read_wav`：读取任意 wav 文件，返回采样率与原始波形数组。
- `select_mono_channel`：多通道音频中选取指定通道，统一为单通道。
- `read_wav_mono`：读取任意 wav 并输出指定 dtype 的单通道波形。
- `read_english_wav_mono`：默认样例音频读取入口，支持可选外部路径覆盖。
- `preemphasis_english_wav`：读取样例音频后直接执行预加重并返回结果。

总结：`audio_io.py` 负责“把音频变成可计算输入”，避免上层脚本重复写读取与通道处理逻辑。

## 3. 环境依赖

推荐 Python 3.10+，需要安装：

- numpy
- scipy
- matplotlib

安装示例：

```bash
pip install numpy scipy matplotlib
```

## 4. 运行方式

### 4.1 MFCC 分步骤可视化

默认读取 `data/english.wav`：

```bash
python mfcc_feature_extraction_step_by_step_visualization/mfcc_step_demo.py
```

指定输入 wav：

```bash
python mfcc_feature_extraction_step_by_step_visualization/mfcc_step_demo.py --wav data/english.wav
```

输出图像目录：

- `mfcc_feature_extraction_step_by_step_visualization/mfcc_step_plots/`

### 4.2 窗函数与频谱泄漏对比实验

```bash
python window_function_and_spectral_leakage_comparative_experiments/window_fft_compare.py
```

输出图像目录：

- `window_function_and_spectral_leakage_comparative_experiments/window_fft_compare_plots/`

## 5. MFCC 文字流程图（答辩版）

```text
输入波形 signal
  └─ 来自 data/english.wav 或外部 wav
        |
        v
audio_io.read_wav_mono / read_english_wav_mono
  └─ 统一读取、单通道化、dtype 规范化
        |
        v
base.mfcc(...)
  ├─ calculate_nfft：确定 FFT 点数
  ├─ fbank(...)
  │   ├─ sigproc.remove_dc_offset：去直流分量（默认启用）
  │   ├─ sigproc.add_dither：加入小幅抖动（默认启用）
  │   ├─ sigproc.preemphasis：预加重
  │   ├─ sigproc.framesig(Hamming)：分帧+Hamming窗（默认启用）
  │   ├─ sigproc.powspec：功率谱
  │   ├─ sigproc.simple_energy_vad_mask：能量阈值 VAD（默认启用）
  │   ├─ get_filterbanks：Mel 三角滤波器组
  │   └─ dot(pspec, fbank.T)：Mel 能量
  ├─ log：对数压缩动态范围
  ├─ DCT：得到倒谱系数
  ├─ lifter：升倒谱
  ├─ appendEnergy：可选替换第0维为 log-energy
  └─ sigproc.cmvn：逐句 CMVN 标准化（默认启用）
        |
        v
输出 MFCC 特征矩阵 [num_frames, numcep]
  └─ 可继续计算 delta / delta-delta 作为动态特征
```

## 6. 30 秒答辩讲解稿

本项目把 MFCC 特征提取拆成三层：`audio_io.py` 负责音频读取和单通道预处理，`sigproc.py` 负责底层信号处理，`base.py` 负责特征构建与整合。当前链路默认启用 DC 去除、dither、Hamming 窗、简单能量 VAD 和 CMVN，最终输出可直接用于训练与推理的 MFCC 特征。整体流程是“时域波形 → 预处理与筛帧 → 频域功率谱 → Mel 能量 → 倒谱特征标准化”，并提供逐步骤可视化与频谱泄漏对比实验用于结果解释。

## 7. Git 日常流程

```bash
git pull
git status
git add 文件1 文件2
git commit -m "本次改动说明"
git push
```
