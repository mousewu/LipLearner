# 视觉语音识别(VSR)技术深度剖析

> 本文是 [VSR_RESEARCH_SURVEY_zh.md](VSR_RESEARCH_SURVEY_zh.md) 的**技术详解版**,
> 深入到架构参数、损失函数、数据处理与工程实现层面。
>
> 材料来源:中科院计算所 VIPL 视听语言感知与理解组 2019–2025 全部公开论文、
> 领域内关键工作,以及**本项目对 LipLearner / mpc001-TCN / auto_avsr / CNVSRC
> 四套代码的实际阅读与端侧移植实测**。

---

## 目录

- [第一部分:问题的数学本质](#第一部分问题的数学本质)
- [第二部分:典型架构逐层拆解](#第二部分典型架构逐层拆解)
- [第三部分:数据处理管线(工程关键)](#第三部分数据处理管线工程关键)
- [第四部分:训练策略与损失函数](#第四部分训练策略与损失函数)
- [第五部分:五大技术难点的攻防](#第五部分五大技术难点的攻防)
- [第六部分:技术路线演进(附完整时间线)](#第六部分技术路线演进附完整时间线)
- [第七部分:端侧部署的工程约束](#第七部分端侧部署的工程约束)
- [第八部分:性能水位与差距分析](#第八部分性能水位与差距分析)

---

## 第一部分:问题的数学本质

### 1.1 形式化定义

VSR 要学习的映射是:

```
f : V → Y
V = (v₁, v₂, ..., v_T)     唇部图像序列,v_t ∈ ℝ^(H×W)
Y = (y₁, y₂, ..., y_L)     文本序列,L ≠ T(长度不对齐)
```

三个特性决定了它的难度:

1. **T ≠ L**:输入输出长度不对齐 → 必须用 CTC 或 Attention 做对齐
2. **多对一的信息坍缩**:多个不同的 `y` 对应几乎相同的 `V`(见 1.2)
3. **强时序依赖**:单帧无意义,`y_i` 依赖 `v_t` 的一个**时间窗口**(协同发音)

### 1.2 核心矛盾:视位坍缩(Viseme Collapse)

这是 VSR 与 ASR 的**本质区别**,也是所有技术难点的源头。

设音素集 `P`(英语 |P| ≈ 40),视位集 `Ω`(|Ω| ≈ 10–14),存在映射:

```
φ : P → Ω        且 φ 是满射非单射(多对一)
```

**信息论表述**:视觉观测 `V` 与文本 `Y` 之间的互信息存在上界

```
I(V; Y) < H(Y)
```

即使有完美的模型和无限数据,也无法从 `V` 完全恢复 `Y`。缺失的信息量约等于:

```
ΔH ≈ log(|P| / |Ω|) ≈ log(40/12) ≈ 1.74 bit / 音素
```

**具体表现**(以英语为例):

| 视位组 | 包含音素 | 视觉不可区分的原因 |
|---|---|---|
| 双唇音 | /p/ /b/ /m/ | 仅差**清浊**与**鼻腔共鸣**,唇形完全相同 |
| 唇齿音 | /f/ /v/ | 仅差声带振动 |
| 齿间音 | /θ/ /ð/ | 同上(think / this) |
| 齿龈音 | /t/ /d/ /n/ /s/ /z/ | 舌位差异在口腔内部,外部不可见 |
| 软腭音 | /k/ /g/ /ŋ/ | 发音部位完全在口腔深处 |

因此 `pat / bat / mat`、`fan / van`、`kill / gill` 在纯视觉下**理论上不可区分**。

> **这一条决定了整个领域的走向**:既然视觉信息本身不足,唯一出路是**引入外部信息**
> —— 语言模型、上下文、先验知识。这正是第六部分演进的主线。

### 1.3 中文的额外坍缩:声调不可见

普通话的 **4 个声调完全由基频(F0)决定**,而 F0 是**声带振动频率**,
在唇部**没有任何视觉对应物**。

```
妈(mā) 麻(má) 马(mǎ) 骂(mà)  →  唇部运动完全相同
```

这在英语中没有对应问题。定量地说,中文 VSR 在音节层面额外损失约:

```
ΔH_tone = log₂(4) = 2 bit / 音节
```

叠加上汉语**单音节同音字极多**(如 "shi" 对应 100+ 汉字),
中文 VSR 的理论天花板显著低于英语 —— 这与实测数据吻合(见第八部分)。

---

## 第二部分:典型架构逐层拆解

现代 VSR 系统普遍是三段式:**时空前端 → 时序骨干 → 解码头**。
以下参数均来自本项目实际阅读的源码。

### 2.1 通用架构骨架

```
唇部灰度序列 (B, T, 1, 88, 88)
        │
   ┌────▼─────────────────────────────────┐
   │ ① 时空前端 (Spatiotemporal Frontend) │
   │   Conv3D(1→64, k=(5,7,7), s=(1,2,2)) │  ← 时间维 kernel=5,捕捉局部动态
   │   BatchNorm3D + ReLU/PReLU/Swish     │
   │   MaxPool3D(k=(1,3,3), s=(1,2,2))    │  ← 只在空间降采样,保留时间分辨率
   └────┬─────────────────────────────────┘
        │  (B×T, 64, 22, 22)
   ┌────▼─────────────────────────────────┐
   │ ② 2D 骨干:ResNet-18 / ShuffleNetV2  │
   │   逐帧提取空间特征 → 512 维          │
   │   (可选 SE 通道注意力)               │
   └────┬─────────────────────────────────┘
        │  (B, T, 512)
   ┌────▼─────────────────────────────────┐
   │ ③ 时序骨干                           │
   │   BiGRU / MS-TCN / Conformer         │
   └────┬─────────────────────────────────┘
        │
   ┌────▼─────────────────────────────────┐
   │ ④ 解码头                             │
   │   词分类 / CTC / Attention / LLM     │
   └──────────────────────────────────────┘
```

**关键设计点**:第 ① 层的 `Conv3D` 时间核为 5,而 `MaxPool3D` 的时间步长为 1
—— 这是刻意的:**空间可以大幅降采样,时间分辨率必须保留**,因为唇动的判别信息
恰恰在毫秒级的时序变化中。

### 2.2 词级架构 A:LipLearner 编码器(ResNet18 + BiGRU)

本项目 [pretraining/model/model.py](pretraining/model/model.py) 的实际配置:

```python
frontend3D:  Conv3d(1, 64, kernel_size=(5,7,7), stride=(1,2,2), padding=(2,3,3))
resnet18:    ResNet(BasicBlock, [2,2,2,2], se=True)      # 带 SE 模块
gru:         nn.GRU(512 + 1, 1024, num_layers=3,
                    batch_first=True, bidirectional=True, dropout=0.2)
head:        nn.Linear(1024*2, 500)                      # 投影头,输出 500 维
输出:        head(h).mean(dim=1)                          # 时间维平均池化
```

**注意 `512 + 1` 中的那个 `+1`** —— 这是**词边界(word boundary)信号**,
逐帧标注"这一帧是否在说话区间内"。来自 VIPL 的 *Learn an Effective Lip Reading
Model without Pains*,是该论文最有效的单项技巧。

**SE 模块**(Squeeze-and-Excitation)的作用:
```python
w = GlobalAvgPool(out)         # (B,C,1,1)  squeeze
w = Conv1x1(C → C/16) → ReLU
w = Conv1x1(C/16 → C) → Sigmoid  # excitation
out = out * w                  # 通道重加权
```
本质是**通道注意力**,让网络自适应地强调对唇动敏感的特征通道。

### 2.3 词级架构 B:mpc001 TCN 系列

时序骨干换成 **TCN(时序卷积网络)**,相比 RNN 可并行、感受野可控:

| 变体 | 骨干 | 时序头 | 参数量 | LRW 准确率 |
|---|---|---|---|---|
| `snv05x_tcn1x` | ShuffleNetV2 0.5× | TCN (k=3, 4层) | 最小 | 79.9% |
| `snv1x_tcn1x` | ShuffleNetV2 1× | TCN (k=3, 4层) | | 82.7% |
| `snv1x_dsmstcn3x` | ShuffleNetV2 1× | **DS-MS-TCN** (k={3,5,7}) | | 85.3% |
| `resnet18_mstcn` | ResNet-18 | **MS-TCN** (k={3,5,7}) | 最大 | 88.9% |

**MS-TCN(多尺度 TCN)**的关键:并行多个不同 kernel size 的分支
```
kernel_size = [3, 5, 7]   →  三个分支覆盖短/中/长时程
num_channels = hidden_dim × len(kernel) × width_mult   # 256×3×1 = 768
num_layers = 4,每层膨胀率翻倍 → 感受野指数增长
```
这直接对应了**协同发音**的多尺度特性:音素级(短)、音节级(中)、词级(长)。

**DS-TCN(深度可分离 TCN)**:把标准卷积拆成 depthwise + pointwise,
参数量降低约 8–9 倍,是端侧友好的关键设计。

### 2.4 句子级架构:Conformer + CTC/Attention 混合

auto_avsr 与 CNVSRC 使用同构设计(本项目实测配置):

```python
frontend:      ResNet (视觉前端) → 512 维
proj_encoder:  Linear(512 → 768)
encoder:       ConformerEncoder(
                   attention_dim = 768,
                   attention_heads = 12,
                   linear_units = 3072,      # FFN 维度
                   num_blocks = 12,
                   cnn_module_kernel = 31,   # 卷积模块核大小
               )
decoder:       TransformerDecoder(num_blocks=6, dim=768, heads=12)
ctc:           CTC(odim, 768)
loss:          L = 0.1 × L_ctc + 0.9 × L_attention
```

**为什么是 Conformer 而非纯 Transformer?**

Conformer 每个 block 内串联了 **卷积模块** 和 **自注意力**:
```
x = x + ½·FFN(x)                    # Macaron 结构前半
x = x + MHSA(x)                     # 多头自注意力:全局依赖
x = x + Conv(x)   ← kernel=31       # 卷积:局部时序模式
x = x + ½·FFN(x)                    # Macaron 结构后半
x = LayerNorm(x)
```

这对 VSR 特别合适:
- **卷积分支**捕捉音素级的局部唇动模式(kernel=31 ≈ 1.24 秒 @25fps)
- **注意力分支**捕捉跨词的长程语言依赖
- 二者互补,恰好匹配语音的层次结构

**CTC/Attention 混合的意义**:
- CTC 提供**单调对齐**约束(语音天然是单调的),稳定训练、加速收敛
- Attention 提供**语言建模**能力,处理歧义
- 推理时联合打分:`score = λ·log P_ctc + (1-λ)·log P_att`

### 2.5 自监督架构:AV-HuBERT 与 ES³

#### AV-HuBERT(Meta,2022)
```
① 用 MFCC 做 k-means 聚类 → 得到伪标签(离散单元)
② 随机 mask 音频流和视频流(独立 mask)
③ 模型预测被 mask 帧的聚类标签
④ 用学到的表征重新聚类 → 迭代精炼伪标签,回到 ②
```
架构:FFN 音频提取器 + 改进 ResNet-18 视频提取器 + 融合模块 + Transformer 骨干。

**结果**:LRS3 上 433h 标注 + 自训练 → **WER 26.9%**;仅 30h 标注 → 32.5%。

#### ES³(VIPL,CVPR 2024)—— 对 AV-HuBERT 范式的批判性改进

**核心论点**:以往方法主要"用音频指导视频",只学到了**共享信息**,
但音视频存在**内在不对称性**。应当分别获取三类信息:

| 信息类型 | 含义 | 举例 |
|---|---|---|
| **Shared(共享)** | 两模态都有 | 音素身份 |
| **Unique(独有)** | 仅单模态有 | 音频独有:声调/清浊;视觉独有:唇形轮廓 |
| **Synergistic(协同)** | 需两者结合才涌现 | 在噪声下的互补消歧 |

**三阶段演进策略**:
```
Stage 1: 用更易学的音频初始化 → 捕获 audio-unique + shared
Stage 2: 引入 video-unique,在已有共享知识上 bootstrap 视听表征
Stage 3: 最大化总视听信息(含 synergistic)
```
实现为简单的 Siamese 框架。**效果**:LRS2-BBC 上最小模型用
**1/2 参数 + 1/8 无标注数据(223h)** 达到 SOTA 水平。

### 2.6 LLM 时代架构:VALLR(ICCV 2025)

**两阶段音素中心框架**,是当前 LRS3 的 SOTA(WER 18.7%):

```
Stage 1:  Video Transformer + CTC head  →  音素序列
          (任务复杂度大幅降低,且天然说话人无关)
                    │
                    ▼  紧凑的音素序列
Stage 2:  微调的 LLM  →  重建词与句子
          (用语言上下文解决视位歧义)
```

**为什么这个设计如此有效?** 它直接对应了 1.2 的数学本质:

1. **视觉端只做视觉能做的事** —— 预测音素(甚至只是视位),不强求预测词
2. **消歧交给语言模型** —— `/p/ /b/ /m/` 的选择由上下文决定,这本就是语言学问题
3. **数据效率极高** —— 论文报告用比次优方法**少 99.4% 的标注数据**达到 SOTA

**LLM 规模的影响**(VALLR 内部消融,同一视觉前端):
| LLM | 参数量 | LRS3 WER |
|---|---|---|
| GPT-2 Small | 0.12B | 23.9% |
| Llama 3.2-3B | 3B | **18.7%** |

> ⚠️ 但这条曲线**不能**推广为"LLM 大幅提升了 VSR" —— 见下一节 2.7 的反例研究。

### 2.7 关键修正:LLM 到底带来了多少提升?⭐

这是一个容易被"LLM 范式"叙事误导的问题。把**同为纯视觉、同在 LRS3** 的结果并列:

| 方法 | 用 LLM? | 标注数据 | LRS3 WER |
|---|---|---|---|
| AV-HuBERT + Transformer 解码器 | ❌ | 433h | 28.6% |
| GLip (BMVC 2025) | ❌ | — | 30.1% |
| Fast Conformer | ❌ | — | 25.5% |
| **Auto-AVSR** | ❌ | 3448h(伪标签) | **20.3%** |
| VSP-LLM | ✅ | — | 26.7% |
| Not Only Vision (ICCV 2025) | ❌(用外围信息) | — | 22.03% |
| **VALLR** (ICCV 2025) | ✅ | **极少(少 99.4%)** | **18.7%** |

**关键观察**:
1. 用 LLM 的最佳(18.7%)对比不用 LLM 的最佳(20.3%),**仅领先 1.6 个百分点(相对 ~8%)**
2. **多数 LLM 方法(24–28%)反而不如传统的 Auto-AVSR(20.3%)**

#### 反例研究:《From Hype to Insight》(2025)

有一篇论文专门审视了这个问题,题目本身就是态度:
**《From Hype to Insight: Rethinking Large Language Model Integration in Visual Speech Recognition》**

其系统性消融的发现:

| 实验 | 结果 | 含义 |
|---|---|---|
| 解码器 **1B → 13B** | 仅在数据充足多样时(LRS2+LRS3 = 657h)才有提升;**单用 LRS3 提升微乎其微** | LLM 规模收益依赖数据 |
| **4 个不同 13B LLM** 对比 | 彼此仅差 **~1.5% WER** | 换哪个 LLM 几乎不重要 |
| **4-bit QLoRA vs 16-bit LoRA** | 性能几乎相同 | 适配方式是次要因素 |
| **语义指标**(BERTScore/METEOR) | AV-HuBERT 与 VSP-LLM 几乎一致 | LLM 精修的是**词法**而非**语义** |

论文的核心结论:

> *"LLM 解码器带来的 WER 改善是边际的,收益主要来自**更强的语言建模,而非更好的视觉理解**。"*
>
> *"LLM decoders refine contextual reasoning rather than visual features, emphasizing the
> need for **stronger visual encoders** to drive meaningful progress."*

作者明确指出:**瓶颈在视觉编码器,而非解码器设计**(其最佳模型 LRS3 WER 24.7%)。

#### 那 LLM 的真正价值在哪?

**在数据效率,而非绝对精度。**

VALLR 的真正亮点不是 18.7% 这个数字,而是它用**比次优方法少 99.4% 的标注数据**
达到了这个成绩。语言先验可以**替代**一部分对标注数据的需求。

#### 与理论的呼应

这个结论恰好印证了 1.2 的视位坍缩:

```
LLM 能做的:用语言先验去猜视觉分不清的音素(/p/ /b/ /m/)
LLM 不能做的:凭空创造视觉信号里不存在的信息
```

推论有两个,都对工程实践有直接影响:
- LLM 的帮助**依赖上下文长度** → 在**孤立短命令**上几乎无效
  (这正是 LipLearner 式"注册命令"路线在短指令场景更实用的原因)
- 要实质性突破,**必须提升视觉编码器**,而非继续堆解码器

> **领域共识正在回摆**:2024 年大量工作涌向 LLM;2025 年的批判性研究表明,
> 瓶颈依然在视觉端。

---

## 第三部分:数据处理管线(工程关键)

⚠️ 这一部分在论文中常被一笔带过,但**本项目实测证明它对结果的影响不亚于模型选择**。

### 3.1 标准预处理流程

```
原始视频
  │
  ├─① 人脸检测 + 关键点定位
  │    · 4 点(BlazeFace):右眼/左眼/鼻尖/嘴中心 —— auto_avsr
  │    · 68 点(dlib/FAN):完整人脸轮廓 —— CNVSRC
  │
  ├─② 关键点时间平滑
  │    window_margin = 12(±6 帧居中窗口)
  │    平滑形状,但用当前帧质心重新定位 → 抑制抖动而不引入滞后
  │
  ├─③ 仿射对齐到"平均脸"
  │    estimateAffinePartial2D(landmarks, mean_face_reference)
  │    → 变换到 256×256 参考空间
  │    消除:头部姿态、尺度、平面内旋转
  │
  ├─④ 以变换后的嘴部关键点为中心裁 96×96
  │
  ├─⑤ CenterCrop 96→88(测试)/ RandomCrop(训练)
  │
  ├─⑥ 灰度化(ITU-R 601-2 luma:0.299R + 0.587G + 0.114B)
  │
  └─⑦ 归一化:x/255 → (x − 0.421) / 0.165
```

**注意 ③ 的重要性**:模型训练时看到的是**对齐到标准姿态的人脸**,
而非"嘴周围的方框"。本项目实测:若跳过对齐直接裁唇部方框,
识别结果从 `PLEASE CALL YOUR DOCTOR` 退化为完全无意义的输出。

**归一化常数 (0.421, 0.165)** 是 LRW 灰度唇部区域的全局均值/标准差,
auto_avsr 与 CNVSRC 使用**完全相同**的值 —— 这也是为什么本项目能让中文模型
复用英文的预处理管线。

### 3.2 帧率:一个被严重低估的因素

**所有主流 VSR 模型均在 25 fps 上训练**(LRW/LRS/CN-CVS 均为 25fps)。

模型学到的是**特定时间尺度上的唇动模式**。若推理时帧率不匹配:

```
实际帧率 16fps,却按连续帧送入 →  模型认为语速快了 25/16 ≈ 1.56 倍
```

本项目实测,这一项就足以让识别从可用退化到不可用。
**解决方案**:按时间戳重采样到 25fps,而非简单地按帧送入。

### 3.3 训练时的数据增强

来自 LipLearner 与 mpc001 的实际实现:

| 增强 | 实现 | 针对的问题 |
|---|---|---|
| **RandomCrop** | 88–100 随机尺寸 + 随机位置 | 定位误差 |
| **HorizontalFlip** | p=0.5 | 左右差异 |
| **Fisheye Distortion** | 网格形变,magnitude ≤ 0.07 | **手机不同持握距离/角度的透视畸变** |
| **RandomFrameDrop** | 随机丢弃发音区间外的帧 | 语速/时机差异 |
| **Per-frame shaking** | 随机抖动部分帧的裁剪位置 | 手持抖动 |
| **Mixup** | 样本线性插值 | 过拟合 |
| **Time Mask** | 随机遮蔽时间段 | 遮挡/丢帧鲁棒性 |

> LipLearner 的 **fisheye 增强** 是为移动端场景特别设计的 —— 这类"面向部署场景"
> 的增强设计,是学术模型能否落地的关键细节。

---

## 第四部分:训练策略与损失函数

### 4.1 三类主要目标函数

#### (a) 词级分类:交叉熵 + Label Smoothing
```
L = −Σ [(1−ε)·y_true + ε/K] · log p
ε = 0.1(典型值),K = 类别数
```
Label smoothing 缓解**视位歧义导致的过度自信** —— 对 VSR 尤其重要,
因为许多类别本就视觉相似,强制 one-hot 会损害泛化。

#### (b) 句子级:CTC + Attention 联合
```
L = λ·L_CTC + (1−λ)·L_Attention,  λ = 0.1(训练)/ 0.3(CNVSRC 解码)

L_CTC = −log Σ_{π ∈ B⁻¹(y)} P(π|V)      # 对所有合法对齐求和
L_Att = −Σ_i log P(y_i | y_<i, V)          # 自回归
```

#### (c) 对比学习:InfoNCE(LipLearner 的做法)
```python
logits = (z_a @ z_b.T) / τ          # τ = 0.07
loss = CrossEntropy(logits, I)       # 对角线为正样本
```
其中 `z_a`、`z_b` 是同一个词的两个不同片段的 L2 归一化嵌入。

> **这是 LipLearner 的关键设计**:训练出的不是分类器,而是一个**度量空间**。
> 因此新命令只需几个样本即可加入 —— 无需重训编码器,只需在嵌入之上训练轻量分类器。
> 本项目实测:6 类 31 样本的 Softmax 回归仅 3006 个参数,300 轮梯度下降耗时毫秒级。

### 4.2 训练技巧的量化价值(VIPL 的核心贡献之一)

*Learn an Effective Lip Reading Model without Pains* 系统性地量化了各项技巧:

| 数据集 | Baseline | + 全部技巧 | 提升 |
|---|---|---|---|
| LRW | 83.7% | **88.4%** | +4.7 |
| LRW-1000 | 38.2% | **55.7%** | **+17.5** |

技巧清单:词边界信息、Cosine LR 调度、Mixup、Label Smoothing、SE 模块。

> **中文数据集提升 17.5 个百分点,全部来自训练策略而非架构。**
> 这个结果的启示是:在 VSR 这种**低信噪比**任务上,优化过程的细节
> 比模型容量更关键。

### 4.3 强化学习:直接优化评测指标

*Pseudo-Convolutional Policy Gradient*(FG 2020)解决 seq2seq 的两个痼疾:
1. **Exposure bias**:训练用 teacher-forcing,推理用自己的预测 → 分布不匹配
2. **目标不一致**:训练优化交叉熵,评测用 CER/WER

方法:把 **CER 本身作为奖励**引入策略梯度,并在奖励/损失维度上做
"伪卷积"操作以纳入每个时间步周围的上下文。
结果:LRW 83.5%、LRW-1000 38.70%、GRID WER 11.2%。

---

## 第五部分:五大技术难点的攻防

### 难点 1:视位歧义(理论天花板)

| 攻法 | 代表工作 | 效果 |
|---|---|---|
| 强语言模型消歧 | **VALLR**(音素→LLM) | LRS3 **18.7%**,SOTA |
| 引入外部上下文 | **Not Only Vision**(ICCV 2025) | LRS3 **22.03%** |
| 细粒度判别约束 | Mutual Information Max(FG 2020) | 区分 spend/spending |
| 音素作为中间表示 | VALLR、多语言 SBL | 降低任务复杂度 |

**Not Only Vision 的"外围信息"分类**(这是认知层面的创新):

| 类型 | 内容 | 类比人类 |
|---|---|---|
| Contextual Guidance | 话题、场景描述 | 知道在聊什么就更好猜 |
| Task Expertise | 唇读先验经验 | 专业唇读员的技巧 |
| Linguistic Perturbation | 无关但同时处理的信号 | 干扰项 |

技术上用**分层处理 + 动态路由**缓解模态冲突,按相关性选择性利用。

### 难点 2:说话人差异

**Speaker-Adaptive Lip-Reading**(BMVC 2023)的分层洞察:

```
浅层网络:说话人特征 >> 内容信号   →  用自适应特征【增强】内容
深层网络:两者都已充分表征        →  用自适应特征【抑制】无关噪声
```

依据:*"说话人自身特征总能被浅层网络很好刻画,而与说话内容相关的动态特征
总是需要深层时序网络"*。

其他路线:
- **解耦表征**:Audio-guided Disentangled(FCS 2024)—— 双分支 + 瓶颈,分离内容与身份
- **互信息约束**:LMIM/GMIM 抑制身份相关的方差

### 难点 3:真实场景的视觉退化

**GLip**(BMVC 2025)针对光照/遮挡/模糊/姿态,两个关键洞察:

1. 先学**粗对齐**(全局+局部视觉特征 ↔ 声学语音单元),再学精确映射
2. **恶劣条件下,某些局部区域(如未遮挡部分)比全局特征更具判别力**

架构:双路特征提取 + 两阶段渐进学习 + **上下文增强模块(CEM)**,
在时空两个维度动态融合局部与全局。

**Can We Read Speech Beyond the Lips?**(FG 2020)的反直觉结论也属此类:
引入唇部之外的区域(**甚至只有上半脸**)都能稳定提升性能。
→ 工程启示:**不要只裁嘴部**。

### 难点 4:数据稀缺

| 路线 | 代表 | 关键数字 |
|---|---|---|
| **视听自监督** | AV-HuBERT | 30h 标注即达 WER 32.5% |
| **三类信息演进** | ES³(CVPR 2024) | 1/8 数据 + 1/2 参数达 SOTA |
| **单模态弱监督** | UniLip(BMVC 2023) | 无监督 LRS3 **WER 51.2%** |
| **ASR 伪标签** | Auto-AVSR | 438h → 3448h,WER 36→20.3 |
| **语言先验替代数据** | VALLR | 少 **99.4%** 标注数据达 SOTA |

> **本项目实测印证**:同架构下 438h 训练的模型识别全错(`IS ALL THAT`),
> 3291h 训练的模型识别全对(`PLEASE CALL THE DOCTOR`)。

### 难点 5:时序对齐与协同发音

- **协同发音**:同一音素的口型受前后音素影响 → 需要多尺度时序建模(MS-TCN)
- **长度不对齐**:CTC(单调)+ Attention(灵活)混合
- **运动 vs 外观**:Deformation Flow(FG 2020)显式建模帧间形变流,
  配合**双向知识蒸馏**让两支互学

---

## 第六部分:技术路线演进(附完整时间线)

### 6.1 演进主线:信息来源的不断扩张

```
2016–2019  只看嘴部          →  词级分类,闭集
    │
2020       扩展到全脸        →  Can We Read Speech Beyond the Lips?
    │      显式建模运动      →  Deformation Flow
    │      信息论约束        →  Mutual Information Maximization
    │
2021–2022  训练策略系统化    →  Learn without Pains(+17.5% 中文)
    │      音频作为监督      →  Audio-Driven Deformation Flow
    │
2023       自监督/弱监督     →  UniLip(单模态数据)
    │      说话人自适应      →  Separable Hidden Unit
    │
2024       视听表征革新      →  ES³(shared/unique/synergistic)
    │      解耦表征          →  Audio-guided Disentangled
    │
2025       **引入视觉之外**  →  Not Only Vision(外围信息)
    │      **鲁棒性优先**    →  GLip(局部优于全局)
    └───►  **LLM 消歧**      →  VALLR(音素 → LLM)
```

### 6.2 VIPL 完整论文清单(按主题归类)

| 年份 | 论文 | 会议 | 主题 |
|---|---|---|---|
| 2019 | LRW-1000 | FG(Oral) | 数据集 |
| 2020 | Can We Read Speech Beyond the Lips? | FG | RoI 选择 |
| 2020 | Deformation Flow Two-Stream | FG | 运动建模 |
| 2020 | Mutual Information Maximization | FG | 特征约束 |
| 2020 | Pseudo-Convolutional Policy Gradient | FG | 序列优化 |
| 2020 | Synchronous Bidirectional Learning | BMVC | 多语言 |
| 2021 | Learn Effective Model without Pains | ICMEW | 训练策略 |
| 2021 | UniCon | ACM MM | 主动说话人检测 |
| 2021 | AVA-ActiveSpeaker 冠军 | CVPR-W | 竞赛 |
| 2022 | Audio-Driven Deformation Flow | ICPR | 音频引导 |
| 2023 | UniLip | BMVC | 弱监督 |
| 2023 | Speaker-Adaptive Lip-Reading | BMVC | 说话人差异 |
| 2023 | Cooperative Dual Attention | BMVC | 语音增强 |
| 2024 | **ES³** | **CVPR** | 视听自监督 |
| 2024 | Audio-guided Disentangled | FCS | 解耦表征 |
| 2025 | **Not Only Vision** | **ICCV** | 外围信息 |
| 2025 | CogCM | ICCV | 语音增强 |
| 2025 | GLip | BMVC | 鲁棒性 |

### 6.3 六大趋势总结

| # | 趋势 | 从 | 到 |
|---|---|---|---|
| 1 | 任务粒度 | 词级闭集分类 | 句子级开放词汇 |
| 2 | 监督方式 | 全监督 | 自监督 + 伪标签 + 语言先验 |
| 3 | 信息来源 | 只看嘴 | 全脸 → 上下文 → 外围信息 |
| 4 | 解码方式 | CTC / Seq2Seq | LLM 作为语言先验(⚠️ 精度收益有限,主要价值在**数据效率**,见 2.7)|
| 5 | 评价重心 | 干净数据集指标 | 真实场景鲁棒性 |
| 6 | 语言覆盖 | 单语(英语) | 多语言 + 跨语言迁移 |

---

## 第七部分:端侧部署的工程约束

本节全部来自**本项目在 nubia P0110 / Android 16 / arm64 上的实测**。

### 7.1 模型能否上端侧:取决于解码方式

| 解码 | 计算图 | 能否 ONNX 静态图 | 端侧可行 |
|---|---|---|---|
| 词分类 | 单次前向 | ✅ | ✅ |
| **CTC 贪心** | 单次前向 + argmax | ✅ | ✅ |
| CTC beam search | 前向 + 搜索 | 前向可,搜索需手写 | ⚠️ |
| **Attention 自回归** | 逐 token 循环 | ❌ 带状态 | ❌ 需手写解码器 |
| + LLM | 更大的自回归 | ❌ | ❌ |

**实测对比**(同一段视频,同一模型):
| 解码 | 结果 | 耗时 |
|---|---|---|
| Attention + CTC + beam | `PLEASE CALL THE DOCTOR` ✅ | 2.7s |
| CTC 贪心 | `PLEASE CALL YOUR DOCTOR` ✅ | **0.36s** |

> **结论**:CTC 贪心只差一个词但快 7 倍且能上端侧,是当前端侧自由 VSR 的唯一现实路径。
> 但代价是失去了 LLM 消歧能力 —— 而那恰恰是 SOTA 方法的核心。

### 7.2 六个必须避开的坑(均为实测)

| # | 坑 | 现象 | 根因 | 解法 |
|---|---|---|---|---|
| 1 | **NNAPI 静默算错** | 同输入桌面对、手机错 | Conformer 动态长度注意力 NNAPI 支持不全,图切分后回退,**不报错** | 禁用 NNAPI |
| 2 | **ARM fp16 精度崩溃** | fp16 桌面对、手机错 | x86 内部转 fp32 计算;ARM 用原生 fp16,12 层累积误差放大 | 端侧用 fp32 |
| 3 | **旋转元数据未应用** | 输出完全无意义 | 手机竖屏 mp4 带 rotation=90,PyAV 不自动应用 | 用 OpenCV 解码 |
| 4 | **帧率不匹配** | 管线全对但不准 | 采集 16fps 当作 25fps → 语速失真 1.56 倍 | 按时间戳重采样 |
| 5 | **对齐方式错误** | 输出垃圾 | 用"唇部方框"而非"平均脸仿射对齐" | 复现完整对齐流程 |
| 6 | **大模型 OOM** | 启动即崩 | `readBytes()` 把 145MB 读进 256MB Java 堆 | 流式落盘 + mmap |

### 7.3 端侧性能实测

| 项 | 数值 |
|---|---|
| BlazeFace 检测(4点) | ~9 ms/帧 |
| MediaPipe Face Mesh(468点) | ~25 ms/帧 |
| 仿射对齐 + 裁剪 | ~2-4 ms/帧 |
| 图像转换(720p 旋转镜像) | ~14 ms/帧 |
| 图像转换(1080p) | ~24 ms/帧 ⚠️ |
| 词级编码器推理(12MB 模型) | 40–60 ms |
| 句子级编码器 + CTC(738MB) | 0.4–1.3 s |
| 端上分类器训练(6类31样本) | 毫秒级 |

> **关键取舍**:用 468 点 Face Mesh 会把帧率压到 ~19fps(低于模型要求的 25fps),
> 换成 4 点 BlazeFace 后恢复到 ~30fps。**精度换帧率是划算的** ——
> 因为帧率不足对 VSR 的伤害大于关键点精度的损失。

---

## 第八部分:性能水位与差距分析

### 8.1 当前 SOTA 一览

| 任务 | 数据集 | 最佳 | 方法 |
|---|---|---|---|
| 英文句子级 VSR | LRS3 | **WER 18.7%** | VALLR(音素+LLM) |
| 英文句子级 VSR | LRS3 | WER 20.3% | Auto-AVSR(3448h 伪标签) |
| 英文句子级 VSR | LRS3 | WER 22.03% | Not Only Vision(外围信息) |
| 英文句子级 VSR | LRS3 | WER 26.9% | AV-HuBERT(433h + 自训练) |
| 英文**视听** ASR | LRS3 | **WER < 1%** | MMS-LLaMA / Llama-AVSR |
| 英文词级 | LRW | 88.4–89% | Learn without Pains / DC-TCN |
| 中文词级 | LRW-1000 | **56.0%** | Learn without Pains |
| **中文句子级** | CNVSRC | **CER 30–34%** | NPU-ASLP(2024 冠军) |

### 8.2 三个关键差距

#### 差距一:纯视觉 vs 视听(约 20 倍)
```
LRS3:  纯视觉 WER ≈ 19%   vs   视听 WER < 1%
```
这个差距**定量地测出了视觉通道的信息缺口**。它不会随模型进步而消失,
因为根源是 1.2 的视位坍缩。

#### 差距二:英文 vs 中文(约 1.7 倍)
```
词级:  LRW 88.4%     vs   LRW-1000 56.0%
句子级:LRS3 WER 18.7% vs   CNVSRC CER 30–34%
```
原因(见 1.3):**声调不可见**(额外 2 bit/音节损失)+ 同音字海量 + 数据更少
+ LRW-1000 刻意保留自然长尾分布。

> 新方向:**VALLR-Pin** 用**拼音引导**做中文 VSR —— 先识别拼音(视觉可及部分),
> 再用语言模型恢复汉字,正是绕开"声调不可见"的思路。

#### 差距三:数据集 vs 真实场景
本项目实测:同一模型、同一部手机,
- **系统相机规整录制** → `PLEASE CALL YOUR DOCTOR` ✅
- **实时手持按住说话** → `BE CALLED TO` / `PEACE IS CALLED CORNER` ❌

差异来源:分辨率、帧稳定性、ISP 处理、**按下/松开切掉词头词尾**。
学术指标建立在规整数据上,落地时这个差距必须单独工程化处理。

---

## 第九部分:结论与工程建议

### 9.1 对技术难点的总判断

VSR 的难点是**分层的**,且**越底层越难解**:

```
┌─────────────────────────────────────────────┐
│ L1 信息层:视位坍缩(不可消除)              │  ← 只能靠外部信息补偿
│    → 出路:LLM / 上下文 / 音素中间表示       │
├─────────────────────────────────────────────┤
│ L2 表观层:说话人/姿态/光照/遮挡             │  ← 可靠模型与数据缓解
│    → 出路:对齐、解耦、自适应、局部-全局融合 │
├─────────────────────────────────────────────┤
│ L3 数据层:标注稀缺                          │  ← 已被自监督大幅缓解
│    → 出路:自监督、伪标签、语言先验          │
├─────────────────────────────────────────────┤
│ L4 工程层:预处理/帧率/端侧算力              │  ← 完全可控,但常被低估
│    → 出路:严格复现参考管线                  │
└─────────────────────────────────────────────┘
```

**一个重要观察**:2025 年的前沿工作(VALLR、Not Only Vision)已经把重心
从 L2/L3 转向了 **L1** —— 承认视觉信息不足,转而系统性地引入外部信息。
这是范式层面的转变。

**但同年的批判性研究给出了重要限定**(见 2.7):引入 LLM 带来的精度提升是**边际的**
(LRS3 上纯视觉最佳 18.7% vs 无 LLM 的 20.3%),其收益主要在**数据效率**而非绝对精度。
《From Hype to Insight》明确指出 **瓶颈仍在 L2 的视觉编码器**。

因此更准确的判断是:
```
L1(信息不足)   → LLM/上下文可以缓解,但有上限,且依赖足够的上下文长度
L2(视觉表观)   → 【当前真正的瓶颈】需要更强的视觉编码器
```
换言之,**L1 的补偿手段已趋于成熟,L2 反而成了新的主要矛盾**。

### 9.2 工程实践建议

| 场景 | 建议 |
|---|---|
| **产品级可用** | 词级编码器 + 少样本注册(封闭命令集),这是目前唯一能做到高准确率的路线 |
| **英文自由识别** | auto_avsr WER20.3 + CTC 贪心;务必用规整录制而非实时手持 |
| **中文自由识别** | 暂不成熟(CER 30%+),不建议作为产品核心功能 |
| **端侧部署** | 禁用 NNAPI、用 fp32、严格对齐预处理、保证 25fps |
| **选型优先级** | 数据规模 > 训练策略 > 模型架构 |
| **调试方法** | **把模型实际"看到"的画面导出成图** —— 本项目多个 bug 均由此定位 |

### 9.3 未来可能的突破口

1. **音素/视位作为中间表示** —— VALLR 已证明其数据效率优势
2. **多模态 LLM 端到端** —— 把视频 token 直接喂给 LLM
3. **个性化 + 自由识别的结合** —— 当前二者互斥:个性化准但闭集,自由识别开放但不准
4. **面向部署场景的数据集** —— 现有数据集(BBC/TED)与手机自拍差距过大
5. **中文的拼音/声母韵母中间表示** —— 绕开声调不可见的结构性方案

---

## 参考文献

### VIPL 视听语言感知与理解组
- 主页:https://vipl.ict.ac.cn/research/speech/
- 代码合集:https://github.com/VIPL-Audio-Visual-Speech-Understanding

| 论文 | 链接 |
|---|---|
| LRW-1000 (FG 2019) | https://arxiv.org/abs/1810.06990 |
| Can We Read Speech Beyond the Lips? (FG 2020) | https://arxiv.org/abs/2003.03206 |
| Deformation Flow Two-Stream (FG 2020) | https://arxiv.org/abs/2003.05709 |
| Mutual Information Maximization (FG 2020) | https://arxiv.org/abs/2003.06439 |
| Pseudo-Convolutional Policy Gradient (FG 2020) | https://arxiv.org/abs/2003.03983 |
| Synchronous Bidirectional Learning (BMVC 2020) | https://arxiv.org/abs/2005.03846 |
| Learn Effective Model without Pains | https://arxiv.org/abs/2011.07557 |
| Speaker-Adaptive Lip-Reading (BMVC 2023) | https://arxiv.org/abs/2310.05058 |
| UniLip (BMVC 2023) | https://proceedings.bmvc2023.org/190/ |
| ES³ (CVPR 2024) | https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_ES3_Evolving_Self-Supervised_Learning_of_Robust_Audio-Visual_Speech_Representations_CVPR_2024_paper.html |
| Audio-guided Disentangled (FCS 2024) | https://link.springer.com/article/10.1007/s11704-024-3787-8 |
| Not Only Vision (ICCV 2025) | https://openaccess.thecvf.com/content/ICCV2025/html/Yuan_Not_Only_Vision_Evolve_Visual_Speech_Recognition_via_Peripheral_Information_ICCV_2025_paper.html |
| GLip (BMVC 2025) | https://arxiv.org/abs/2509.16031 |

### 领域内其他关键工作
| 论文 | 链接 |
|---|---|
| AV-HuBERT (ICLR 2022) | https://arxiv.org/abs/2201.02184 |
| VALLR (ICCV 2025) | https://arxiv.org/abs/2503.21408 |
| **From Hype to Insight: Rethinking LLM Integration in VSR (2025)** | https://arxiv.org/abs/2509.14880 |
| Auto-AVSR | https://github.com/mpc001/auto_avsr |
| CNVSRC 2024 挑战赛 | https://arxiv.org/abs/2506.02010 |
| mpc001 TCN 系列 | https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks |

### 本项目实测来源
- 端侧移植:[LipLearner_Android/](LipLearner_Android/)
- 模型调研与实测:[OPEN_SOURCE_MODELS_zh.md](OPEN_SOURCE_MODELS_zh.md)
- 概览版:[VSR_RESEARCH_SURVEY_zh.md](VSR_RESEARCH_SURVEY_zh.md)
