# 唇读 / 视觉语音识别(VSR)开源模型调研与实测报告

本文整理本项目调研过的全部开源唇读模型:**能否下载、许可如何、是否适合端侧、以及实际测试结果**。
所有"实测"部分均为在本项目中真实运行得到的数据(桌面 macOS M4 + nubia P0110 / Android 16 / arm64)。

> 阅读提示:唇读模型分两大类,能力和用法完全不同 ——
> **① 词级编码器**(输出定长向量,配合少样本学习做自定义命令)
> **② 句子级 VSR**(开放词汇,直接输出文本)
> 选型前先明确你要哪一类。

---

## 目录

- [一、总览对比](#一总览对比)
- [二、词级模型(少样本命令定制)](#二词级模型少样本命令定制)
- [三、句子级模型(开放词汇自由识别)](#三句子级模型开放词汇自由识别)
- [四、实测结果详录](#四实测结果详录)
- [五、端侧部署踩坑记录](#五端侧部署踩坑记录重要)
- [六、许可与合规](#六许可与合规)
- [七、选型建议](#七选型建议)

---

## 一、总览对比

| 模型 | 类型 | 语言 | 能下载 | 已实测 | 端侧可用 |
|---|---|---|---|---|---|
| [mpc001 TCN 系列](#21-mpc001--lipreading_using_temporal_convolutional_networks) | 词级(LRW 500词) | 英 | ✅ 直链 | ✅ **已集成 APK** | ✅ 12–145MB |
| [VIPL Feng](#22-vipl--learn-an-effective-lip-reading-model-without-pains) | 词级(LRW-1000) | 中 | ❌ 仅百度盘 | ❌ 未获取 | 未知 |
| [LipLearner 原版](#23-liplearner-原版编码器) | 词级(对比学习) | — | ❌ 登录墙 | ❌ 未获取 | 理论可用 |
| [auto_avsr](#31-auto_avsr英文句子级) | 句子级 | 英 | ✅ 直链 | ✅ **已集成 APK** | ⚠️ 见下文 |
| [CNVSRC baseline](#32-cnvsrc-baseline中文句子级) | 句子级 | 中 | ✅ HuggingFace | ⚠️ 部分验证 | ⚠️ 见下文 |
| [AV-HuBERT](#33-av-hubert未采用) | 自监督表示 | 英 | ✅ 需同意许可 | ❌ 未采用 | ❌ 过重 |

---

## 二、词级模型(少样本命令定制)

这类模型输出**定长特征向量**,本身不直接"读出话",而是作为编码器,配合用户注册的少量样本训练轻量分类器 —— 即 LipLearner 的做法。

### 2.1 mpc001 / Lipreading_using_Temporal_Convolutional_Networks

- **仓库**:https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks
- **论文**:ICASSP'20/'21,含《Towards Practical Lipreading with Distilled and Efficient Models》
- **训练数据**:LRW(500 个英文词,BBC 新闻)
- **下载**:README 的 Model Zoo 表格,bit.ly 短链 → Google Drive,**免登录直接可下**

| 模型 | 结构 | LRW 准确率 | 原始 .pth | 导出 ONNX |
|---|---|---|---|---|
| `snv05x_tcn1x` | ShuffleNetV2 0.5× + TCN | 79.9% | 11.8MB | 12MB |
| `snv1x_tcn1x` | ShuffleNetV2 1× + TCN | 82.7% | 15.5MB | 15MB |
| `snv1x_dsmstcn3x` | ShuffleNet + DS-MSTCN | 85.3% | 37.5MB | 37MB |
| `resnet18_mstcn` | ResNet18 + MS-TCN | 88.9% | 146MB | 145MB |

**实测结论:✅ 4 个全部下载成功、成功导出 ONNX、已集成进 APK 并在真机运行。**

导出细节(见 `tools/export_mpc001_onnx.py`):
- 归一化 `(x/255 - 0.421)/0.165` 已固化进 ONNX 图,客户端只需送 [0,1] 灰度
- ⚠️ **ShuffleNet 的 channel-shuffle 把 batch 维写死在 Reshape 里**,导致动态时间轴导出后在 T≠29 时报错。解决:导出为**固定 29 帧**图(模型本就在 29 帧 LRW 片段上训练),客户端把任意长度重采样到 29 帧
- torch ↔ ONNXRuntime 数值误差 < 1.4e-04

**真机识别实测**(nubia P0110,`snv05x_tcn1x`,自注册命令):

| 命令数 | 样本数 | 典型置信度 | 表现 |
|---|---|---|---|
| 2 条 | 7 | 0.89 ~ 0.99 | 好 |
| 6 条 | 31 | 0.35 ~ 0.83 | 明显下降,唇形相近的词(调高音量/降低音量)易混 |

端上分类器训练实测(300 轮全批量梯度下降):
```
epoch 0    loss=1.7918
epoch 299  loss=0.3878     → 6 类 / 31 样本 / 更新 3006 个参数
```
编码延迟约 **40–60ms**,松手到出结果几乎无感。

---

### 2.2 VIPL / learn-an-effective-lip-reading-model-without-pains

- **仓库**:https://github.com/VIPL-Audio-Visual-Speech-Understanding/learn-an-effective-lip-reading-model-without-pains
- **说明**:**这是 LipLearner 论文所用编码器的底座**(Feng et al.)
- **训练数据**:LRW / LRW-1000(中文 1000 词)
- **下载**:❌ **仅提供百度网盘**(提取码 `ivgl`),需百度账号
- **实测结论:未能获取。** 本项目环境无法下载百度盘资源
- ⚠️ **重要澄清**:这是**词级闭集分类器**(1000 个固定中文词),**不是**中文自由识别模型。若需中文开放词汇,应使用 CNVSRC

---

### 2.3 LipLearner 原版编码器

- **来源**:本仓库 README 的 Google Drive 链接(`LipLearner_pretrained_model.pt`)
- **特点**:ResNet18 + BiGRU,**对比学习(InfoNCE)训练**,输出 500 维度量空间嵌入
- **下载**:❌ **被 Google 登录墙/配额限制拦截**(匿名下载只返回登录页)
- **实测结论:未能获取。** 需用户手动从浏览器下载
- **价值**:因为是对比学习训练的**度量空间**,理论上少样本效果优于把分类器倒数第二层当嵌入用(即 mpc001 的用法)。工具已备好(`LipLearner_Android/scripts/build_model.sh`),拿到权重即可接入

---

## 三、句子级模型(开放词汇自由识别)

这类模型**无需注册命令**,直接输出任意句子文本。代价是模型大、解码复杂。

### 3.1 auto_avsr(英文句子级)

- **仓库**:https://github.com/mpc001/auto_avsr
- **结构**:ResNet 前端 + **Conformer 编码器(12层/768维)** + Transformer 解码器 + CTC
- **词表**:5049 SentencePiece unigram
- **下载**:✅ Google Drive 直链,免登录

| 权重 | 训练数据 | LRS3 WER | 大小 |
|---|---|---|---|
| `vsr_trlrs3_base` | LRS3 438h | 36.0% | 955MB |
| `vsr_trlrs2lrs3vox2avsp_base` | 3291h(LRS2+LRS3+VoxCeleb2+AVSpeech) | **20.3%** | 955MB |

> 注:两个权重体积相同,差别在**训练数据量**而非模型规模。

**实测结论:✅ 已跑通并集成进 APK。**

关键实测数据(同一段手机录制的英文视频,内容 `PLEASE CALL THE DOCTOR`):

| 配置 | 输出 | 耗时 |
|---|---|---|
| WER36 模型 + 完整解码 | `IS ALL THAT` ❌ | 3.9s |
| **WER20.3 模型 + 完整解码** | **`PLEASE CALL THE DOCTOR`** ✅ | 2.7s |
| **WER20.3 模型 + CTC 贪心** | **`PLEASE CALL YOUR DOCTOR`** ✅ | 0.9s |
| 导出 ONNX + CTC 贪心(端侧同款) | `PLEASE CALL YOUR DOCTOR` ✅ | **0.36s** |

**两个重要发现:**
1. **模型强弱差距巨大** —— WER36 全错,WER20.3 全对。别用小模型
2. **CTC 贪心足够可用** —— 只比完整 beam search 差一个词,但**是单次前向**,能上端侧(完整解码需要自回归 + beam search,无法用 ONNX 静态图表达)

**端侧化方案**(`tools/free_vsr/export_ctc_onnx_en.py`):
- 只导出 **编码器 + CTC 头**,丢掉自回归解码器 → 单次前向,775MB
- 客户端 CTC 贪心解码(argmax → 去重 → 去 blank),几十行 Kotlin

---

### 3.2 CNVSRC baseline(中文句子级)

- **仓库**:https://github.com/DataoceanAI/CNVSRC2023Baseline
- **背景**:CNVSRC 中文连续视觉语音识别挑战赛官方基线
- **结构**:与 auto_avsr 同构(Conformer + CTC/Attention),**词表 5906 个汉字**
- **训练数据**:CN-CVS(中文大词汇视听数据集)
- **下载**:✅ **HuggingFace `DataOceanAI/CNVSRC2023Baseline`**(也有 ModelScope),公开可下

**实测结论:⚠️ 已导出并集成,但准确率未达可用水平。**

已完成并验证:
- ✅ ONNX 导出 777MB,torch↔ORT 误差 1.7e-04
- ✅ 预处理与英文一致(96×96 平均脸对齐)
- ✅ 手机端与桌面**结果完全一致**(同数据同模型,输出均为 `力啊`)→ 端侧实现无 bug
- ✅ 帧率 25.3fps(达标)

未达成:
- ❌ 实时识别输出无意义(如 `因为男人叶我儿`、`我大了后啊`)
- ❌ **从未在"已知正确答案"的中文视频上验证过** —— 因此**无法断定**是模型泛化能力不足,还是输入条件不合适

补充实验(排除了一个假设):中文模型对裁剪方式**不敏感**。同一视频用 68 点(其训练约定)和 4 点(英文约定)对齐,输出分别为 `没享受好听` / `没好一点`,同为无意义中文 → 说明用 4 点对齐不是问题所在。

---

### 3.3 AV-HuBERT(未采用)

- **仓库**:https://github.com/facebookresearch/av_hubert(2024-09 已归档只读)
- **类型**:自监督视听表示学习,迁移性强
- **下载**:✅ 需点击同意许可
- **未采用原因**:fairseq 框架、模型过大、ONNX 导出困难、**CC-BY-NC 非商用**

---

## 四、实测结果详录

### 4.1 英文自由 VSR 端到端(已跑通)

真机 nubia P0110,实时按住说话:

| 阶段 | 结果 |
|---|---|
| 采集帧率 | 30.4 fps(优化后) |
| 单帧耗时 | convert 14ms + detect 9ms + align 4ms ≈ 27ms |
| 推理 | 0.4–1.3s(CPU,fp32) |
| 手机 vs 桌面 | **完全一致**(同帧数据交叉验证) |

识别质量:**视频文件路径明显优于实时按住**。实时输出如 `BE CALLED TO`、`PEACE IS CALLED CORNER`(核心词 CALL 稳定命中,但整句不准)。

### 4.2 少样本命令识别(已跑通,推荐)

见 [2.1](#21-mpc001--lipreading_using_temporal_convolutional_networks) 实测表。**2 条命令时置信度 0.9+,实用**。

### 4.3 中文自由 VSR

见 [3.2](#32-cnvsrc-baseline中文句子级)。端侧实现正确,但识别质量未达可用,**结论待定**。

---

## 五、端侧部署踩坑记录(重要)

这些坑耗费了大量调试时间,若你要做类似移植,建议先看这里。

### 坑 1:NNAPI(NPU 加速)静默算错 ⚠️ 最隐蔽

**现象**:同一份输入、同一个 ONNX,桌面 CPU 输出 `PLEASE CALLED THE DAUGHTER`,手机开 NNAPI 输出 `YORK`。

**原因**:Conformer 的**动态长度注意力** NNAPI 支持不完整。ONNX Runtime 会切分计算图,部分算子跑 NPU、部分回退 CPU,结果被破坏 —— **且不报任何错**。

**解决**:禁用 NNAPI,用 CPU。fp32 单次前向约 0.4s,速度完全够用。

### 坑 2:fp16 在 ARM 上精度崩溃

**现象**:fp16 模型在 x86 桌面结果正确,在手机 ARM 上与桌面不一致。

**原因**:x86 上 ORT 把 fp16 转 fp32 计算;ARM 有原生 fp16 指令,12 层 Conformer 累积误差被放大。

**解决**:端侧用 **fp32**(体积翻倍但正确)。fp16 量化只适合桌面。

### 坑 3:视频旋转元数据未应用

**现象**:识别输出完全无意义。

**原因**:手机竖屏录制的 mp4 带 `rotation=90` 元数据,PyAV 解码**不自动应用**,送进模型的是躺倒画面,人脸对齐全崩(裁到了鼻子)。

**解决**:改用 OpenCV 解码(自动应用旋转)。修复后耗时也从 17.5s 降到 2.7s。

### 坑 4:帧率不匹配导致语速失真

**现象**:管线全对但识别不准。

**原因**:模型按 **25fps** 训练,而 CameraX 分析流因每帧处理过重只能到 **16fps**,却被当作连续帧送入 → 模型认为语速快了近一倍。

**解决**:
1. 按**时间戳重采样到 25fps**
2. 降低每帧开销:用 **BlazeFace(4关键点,~9ms)** 替代 **Face Mesh(468点,~25ms)**
3. 分析流用 720p 而非 1080p(1080p 时整帧旋转镜像要 24ms)

### 坑 5:必须复现训练时的人脸对齐

**现象**:用"唇部中心正方形"裁剪,识别输出垃圾。

**原因**:模型训练于**仿射对齐到平均脸**后的裁剪,不是简单的嘴部方框。

**解决**:在 Kotlin 复现完整流程 —— 关键点 → 时间平滑 → Umeyama 相似变换对齐到平均脸(256×256)→ 裁 96×96 → 中心裁 88。
⚠️ 裁剪中心须用**变换后的实际关键点**,不能用参考平均脸的位置(68 点拟合时偏差明显)。

### 坑 6:大模型加载 OOM

**现象**:App 启动即崩。

**原因**:`assets.readBytes()` 把 145MB 模型一次性读进 Java 堆(上限 256MB)。

**解决**:流式落盘 + ONNX Runtime 从**文件路径 mmap**(不占 Java 堆)。

---

## 六、许可与合规

⚠️ **本文所有模型权重均为研究/非商用**,这是唇读领域的普遍情况:

| 模型 | 代码许可 | 权重限制 |
|---|---|---|
| mpc001 TCN | — | LRW 派生,**非商用** |
| auto_avsr | Apache 2.0 | LRS2/LRS3/VoxCeleb2 派生,受数据集条款约束 |
| CNVSRC | — | CN-CVS 派生,研究用途 |
| AV-HuBERT | — | **CC-BY-NC 明确非商用** |
| VIPL Feng | MIT(代码) | LRW-1000 派生 |

**要商用,必须在许可宽松的数据上自行训练。** 现有开源权重无一可直接商用。

---

## 七、选型建议

| 你的目标 | 推荐 |
|---|---|
| **手机端可用的产品** | **mpc001 词级编码器 + 少样本注册**(本项目已验证可用,置信度 0.9+) |
| 英文自由识别、可接受不完美 | auto_avsr WER20.3 + CTC 贪心;**用视频文件路径效果远好于实时** |
| 中文自由识别 | CNVSRC 是目前唯一可下的开放词汇中文模型,但**本项目尚未验证其在手机自拍场景可用** |
| 追求最高精度、不限体积 | auto_avsr 完整解码(注意力+beam search),仅限服务端 |
| 商用 | 以上均不可直接用,需自行采集数据训练 |

### 一条重要经验

**"能识别"和"识别得准"之间隔着大量工程细节。** 本项目中,同一个模型、同一段视频,仅因为
旋转元数据、帧率、对齐方式、NNAPI、fp16 这五个因素,输出可以从 `PLEASE CALL YOUR DOCTOR`
退化到 `YORK`。移植唇读模型时,**务必逐帧比对你的预处理与参考实现** —— 把模型实际"看到"的
画面导出成图片检查,是最高效的排错手段(本项目多个 bug 均由此定位)。

---

## 附:本项目相关脚本

| 脚本 | 用途 |
|---|---|
| `tools/export_mpc001_onnx.py` | mpc001 词级模型 → ONNX |
| `tools/export_onnx.py` | LipLearner 原版编码器 → ONNX |
| `tools/free_vsr/export_ctc_onnx_en.py` | auto_avsr 编码器+CTC → ONNX |
| `tools/free_vsr/export_ctc_onnx_cn.py` | CNVSRC 编码器+CTC → ONNX |
| `tools/free_vsr/run_demo_en.py` | 桌面英文识别(完整解码 vs CTC 贪心对比) |
| `tools/free_vsr/run_demo_cn.py` | 桌面中文识别 |
| `tools/free_vsr/dump_crops.py` | **导出模型实际看到的裁剪画面**(排错利器) |
| `tools/free_vsr/test_onnx_ctc_en.py` | 验证 ONNX 端到端与 PyTorch 一致 |
