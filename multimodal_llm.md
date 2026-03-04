# 多模态大模型

## 1. 核心概念

**多模态大模型（Multimodal Large Language Model）** 是能够同时处理和理解多种模态数据（文本、图像、音频、视频）的AI系统。代表性模型包括CLIP、GPT-4V、Gemini等。

### 为什么需要多模态？

**单模态的局限**：
- 纯文本模型无法理解图片内容
- 纯视觉模型无法理解文字描述
- 人类感知是多模态的（看、听、读同时进行）

**多模态的优势**：
- ✅ 跨模态理解：看图说话、听声识物
- ✅ 零样本泛化：训练时见过图文对，测试时能处理新任务
- ✅ 统一表征：不同模态映射到同一语义空间

---

## 2. 核心技术架构

### 2.1 CLIP（Contrastive Language-Image Pre-training）

**提出者**：OpenAI, 2021年

**核心思想**：通过对比学习将图像和文本映射到同一特征空间。

**架构图**：
```
[文本编码器]           [图像编码器]
  (Transformer)         (ViT / ResNet)
       ↓                      ↓
  Text Embedding         Image Embedding
       ↓                      ↓
        ━━━━━━━━━━━━━━━━━━━━━━
              对比学习（Contrastive Loss）
              正样本：匹配的图文对 → 相似度高
              负样本：不匹配的图文对 → 相似度低
```

**训练数据**：4亿图文对（从互联网爬取）

**核心公式**：
```
相似度 = cos(Text_Embedding, Image_Embedding)
Loss = -log( exp(sim(i,i) / τ) / Σ exp(sim(i,j) / τ) )
```
其中τ是温度参数，控制分布陡峭程度。

---

### 2.2 Vision Transformer (ViT)

**核心创新**：将图像看作"token序列"，用纯Transformer处理视觉任务。

**实现步骤**：
1. **图像分块**：将224×224图像切成16×16的patches（共14×14=196个patch）
2. **线性投影**：每个patch展平为向量，通过FC层映射到768维
3. **位置编码**：加上可学习的位置向量（2D位置编码）
4. **Transformer编码**：送入标准Transformer Encoder（12层）
5. **分类头**：取[CLS] token输出，接FC层分类

**代码示例**（简化版）：
```python
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, dim=768, depth=12):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2  # 196
        
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, nhead=12),
            num_layers=depth
        )
        self.head = nn.Linear(dim, num_classes)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, 196, 768)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, 768)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 197, 768)
        x = x + self.pos_embed
        
        x = self.transformer(x)
        return self.head(x[:, 0])  # 取CLS token
```

---

### 2.3 跨模态特征对齐

**问题**：文本embedding和图像embedding如何对齐到同一语义空间？

**方法1：对比学习（CLIP采用）**
```python
# 伪代码
text_features = text_encoder(text)  # (B, 512)
image_features = image_encoder(image)  # (B, 512)

# L2归一化
text_features = F.normalize(text_features, dim=-1)
image_features = F.normalize(image_features, dim=-1)

# 计算相似度矩阵
logits = text_features @ image_features.T  # (B, B)

# 对比损失
labels = torch.arange(B).to(device)  # 对角线为正样本
loss = (cross_entropy(logits, labels) + cross_entropy(logits.T, labels)) / 2
```

**方法2：Cross-Attention（Flamingo采用）**
```python
class CrossAttentionLayer(nn.Module):
    def forward(self, text_hidden, image_hidden):
        # text作为Query，image作为Key和Value
        Q = self.q_proj(text_hidden)
        K = self.k_proj(image_hidden)
        V = self.v_proj(image_hidden)
        
        attention = softmax(Q @ K.T / sqrt(d_k))
        output = attention @ V
        return output
```

**方法3：Adapter（LLaVA采用）**
- 冻结LLM和视觉编码器
- 训练轻量级Adapter（MLP）将图像特征映射到LLM的token空间
- 图像特征作为"视觉token"直接拼接到文本序列中

---

## 3. 经典模型对比

| 模型 | 发布时间 | 架构特点 | 训练数据 | 主要能力 |
|------|---------|---------|---------|---------|
| **CLIP** | 2021.01 | 双塔结构（文本+图像编码器） | 4亿图文对 | 零样本图像分类、图文检索 |
| **DALL-E 2** | 2022.04 | 基于CLIP + Diffusion | CLIP特征空间 | 文生图、图像编辑 |
| **Flamingo** | 2022.04 | Perceiver + Cross-Attention | M3W数据集（43亿图文对） | Few-shot视觉问答 |
| **LLaVA** | 2023.04 | CLIP ViT + LLaMA + MLP Adapter | GPT-4生成的15.8万指令数据 | 视觉指令遵循、多轮对话 |
| **GPT-4V** | 2023.09 | 未公开（推测基于CLIP+GPT-4） | 未公开 | 通用视觉理解、OCR、图表分析 |
| **Gemini** | 2023.12 | 原生多模态Transformer | 未公开 | 文本、图像、音频、视频统一处理 |

---

## 4. 核心技术细节

### 4.1 视觉编码器选择

**ResNet vs ViT 对比**：

| 维度 | ResNet (CNN) | ViT (Transformer) |
|------|-------------|------------------|
| **归纳偏置** | 强（局部性、平移不变性） | 弱（纯Attention学习） |
| **数据需求** | 少（适合小数据集） | 多（需大规模预训练） |
| **计算效率** | 高（卷积局部计算） | 中（全局Attention开销大） |
| **长距离依赖** | 弱（需堆叠多层） | 强（一层即可全局建模） |
| **CLIP中表现** | 略低于ViT | 当前主流选择 |

**实际应用**：
- CLIP: ResNet-50 或 ViT-B/16
- LLaVA: CLIP ViT-L/14
- Gemini: 未公开（推测为定制ViT）

---

### 4.2 训练技巧

**1. 温度参数τ调优**
```python
# τ越小，相似度分布越陡峭，模型越自信
logits = similarity / temperature  # temperature通常为0.07
```

**2. 大批量训练**
- CLIP使用32,768的batch size
- 需要分布式训练（数百GPU）

**3. 数据增强**
```python
# 图像：随机裁剪、色彩抖动、高斯模糊
# 文本：保持原样（不做回译等增强，避免语义漂移）
```

---

### 4.3 零样本能力

**CLIP的零样本分类**：
```python
# 假设分类1000个ImageNet类别
text_prompts = ["a photo of a {class}".format(class=c) for c in classes]
text_features = clip_model.encode_text(text_prompts)  # (1000, 512)

image_feature = clip_model.encode_image(image)  # (1, 512)
similarities = image_feature @ text_features.T  # (1, 1000)
pred = similarities.argmax()
```

**性能对比（ImageNet Top-1）**：
- 监督学习ResNet-50：76.5%
- CLIP Zero-Shot ViT-L/14：**75.5%**（未见过ImageNet训练数据！）

---

## 5. 常见面试题

### Q1：CLIP是如何实现零样本学习的？

**答案**：
CLIP通过对比学习在4亿图文对上预训练，学习了**通用的图文对齐**能力。

**具体流程**：
1. 预训练阶段：学习图像和文本在语义空间的对应关系
2. 零样本推理：
   - 将类别名转为自然语言描述（如"一只猫" → "a photo of a cat"）
   - 用文本编码器生成类别文本的embedding
   - 计算图像embedding与所有类别embedding的相似度
   - 选择相似度最高的类别

**关键点**：
- 不需要针对下游任务Fine-tuning
- 泛化能力强（见过"猫"和"照片"的图文对，能理解"猫的照片"）

---

### Q2：Vision Transformer相比CNN有什么优缺点？

**答案**：

**优点**：
1. **全局建模能力强**：一层Attention即可捕捉图像全局依赖（CNN需堆叠多层）
2. **可扩展性好**：模型越大、数据越多，效果越好（CNN遇到瓶颈）
3. **统一架构**：文本和视觉都用Transformer，易于多模态融合

**缺点**：
1. **数据饥渴**：小数据集上表现不如CNN（缺少归纳偏置）
2. **计算开销大**：O(n²)的Attention复杂度（CNN是O(k²n），k为卷积核大小）
3. **位置编码敏感**：图像分辨率变化时需要插值位置编码

**实际应用建议**：
- 数据充足（>100万）且算力足够 → 优先ViT
- 数据稀缺或实时推理 → 考虑CNN或Hybrid模型

---

### Q3：多模态模型中如何实现跨模态特征对齐？

**答案**（来源：掘金面试题）：

**核心方法**：

**1. 对比学习（CLIP范式）**
- **正样本**：匹配的图文对（如猫的图片 + "一只猫"）
- **负样本**：batch内其他图文对
- **目标**：拉近正样本距离，推远负样本距离
- **损失函数**：InfoNCE Loss

**2. Cross-Attention（Flamingo范式）**
- 文本作为Query，图像作为Key/Value
- 在LLM的每几层插入Cross-Attention
- 优点：可处理多张图片、支持Few-shot

**3. Adapter/Projector（LLaVA范式）**
- 冻结预训练的LLM和视觉编码器
- 训练轻量级MLP将视觉特征映射到LLM token空间
- 优点：参数高效、训练快

**工程实践**：
```python
# LLaVA的Projector实现
class VisionProjector(nn.Module):
    def __init__(self, vision_dim=1024, llm_dim=4096):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )
    
    def forward(self, visual_features):
        return self.proj(visual_features)  # (B, N, 1024) -> (B, N, 4096)
```

---

### Q4：GPT-4V是如何处理图像的？

**答案**（基于公开信息推测）：

**架构猜测**：
1. **视觉编码器**：类似CLIP的ViT（可能更大，如ViT-G/14）
2. **跨模态融合**：
   - 方案A：图像token作为"前缀"，拼接到文本token序列中
   - 方案B：Cross-Attention层将视觉信息注入GPT-4
3. **多任务预训练**：图像描述、VQA、OCR、图表理解等

**能力验证**：
- ✅ OCR：识别图片中的文字（支持多语言）
- ✅ 图表分析：理解柱状图、折线图、表格
- ✅ 多轮对话：结合图片内容进行推理
- ✅ 空间理解：识别物体位置关系

**局限性**：
- ❌ 幻觉问题：可能编造图片中不存在的细节
- ❌ 数数困难：数图片中物体数量不准确
- ❌ 分辨率限制：超大图片会压缩损失细节

---

### Q5：多模态模型的评估指标有哪些？

**答案**：

**1. 图文检索任务**
- **Recall@K**：Top-K结果中包含正确答案的比例
- **MRR（Mean Reciprocal Rank）**：正确答案排名的倒数平均值

**2. 零样本分类**
- **Top-1 Accuracy**：最高预测类别的准确率
- **Top-5 Accuracy**：前5预测中包含正确类别的准确率

**3. 视觉问答（VQA）**
- **VQA Score**：考虑多个标注员答案的软匹配
- **BLEU / CIDEr**：生成文本与参考答案的相似度

**4. 图像描述生成**
- **CIDEr**：基于TF-IDF的n-gram匹配（常用）
- **SPICE**：基于场景图的语义相似度

**标准数据集**：
- ImageNet（分类）
- COCO（描述生成、VQA）
- Flickr30K（图文检索）
- VQAv2（视觉问答）

---

## 6. 前沿技术

### 6.1 长视频理解

**挑战**：
- 视频帧数多（30fps × 60秒 = 1800帧）
- 时间建模复杂

**方案**：
- **稀疏采样**：每秒取1-2帧
- **时序Transformer**：在ViT基础上加时间维度Attention
- **记忆机制**：用Memory Bank存储历史信息

---

### 6.2 音频-文本多模态

**代表模型**：
- **Whisper**（OpenAI）：语音识别 + 翻译
- **AudioLM**（Google）：音频生成
- **Qwen-Audio**：音频理解 + 对话

**关键技术**：
- Mel-Spectrogram特征提取
- 时序建模（RNN / Transformer Encoder）
- 跨模态对齐（音频embedding ↔ 文本embedding）

---

### 6.3 端到端多模态（Gemini方向）

**核心思想**：
- 不分别训练视觉/文本编码器，而是**从头训练统一的多模态Transformer**
- 不同模态共享同一套参数

**优势**：
- 模态间交互更充分
- 避免预训练模型的偏差累积

**挑战**：
- 训练成本极高（需同时收集图文音数据）
- 架构设计复杂（如何统一不同模态的输入格式）

---

## 7. 实战案例

### 案例1：用CLIP做图像相似度搜索

```python
import clip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 编码图片库
image_database = [...]  # 1000张图片
image_features = []
for img_path in image_database:
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model.encode_image(image)
    image_features.append(feature)

image_features = torch.cat(image_features)  # (1000, 512)

# 搜索查询
query = "a red car"
text = clip.tokenize([query]).to(device)
text_feature = model.encode_text(text)

# 计算相似度
similarities = (text_feature @ image_features.T).squeeze()
top5_indices = similarities.topk(5).indices
print(f"Top 5 similar images: {top5_indices}")
```

---

### 案例2：LLaVA风格的视觉问答

```python
# 伪代码
vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
llm = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
projector = VisionProjector(vision_dim=1024, llm_dim=4096)

# 输入：图片 + 问题
image = load_image("cat.jpg")
question = "What is in this image?"

# 视觉特征提取
visual_features = vision_encoder(image)  # (1, 196, 1024)
visual_tokens = projector(visual_features)  # (1, 196, 4096)

# 文本token化
text_tokens = llm.tokenizer(question)  # (1, 10)

# 拼接：[visual_tokens] + [text_tokens]
input_embeds = torch.cat([visual_tokens, text_tokens], dim=1)

# LLM生成
output = llm.generate(inputs_embeds=input_embeds, max_length=50)
answer = llm.tokenizer.decode(output[0])
print(answer)  # "There is a cat sitting on a couch."
```

---

## 8. 学习路径

### 初级（1-2周）
- [ ] 理解CLIP的核心原理和对比学习
- [ ] 掌握Vision Transformer基础架构
- [ ] 实践CLIP的图文检索任务

### 中级（3-4周）
- [ ] 对比ResNet vs ViT的优劣
- [ ] 学习Cross-Attention机制
- [ ] 了解LLaVA/Flamingo的架构差异

### 高级（5-6周）
- [ ] 研究GPT-4V的能力边界
- [ ] 实现轻量级多模态模型（冻结大模型+Adapter）
- [ ] 探索视频/音频多模态扩展

---

## 参考资料

- [CLIP论文](https://arxiv.org/abs/2103.00020) - Learning Transferable Visual Models From Natural Language Supervision
- [ViT论文](https://arxiv.org/abs/2010.11929) - An Image is Worth 16x16 Words
- [LLaVA论文](https://arxiv.org/abs/2304.08485) - Visual Instruction Tuning
- 掘金：Vision-Language模型中如何实现跨模态特征对齐
- 今日头条：MIT分享的50个LLM面试题
