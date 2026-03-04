# Tokenization技术

## 为什么需要Tokenization
- 神经网络无法直接处理文本
- 需要将文本转为数字ID序列
- 词表大小影响模型效率

## 主流方法

### BPE (Byte Pair Encoding)
- GPT系列使用
- 从字符开始，迭代合并高频pair
- 词表：50K tokens
- 优势：平衡词表大小和序列长度

### WordPiece
- BERT使用  
- 类似BPE，但用likelihood选择merge
- 子词：##ing, ##ed

### SentencePiece  
- 语言无关（包括中文）
- 直接在原始文本训练
- LLaMA使用

## 特殊Token
- [PAD]：填充
- [UNK]：未知词
- [CLS]：分类  
- [SEP]：分隔
- [MASK]：掩码

## 中文Tokenization挑战
- 无空格分词
- 需要分词器：jieba
- 字符级 vs 词级权衡
