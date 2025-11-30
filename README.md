# 双流图序列化学方程式识别系统

基于改进CRNN模型的手写化学方程式识别方法研究

## 模型架构

- #### 1. 双流联合编码器结构

  ```plaintext
  输入: 化学方程式图像 + 文本Token序列
  ├─ 序列编码流 (Sequence Stream)
  │  ├─ 特征提取: MobileNetV3M风格卷积层
  │  │  ├─ Conv2d(1→32) → BatchNorm → ReLU → MaxPool2d
  │  │  ├─ Conv2d(32→64) → BatchNorm → ReLU → MaxPool2d
  │  │  ├─ Conv2d(64→128) → BatchNorm → ReLU → 特殊MaxPool2d (保持序列长度)
  │  │  └─ Conv2d(128→256) → Conv2d(256→512) → BatchNorm → ReLU → 特殊MaxPool2d
  │  ├─ 序列建模: BiGRU (或单向LSTM)
  │  │  └─ 输入: [batch, seq_len, 512] → 输出: [batch, seq_len, 512] (序列特征)
  │  └─ 输出: 序列特征 (sequence_features, 形状 [batch, 33, 512])
  │
  ├─ 图编码流 (Graph Stream)
  │  ├─ 节点特征构建
  │  │  ├─ 原子嵌入 (atom_embedding): Token→索引→512维向量
  │  │  ├─ 语义嵌入 (semantic_embedding): 化学类型(如电荷、下标)→512维向量
  │  │  └─ 组合: 原子嵌入 + 语义嵌入 → 节点特征
  │  ├─ 图注意力建模: 多层多头自注意力 (GATv2风格)
  │  │  └─ 输入: 节点特征序列 → 输出: 增强的节点特征
  │  └─ 输出: 图嵌入 (graph_embeddings, 形状 [batch, 512]) (全局平均池化节点特征)
  │
  └─ 双流融合层
     ├─ 特征投影: 
     │  ├─ 序列特征 → seq_proj (512→512, 保持序列长度)
     │  └─ 图嵌入 → graph_expanded (扩展为 [batch, 33, 512]，与序列长度匹配)
     ├─ 交叉注意力:
     │  ├─ 序列→图注意力: 用图特征增强序列特征
     │  └─ 图→序列注意力: 用序列特征增强图特征
     ├─ 门控融合:
     │  └─ 融合门控 (sigmoid) → 加权合并增强后的序列和图特征
     └─ 输出: 融合特征 (fused_features, 形状 [batch, 33, 512])
  ```

  #### 2. CTC + CRF 混合解码器结构

  ```plaintext
  输入: 融合特征 (fused_features) + 目标序列 (targets) + 节点信息 (node_info)
  ├─ 输出投影: 融合特征 → logits (形状 [batch, 33, 92]，92为类别数)
  │
  ├─ CTC损失计算
  │  ├─ 输入: logits → log_softmax → 维度转换为 [T, N, C]
  │  ├─ 目标序列处理: 移除padding，计算目标长度 (target_lengths)
  │  └─ CTC损失: 基于帧级别预测与目标序列的对齐误差
  │
  ├─ 化学规则惩罚项
  │  ├─ 输入: 预测序列、目标序列、节点信息 (含Token语义)
  │  ├─ 规则检查:
  │  │  ├─ 下标后接左括号 (如"_2(") → 惩罚
  │  │  ├─ 反应条件在开头 (如"\\~=...") → 惩罚
  │  │  └─ 电荷单独出现 (如开头为"|+") → 惩罚
  │  └─ 输出: 惩罚值 (chem_penalty)
  │
  ├─ CRF序列约束 (隐含在解码过程)
  │  └─ 对预测序列施加转移概率约束 (如合理的Token接续关系)
  │
  └─ 总损失: CTC损失 + 0.1×化学规则惩罚项 → 用于模型优化
     解码输出: 贪婪解码 (取logits最大值) → 预测字符序列
  ```
  <img width="1000" height="500" alt="loss_curve" src="https://github.com/user-attachments/assets/9feb9581-7416-4f5c-aea3-54ada186596b" />
![accuracy](https://github.com/user-attachments/assets/20995519-0f15-4fc5-9386-af85357d8cd1)

  ![accuracy](https://github.com/user-attachments/assets/9835f70f-73eb-416f-ba35-03c22e157759)
<img width="1200" height="1500" alt="prediction_samples" src="https://github.com/user-attachments/assets/2a504ce2-3ecf-4427-894c-b103545f240a" />
