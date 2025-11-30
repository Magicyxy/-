import torch
import torch.nn as nn
import torch.nn.functional as F

# 直接导入其他模块
import sys
import os

# 添加models目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 选择使用哪个编码器
# from sequence_encoder import SequenceEncoder  # 使用MobileNetV3版本
from sequence_encoder import ChemicalEquationCNN  # 使用专门为化学方程式设计的CNN
from graph_encoder import ChemicalAwareGraphEncoder


class CrossAttentionFusion(nn.Module):
    def __init__(self, seq_dim, graph_dim, hidden_dim=512, num_heads=8):
        super().__init__()

        self.seq_dim = seq_dim
        self.graph_dim = graph_dim
        self.hidden_dim = hidden_dim

        # 交叉注意力机制
        self.seq_to_graph_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.graph_to_seq_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # 投影层
        self.seq_proj = nn.Linear(seq_dim, hidden_dim)
        self.graph_proj = nn.Linear(graph_dim, hidden_dim)

        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, sequence_features, graph_features):
        batch_size, seq_len, _ = sequence_features.size()

        # 投影到相同维度
        seq_proj = self.seq_proj(sequence_features)
        graph_proj = self.graph_proj(graph_features).unsqueeze(1)
        graph_expanded = graph_proj.expand(-1, seq_len, -1)

        # 序列到图的交叉注意力
        seq_enhanced, _ = self.seq_to_graph_attention(
            query=seq_proj,
            key=graph_expanded,
            value=graph_expanded
        )

        # 图到序列的交叉注意力
        graph_enhanced, _ = self.graph_to_seq_attention(
            query=graph_expanded,
            key=seq_proj,
            value=seq_proj
        )

        # 门控融合
        combined = torch.cat([seq_enhanced, graph_enhanced], dim=-1)
        fusion_gate = self.gate(combined)

        fused_features = fusion_gate * seq_enhanced + (1 - fusion_gate) * graph_enhanced
        fused_features = self.layer_norm(fused_features)

        return fused_features


class ChemicalAwareTwoStreamModel(nn.Module):
    def __init__(self, num_chars, vocab_size, hidden_dim=512):
        super().__init__()

        # 使用专门为化学方程式设计的CNN编码器
        self.sequence_encoder = ChemicalEquationCNN(num_chars, hidden_dim)
        self.graph_encoder = ChemicalAwareGraphEncoder(vocab_size, hidden_dim)
        self.fusion_layer = CrossAttentionFusion(
            seq_dim=hidden_dim,
            graph_dim=hidden_dim,
            hidden_dim=hidden_dim
        )
        self.output_proj = nn.Linear(hidden_dim, num_chars)

    def forward(self, images, equation_tokens_list):
        # 序列流编码
        sequence_logits, sequence_features = self.sequence_encoder(images)

        # 图流编码
        graph_embeddings, node_info = self.graph_encoder(equation_tokens_list)

        # 双流融合
        fused_features = self.fusion_layer(sequence_features, graph_embeddings)

        # 最终输出
        final_logits = self.output_proj(fused_features)

        return final_logits, fused_features, node_info