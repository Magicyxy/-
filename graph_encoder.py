import torch
import torch.nn as nn
import torch.nn.functional as F


class ChemicalAwareGraphEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, num_heads=8, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 化学语义嵌入
        self.semantic_embedding = nn.Embedding(50, hidden_dim)
        self.atom_embedding = nn.Embedding(vocab_size, hidden_dim)

        # 化学语义映射
        self.semantic_to_idx = {
            'atom': 0, 'subscript': 1, 'charge': 2, 'reaction_condition': 3,
            'state_symbol': 4, 'operator': 5, 'number': 6,
            'subscript_2': 7, 'subscript_3': 8, 'subscript_4': 9, 'subscript_5': 10,
            'subscript_6': 11, 'subscript_7': 12,
            'heat_reaction': 13, 'high_temp_reaction': 14, 'light_reaction': 15,
            'electric_reaction': 16, 'ignite_reaction': 17,
            'charge_1+': 18, 'charge_2+': 19, 'charge_3+': 20, 'charge_4+': 21,
            'charge_5+': 22, 'charge_6+': 23, 'charge_7+': 24,
            'charge_1-': 25, 'charge_2-': 26, 'charge_3-': 27, 'charge_4-': 28,
            'charge_5-': 29, 'charge_6-': 30, 'charge_7-': 31,
            'gas': 32, 'precipitate': 33
        }

        # 自注意力层模拟图注意力（不使用掩码）
        self.self_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])

        # 化学规则编码器
        self.rule_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def get_semantic_type(self, token):
        """确定token的化学语义类型"""
        if token in ['_2', '_3', '_4', '_5', '_6', '_7']:
            return 'subscript'
        elif '|' in token or token in ['|+', '|-']:
            return 'charge'
        elif token in ['\\~=', '\\$=', '\\@=', '\\&=', '\\*=']:
            return 'reaction_condition'
        elif token in ['^', '!']:
            return 'state_symbol'
        elif token in ['+', '=']:
            return 'operator'
        elif token.isdigit():
            return 'number'
        else:
            return 'atom'

    def _get_atom_index(self, token):
        """基于提供的索引列表获取原子/符号的索引"""
        element_map = {
            'H': 0, '2': 1, '+': 2, 'C': 3, 'u': 4, 'O': 5, '=': 6, '^': 7,
            '\~=': 8, '\$=': 9, '3': 10, 'F': 11, 'e': 12, '4': 13, 'Z': 14,
            'n': 15, 'S': 16, 'M': 17, 'g': 18, 'A': 19, 'l': 20, '(': 21, ')': 22,
            '6': 23, 'N': 24, 'a': 25, 'z': 26, '!': 27, 'K': 28, 'P': 29, '5': 30,
            'D': 31, 'B': 32, '\@=': 33, 'r': 34, 'I': 35, 'b': 36, '\&=': 37,
            '_2': 38, '_3': 39, '_4': 40, '_5': 41, '_6': 42, '_7': 43, '\*=': 44,
            '|+': 45, '|2+': 46, '|3+': 47, '|4+': 48, '|5+': 49, '|-': 50,
            '|2-': 51, '|3-': 52, '|4-': 53, '|5-': 54, '|6+': 55, '|7+': 56,
            '|6-': 57, '|7-': 58, 'E': 59, 'G': 60, 'y': 61, 'J': 62, '.': 63,
            'L': 64, 'Q': 65, 'R': 66, 'T': 67, 'U': 68, 'V': 69, 'W': 70, 'X': 71,
            'Y': 72, 'c': 73, 'd': 74, 'f': 75, 'j': 76, 'h': 77, 'i': 78, 'k': 79,
            'm': 80, 'o': 81, 'p': 82, 'q': 83, 's': 84, 't': 85, 'v': 86, 'w': 87,
            'x': 88, '7': 89, '8': 90, '9': 91
        }
        return element_map.get(token, 0)  # 默认返回0（H）

    def forward(self, equation_tokens_list):
        batch_graph_embeddings = []
        batch_node_info = []

        for tokens in equation_tokens_list:
            node_features = []
            node_info = []

            # 创建节点特征
            for i, token in enumerate(tokens):
                semantic_type = self.get_semantic_type(token)
                semantic_idx = self.semantic_to_idx.get(semantic_type, 0)

                # 组合特征
                atom_idx = self._get_atom_index(token)
                atom_embed = self.atom_embedding(torch.tensor([atom_idx]))
                semantic_embed = self.semantic_embedding(torch.tensor([semantic_idx]))

                combined_embed = atom_embed + semantic_embed
                node_features.append(combined_embed.squeeze(0))

                node_info.append({
                    'token': token,
                    'semantic_type': semantic_type,
                    'position': i
                })

            if not node_features:
                graph_embedding = torch.zeros(self.hidden_dim)
                batch_graph_embeddings.append(graph_embedding)
                batch_node_info.append([])
                continue

            # 堆叠节点特征
            node_features = torch.stack(node_features).unsqueeze(0)  # [1, seq_len, hidden_dim]

            # 通过自注意力层模拟图神经网络（不使用掩码）
            x = node_features
            for attention_layer in self.self_attention_layers:
                attn_output, _ = attention_layer(x, x, x)
                x = self.layer_norm(x + self.dropout(attn_output))
                x = F.relu(x)

            # 全局平均池化得到图级别表示
            graph_embedding = x.mean(dim=1).squeeze(0)  # [hidden_dim]

            batch_graph_embeddings.append(graph_embedding)
            batch_node_info.append(node_info)

        # 合并批次
        graph_embeddings = torch.stack(batch_graph_embeddings)
        return graph_embeddings, batch_node_info