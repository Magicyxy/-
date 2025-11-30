import torch
import torch.nn as nn
import torch.nn.functional as F


class ChemicalAwareCRFDecoder(nn.Module):
    def __init__(self, num_chars, blank_idx=0):
        super().__init__()
        self.num_chars = num_chars
        self.blank_idx = blank_idx

        self.ctc_loss = nn.CTCLoss(blank=blank_idx, reduction='mean')

        # 字符映射（在训练时设置）
        self.char_to_idx = None
        self.idx_to_char = None

    def set_char_mappings(self, char_to_idx, idx_to_char):
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char

    def _apply_chemical_penalties(self, log_probs, targets, node_info):
        """应用化学规则惩罚"""
        if node_info is None:
            return 0

        batch_penalties = 0

        for batch_idx in range(log_probs.size(0)):
            if batch_idx >= len(node_info):
                continue

            current_node_info = node_info[batch_idx]
            seq_len = log_probs.size(1)

            for t in range(min(seq_len - 1, len(current_node_info) - 1)):
                current_char_idx = targets[batch_idx, t].item() if t < targets.size(1) else self.blank_idx
                next_char_idx = targets[batch_idx, t + 1].item() if t + 1 < targets.size(1) else self.blank_idx

                penalty = self._check_chemical_violation(current_char_idx, next_char_idx, current_node_info, t)
                batch_penalties += penalty

        return batch_penalties

    def _check_chemical_violation(self, current_char_idx, next_char_idx, node_info, position):
        """检查化学规则违反"""
        if position >= len(node_info) or position + 1 >= len(node_info):
            return 0

        current_token = node_info[position][
            'token'] if self.idx_to_char and current_char_idx in self.idx_to_char else ''
        next_token = node_info[position + 1]['token'] if self.idx_to_char and next_char_idx in self.idx_to_char else ''

        penalty = 0

        # 规则1: 下标后不能接左括号
        if current_token.startswith('_') and next_token == '(':
            penalty += 1.0

        # 规则2: 反应条件不能在开头
        if position == 0 and any(symbol in current_token for symbol in ['\\~=', '\\$=', '\\@=', '\\&=', '\\*=']):
            penalty += 1.0

        # 规则3: 电荷不能单独出现
        if '|' in current_token and position == 0:
            penalty += 1.0

        return penalty

    def forward(self, logits, targets, input_lengths, target_lengths, node_info=None):
        log_probs = F.log_softmax(logits, dim=2)
        log_probs_permuted = log_probs.permute(1, 0, 2)

        ctc_loss = self.ctc_loss(log_probs_permuted, targets, input_lengths, target_lengths)

        # 化学规则惩罚
        chem_penalty = self._apply_chemical_penalties(log_probs, targets, node_info)

        total_loss = ctc_loss + 0.1 * chem_penalty

        return total_loss

    def decode(self, logits):
        """使用贪婪解码"""
        _, predictions = torch.max(logits, 2)
        return predictions.tolist()