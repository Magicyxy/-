import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import re


class ChemicalEquationDataset(Dataset):
    def __init__(self, image_dir, label_dir, classes_file, labels_file=None, transform=None, img_size=(32, 128)):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_size = img_size

        # 加载类别映射 - 严格按照提供的索引顺序
        with open(classes_file, 'r', encoding='utf-8') as f:
            self.classes = [line.strip() for line in f.readlines()]

        # 确保类别数量正确
        if len(self.classes) != 80:
            print(f"Warning: Expected 80 classes, but got {len(self.classes)}")

        self.char_to_idx = {char: idx for idx, char in enumerate(self.classes)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.classes)}

        # 化学语义映射（基于准确索引）
        self.chemical_semantics = {
            '_2': 'subscript_2', '_3': 'subscript_3', '_4': 'subscript_4',
            '_5': 'subscript_5', '_6': 'subscript_6', '_7': 'subscript_7',
            '\\~=': 'heat_reaction', '\\$=': 'high_temp_reaction',
            '\\@=': 'light_reaction', '\\&=': 'electric_reaction',
            '\\*=': 'ignite_reaction',
            '|+': 'charge_1+', '|2+': 'charge_2+', '|3+': 'charge_3+',
            '|4+': 'charge_4+', '|5+': 'charge_5+', '|6+': 'charge_6+',
            '|7+': 'charge_7+',
            '|-': 'charge_1-', '|2-': 'charge_2-', '|3-': 'charge_3-',
            '|4-': 'charge_4-', '|5-': 'charge_5-', '|6-': 'charge_6-',
            '|7-': 'charge_7-',
            '^': 'gas', '!': 'precipitate'
        }

        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

        self.labels_dict = {}
        if labels_file and os.path.exists(labels_file):
            with open(labels_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if '\t' in line:
                        img_name, equation = line.strip().split('\t')
                        self.labels_dict[img_name] = equation

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # 加载图像
        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        # 获取真实标签
        if img_name in self.labels_dict:
            equation_str = self.labels_dict[img_name]
            equation_tokens = self._tokenize_with_semantics(equation_str)
        else:
            # 从YOLO标签重建
            label_file = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
            equation_tokens = self._reconstruct_from_yolo(label_file)

        target_indices = [self.char_to_idx.get(token, 0) for token in equation_tokens]
        target_tensor = torch.tensor(target_indices, dtype=torch.long)

        return image, target_tensor, equation_tokens

    def _tokenize_with_semantics(self, equation_str):
        """基于化学语义的tokenization"""
        special_tokens = sorted([
            '\\~=', '\\$=', '\\@=', '\\&=', '\\*=',
            '|2+', '|3+', '|4+', '|5+', '|6+', '|7+',
            '|2-', '|3-', '|4-', '|5-', '|6-', '|7-',
            '_2', '_3', '_4', '_5', '_6', '_7',
            '|+', '|-'
        ], key=len, reverse=True)

        tokens = []
        i = 0
        while i < len(equation_str):
            matched = False
            for special in special_tokens:
                if equation_str.startswith(special, i):
                    tokens.append(special)
                    i += len(special)
                    matched = True
                    break

            if not matched:
                char = equation_str[i]
                tokens.append(char)
                i += 1

        return tokens

    def _reconstruct_from_yolo(self, label_file):
        """从YOLO标签重建token序列"""
        if not os.path.exists(label_file):
            return []

        bboxes = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    bboxes.append({
                        'class_id': class_id,
                        'x_center': float(parts[1]),
                        'token': self.idx_to_char.get(class_id, '?')
                    })

        # 按x坐标排序得到序列
        bboxes.sort(key=lambda x: x['x_center'])
        return [bbox['token'] for bbox in bboxes]

    def collate_fn(self, batch):
        """自定义批次处理函数"""
        images, targets, tokens = zip(*batch)

        # 填充图像到相同尺寸
        images_batch = torch.stack(images)

        # 填充目标序列
        target_lengths = [len(t) for t in targets]
        max_target_len = max(target_lengths)

        padded_targets = torch.zeros(len(targets), max_target_len, dtype=torch.long)
        for i, target in enumerate(targets):
            padded_targets[i, :len(target)] = target

        return images_batch, padded_targets, tokens, target_lengths