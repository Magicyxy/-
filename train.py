import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import json

# 导入模型 - 使用直接导入方式
import sys
import os

sys.path.append('./models')
sys.path.append('./dataset')

from models.fusion_encoder import ChemicalAwareTwoStreamModel
from models.ctc_crf_decoder import ChemicalAwareCRFDecoder
from dataset.chem_dataset import ChemicalEquationDataset


def train():
    # 超参数
    batch_size = 8
    learning_rate = 1e-4
    num_epochs = 100
    hidden_dim = 512

    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 数据集
    train_dataset = ChemicalEquationDataset(
        image_dir='dataset/images/',
        label_dir='dataset/labels/',
        classes_file='dataset/classes.txt',
        labels_file='dataset/labels.txt',
        transform=transform
    )

    # 使用准确的类别数量（80个类别）
    num_chars = len(train_dataset.classes)
    vocab_size = num_chars  # 词汇表大小等于类别数量

    print(f"Number of character classes: {num_chars}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )

    # 模型 - 使用准确的类别数量
    model = ChemicalAwareTwoStreamModel(num_chars, vocab_size, hidden_dim)
    decoder = ChemicalAwareCRFDecoder(num_chars)
    decoder.set_char_mappings(train_dataset.char_to_idx, train_dataset.idx_to_char)

    # 优化器
    optimizer = optim.Adam(
        list(model.parameters()) + list(decoder.parameters()),
        lr=learning_rate,
        weight_decay=1e-5
    )

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # 训练记录
    train_losses = []

    # 创建结果目录
    os.makedirs('results', exist_ok=True)

    # 训练循环
    for epoch in range(num_epochs):
        model.train()

        total_loss = 0
        num_batches = 0

        for batch_idx, (images, targets, tokens, target_lengths) in enumerate(train_loader):
            optimizer.zero_grad()

            # 前向传播
            logits, fused_features, node_info = model(images, tokens)

            # 计算输入长度
            input_lengths = torch.tensor([logits.size(1)] * logits.size(0))

            # 解码器损失
            loss = decoder(logits, targets, input_lengths, torch.tensor(target_lengths), node_info)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 10 == 0:
                print(f'Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        avg_loss = total_loss / num_batches
        train_losses.append(avg_loss)

        print(f'Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')

        scheduler.step()

        # 保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'char_to_idx': train_dataset.char_to_idx,
                'idx_to_char': train_dataset.idx_to_char
            }, f'checkpoint_epoch_{epoch + 1}.pth')

            # 绘制损失曲线
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss Curve')
            plt.legend()
            plt.savefig('results/loss_curve.png')
            plt.close()

            # 保存训练记录
            training_info = {
                'epochs': epoch + 1,
                'loss_history': train_losses,
            }

            with open('results/training_info.json', 'w') as f:
                json.dump(training_info, f, indent=2)


if __name__ == '__main__':
    train()