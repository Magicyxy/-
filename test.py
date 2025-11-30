import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import os
import json

# 导入模型 - 使用直接导入方式
import sys
import os

sys.path.append('./models')
sys.path.append('./dataset')

from models.fusion_encoder import ChemicalAwareTwoStreamModel
from models.ctc_crf_decoder import ChemicalAwareCRFDecoder
from dataset.chem_dataset import ChemicalEquationDataset


def test():
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 测试数据集
    test_dataset = ChemicalEquationDataset(
        image_dir='dataset/images/',
        label_dir='dataset/labels/',
        classes_file='dataset/classes.txt',
        transform=transform
    )

    # 使用准确的类别数量
    num_chars = len(test_dataset.classes)
    vocab_size = num_chars

    print(f"Number of character classes: {num_chars}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=test_dataset.collate_fn
    )

    # 加载模型 - 使用准确的类别数量
    model = ChemicalAwareTwoStreamModel(num_chars, vocab_size)
    decoder = ChemicalAwareCRFDecoder(num_chars)

    # 加载最新检查点
    checkpoint_files = [f for f in os.listdir('.') if f.startswith('checkpoint_epoch_')]
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        decoder.set_char_mappings(checkpoint['char_to_idx'], checkpoint['idx_to_char'])
        print(f"Loaded checkpoint: {latest_checkpoint}")
    else:
        print("No checkpoint found. Please train the model first.")
        return

    model.eval()

    # 测试
    correct_predictions = 0
    total_predictions = 0
    char_correct = 0
    char_total = 0

    predictions_results = []
    sample_images = []

    with torch.no_grad():
        for i, (images, targets, tokens, target_lengths) in enumerate(test_loader):
            if total_predictions >= 20:  # 只测试20个样本
                break

            logits, _, node_info = model(images, tokens)

            # 解码预测
            predictions = decoder.decode(logits)

            # 计算准确率
            target_sequence = targets[0].tolist()
            target_sequence = [x for x in target_sequence if x != 0]  # 移除padding

            predicted_sequence = predictions[0]

            # 序列准确率
            if target_sequence == predicted_sequence:
                correct_predictions += 1

            # 字符准确率
            min_len = min(len(target_sequence), len(predicted_sequence))
            for j in range(min_len):
                if target_sequence[j] == predicted_sequence[j]:
                    char_correct += 1
                char_total += 1

            total_predictions += 1

            # 保存预测结果和图像用于可视化
            target_str = ''.join([test_dataset.idx_to_char.get(idx, '?') for idx in target_sequence])
            predicted_str = ''.join([test_dataset.idx_to_char.get(idx, '?') for idx in predicted_sequence])

            predictions_results.append({
                'target': target_str,
                'predicted': predicted_str,
                'correct': target_str == predicted_str
            })

            # 保存图像用于可视化
            if len(sample_images) < 5:  # 保存5个样本用于可视化
                image_tensor = images[0]
                image_np = (image_tensor.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255
                image_np = image_np.astype(np.uint8)
                sample_images.append((image_np, target_str, predicted_str))

            print(
                f"Sample {total_predictions}: Target: {target_str}, Predicted: {predicted_str}, Correct: {target_str == predicted_str}")

    # 输出结果
    sequence_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    char_accuracy = char_correct / char_total if char_total > 0 else 0

    print(f'\nFinal Results:')
    print(f'Sequence Accuracy: {sequence_accuracy:.4f}')
    print(f'Character Accuracy: {char_accuracy:.4f}')
    print(f'Total Test Samples: {total_predictions}')

    # 保存测试结果
    results = {
        'sequence_accuracy': sequence_accuracy,
        'character_accuracy': char_accuracy,
        'total_samples': total_predictions,
        'predictions': predictions_results
    }

    with open('results/test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 绘制准确率图表
    plt.figure(figsize=(10, 5))
    categories = ['Sequence Accuracy', 'Character Accuracy']
    values = [sequence_accuracy, char_accuracy]

    plt.bar(categories, values, color=['blue', 'green'])
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title('Model Test Performance')

    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')

    plt.savefig('results/accuracy.png')
    plt.close()

    # 生成预测样本可视化
    fig, axes = plt.subplots(len(sample_images), 1, figsize=(12, 3 * len(sample_images)))
    if len(sample_images) == 1:
        axes = [axes]

    for i, (image_np, target_str, predicted_str) in enumerate(sample_images):
        axes[i].imshow(image_np[:, :, 0], cmap='gray')
        axes[i].set_title(f'Target: {target_str}\nPredicted: {predicted_str}',
                          color='green' if target_str == predicted_str else 'red')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('results/prediction_samples.png')
    plt.close()

    print(f"\nSample Predictions saved to results/prediction_samples.png")


if __name__ == '__main__':
    test()