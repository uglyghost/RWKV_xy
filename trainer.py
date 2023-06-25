import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import config
from model import RGM
import os
from tqdm import tqdm

# 设定随机种子
# torch.manual_seed(42)



class CustomDataset(Dataset):
    """
    dataset
    """

    def __init__(self, data, ctx_len):
        """
        This is a custom PyTorch Dataset class.
        """
        self.ctx_len = ctx_len  # 最大文本长度
        self.data = data  # 原数据集
        self.data_type = str(type(self.data))  # 原数据集类型

        unique_chars = sorted(list(set(data)))  # 去重后排序
        self.vocab_size = len(unique_chars)  # 词表长度
        self.stoi = {ch: i for i, ch in enumerate(unique_chars)}  # token to ID
        self.itos = {i: ch for i, ch in enumerate(unique_chars)}  # ID to token
        self.data_size = len(self.data)  # 数据集文本长度
        print(f'Data has {self.data_size} tokens, {self.vocab_size} unique.')
        # Save vocab as json file
        with open('vocab.json', "w", encoding="utf-16") as vocab_file:
            json.dump(self.itos, vocab_file, ensure_ascii=False)  # 以json格式存储词表

    def __getitem__(self, _):
        """
        Returns a random sequence from the dataset.
        随机从数据集中取一段长度为1024的句子
        """

        start_idx = np.random.randint(0, self.data_size - (self.ctx_len + 1))  # 随机取一个开始id
        sequence = [self.stoi[s] for s in self.data[start_idx:start_idx + self.ctx_len + 1]]
        x = torch.tensor(sequence[:-1], dtype=torch.long)  # input id
        y = torch.tensor(sequence[1:], dtype=torch.long)  # output id
        return x, y

    def __len__(self):
        return 10000


if __name__ == '__main__':
    # dataset
    train_dataset = CustomDataset(open(config.datafile, "r", encoding="utf-8").read(), config.ctx_len)
    train_dataloader = DataLoader(train_dataset, shuffle=False, pin_memory=True, batch_size=config.batch_size)
    vocab_size = train_dataset.vocab_size  # vacab size

    # 模型
    model = RGM(vocab_size)
    model.to(device=config.device)

    optimizer = model.configure_optimizers()

    # 训练模型
    for epoch in range(config.epoch):

        for x, y in tqdm(train_dataloader, total=len(train_dataloader)):
            loss = model(x, y)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 打印每个epoch的损失
            print(f"Epoch [{epoch}], Loss: {loss.item():.4f}")

    # # 在测试集上评估模型
    # x_test = torch.tensor([...])  # 测试数据
    # y_test = torch.tensor([...])  # 测试目标数据
    #
    # with torch.no_grad():
    #     outputs = model(x_test)
    #     _, predicted = torch.max(outputs.data, 1)
    #     accuracy = torch.sum(predicted == y_test).item() / len(y_test)
    #     print(f"Test Accuracy: {accuracy:.4f}")
    #
    # # 进行预测
    # x_new = torch.tensor([...])  # 需要预测的新数据
    # with torch.no_grad():
    #     outputs = model(x_new)
    #     _, predicted = torch.max(outputs.data, 1)
    #     print("Predictions:", predicted)
