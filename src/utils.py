import os
import json
import random
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast, GPT2TokenizerFast

# Attempt to retrieve the number of GPUs from environment variables, defaulting to 1 if not specified.
NUM_GPUS = int(os.getenv('RWKV_NUM_GPUS', 1))


class CustomDataset(Dataset):
    """
    dataset
    """

    def __init__(self, data, ctx_len, epoch_length_fixed):
        """
        This is a custom PyTorch Dataset class.
        """
        self.ctx_len = ctx_len  # 最大文本长度
        self.epoch_length_fixed = epoch_length_fixed  # 10000   (10000 // batch_size) * batch_size
        self.data = data  # 原数据集
        self.data_type = str(type(self.data))  # 原数据集类型
        self.vocab_size = int(os.getenv('VOCAB_SIZE', 0))  # 预定义词表大小，默认为 0

        if 'MMapIndexedDataset' in self.data_type or 'numpy' in self.data_type:  # 二进制数据 or numpy数据
            self.data_size = len(self.data._bin_buffer) // 2 if 'MMapIndexedDataset' in self.data_type else len(
                self.data)
            print(f'Current vocab size = {self.vocab_size}, data has {self.data_size} tokens.')
        else:  # 文本数据
            unique_chars = sorted(list(set(data)))  # 去重后排序
            self.vocab_size = len(unique_chars)  # 词表长度
            self.stoi = {ch: i for i, ch in enumerate(unique_chars)}  # token to ID
            self.itos = {i: ch for i, ch in enumerate(unique_chars)}  # ID to token
            self.data_size = len(self.data)  # 数据集文本长度
            print(f'Data has {self.data_size} tokens, {self.vocab_size} unique.')
            # Save vocab as json file
            with open('vocab.json', "w", encoding="utf-16") as vocab_file:
                json.dump(self.itos, vocab_file, ensure_ascii=False)  # 以json格式存储词表

    def __len__(self):
        return self.epoch_length_fixed // NUM_GPUS

    def __getitem__(self, _):
        """
        Returns a random sequence from the dataset.
        随机从数据集中取一段长度为1024的句子
        """
        start_idx = np.random.randint(0, self.data_size - (self.ctx_len + 1))  # 随机取一个开始id
        if 'MMapIndexedDataset' in self.data_type:
            sequence = self.data.get(idx=0, offset=start_idx, length=self.ctx_len + 1).astype(int)
        elif 'numpy' in self.data_type:
            sequence = self.data[start_idx:start_idx + self.ctx_len + 1]
        else:  # 取长度为1025的句子，然后转为token id
            sequence = [self.stoi[s] for s in self.data[start_idx:start_idx + self.ctx_len + 1]]
        x = torch.tensor(sequence[:-1], dtype=torch.long)  # input id
        y = torch.tensor(sequence[1:], dtype=torch.long)  # output id
        return x, y


class Tokenizer():
    def __init__(self, word_name, unknown_char='\ue083'):
        """
        Initializes the Tokenizer class.
        """
        self.char_mode = isinstance(word_name, str)
        if self.char_mode:
            with open(f'{word_name}.json', "r", encoding="utf-16") as result_file:
                self.word_table = json.load(result_file)
            self.vocab_size = len(self.word_table)
            self.stoi = {v: int(k) for k, v in self.word_table.items()}
            self.itos = {int(k): v for k, v in self.word_table.items()}
            self.unknown_char = self.stoi[unknown_char]
        else:
            if word_name[0] == word_name[1]:
                self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=word_name[0])
            else:
                self.tokenizer = GPT2TokenizerFast(word_name[0], word_name[1])
            self.vocab_size = len(self.tokenizer)

    def refine_context(self, context):
        """
        Refines the context by stripping and splitting by newline characters.
        """
        return '\n' + '\n'.join(c.strip().strip('\u3000').strip('\r') for c in context.strip().split('\n') if c)

    def sample_logits(self, out, x, ctx_len, temperature=1.0, top_p_usual=None, top_p_newline=None):
        """
        Samples from the given logits using temperature scaling and top-p probability.
        """
        probs = F.softmax(torch.tensor(out), dim=-1)
        top_p = top_p_newline if self.char_mode and self.itos[int(x[-1])] == '\n' else top_p_usual
        sorted_probs, s_index = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1).numpy()
        cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
        probs[probs < cutoff] = 0
        if temperature != 1.0:
            probs = probs.pow(1.0 / temperature)
        return torch.multinomial(probs, num_samples=1)[0]


def to_float(x):
    """
    Converts a PyTorch tensor to a float.
    """
    return x.cpu().detach().numpy().flatten()[0].astype(float)


def set_seed(seed):
    """
    Sets the seed for randomness for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
