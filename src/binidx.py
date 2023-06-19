# 引入所需的库
import os
import torch
import numpy as np
import struct
from functools import lru_cache
from itertools import accumulate

# 定义一个全局打印函数，只在主节点打印消息
def print_rank_0(*message):
    """如果分布式已经初始化，那么只在主节点上打印。"""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(*message, flush=True)
    else:
        print(*message, flush=True)

# 预热内存映射文件，这里的代码已被注释掉
def _warmup_mmap_file(path):
    pass

# 定义一个数据类型字典，键是整数编码，值是相应的numpy数据类型
dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: float,
    7: np.double,
    8: np.uint16,
}

# 通过数据类型找到对应的编码
def code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)

# 定义索引文件和数据文件的路径格式
def index_file_path(prefix_path):
    return prefix_path + ".idx"

def data_file_path(prefix_path):
    return prefix_path + ".bin"

# 主要的内存映射数据集类，继承自torch.utils.data.Dataset
class MMapIndexedDataset(torch.utils.data.Dataset):
    # 内部的索引类
    class Index(object):
        _HDR_MAGIC = b"MMIDIDX\x00\x00"

        # 初始化函数，读取索引文件，并将其内容存储为numpy数组
        def __init__(self, path, skip_warmup=False):
            with open(path, "rb") as stream:
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, (
                    "Index file doesn't match expected format. "
                    "Make sure that --dataset-impl is configured properly."
                )
                version = struct.unpack("<Q", stream.read(8))
                assert (1,) == version
                (dtype_code,) = struct.unpack("<B", stream.read(1))
                self._dtype = dtypes[dtype_code]
                self._dtype_size = self._dtype().itemsize
                self._len = struct.unpack("<Q", stream.read(8))[0]
                self._doc_count = struct.unpack("<Q", stream.read(8))[0]
                offset = stream.tell()

            if not skip_warmup:
                print_rank_0("    warming up index mmap file...")
                _warmup_mmap_file(path)

            # 用numpy.memmap加载索引文件，并创建一个内存视图
            self._bin_buffer_mmap = np.memmap(path, mode="r", order="C")
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            print_rank_0("    reading sizes...")
            self._sizes = np.frombuffer(
                self._bin_buffer, dtype=np.int32, count=self._len, offset=offset
            )
            print_rank_0("    reading pointers...")
            self._pointers = np.frombuffer(
                self._bin_buffer,
                dtype=np.int64,
                count=self._len,
                offset=offset + self._sizes.nbytes,
            )
            print_rank_0("    reading document index...")
            self._doc_idx = np.frombuffer(
                self._bin_buffer,
                dtype=np.int64,
                count=self._doc_count,
                offset=offset + self._sizes.nbytes + self._pointers.nbytes,
            )

        def __del__(self):
            # 删除对象时，关闭内存映射并删除
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        # 其他一些属性和方法，获取数据集的信息和条目

    def __init__(self, path, skip_warmup=False):
        super().__init__()

        self._path = None
        self._index = None
        self._bin_buffer = None

        self._do_init(path, skip_warmup)

    def __getstate__(self):
        # 获取状态，用于序列化
        return self._path

    def __setstate__(self, state):
        # 设置状态，用于反序列化
        self._do_init(state)

    def _do_init(self, path, skip_warmup):
        # 初始化数据集
        self._path = path
        self._index = self.Index(index_file_path(self._path), skip_warmup)

        if not skip_warmup:
            print_rank_0("    warming up data mmap file...")
            _warmup_mmap_file(data_file_path(self._path))
        print_rank_0("    creating numpy buffer of mmap...")
        self._bin_buffer_mmap = np.memmap(
            data_file_path(self._path), mode="r", order="C"
        )
        print_rank_0("    creating memory view of numpy buffer...")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    # 其他一些属性和方法，包括获取条目、获取大小等
    def __getitem__(self, idx):
        if isinstance(idx, int):
            ptr, size = self._index[idx]
            np_array = np.frombuffer(
                self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr
            )
            return np_array
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            ptr = self._index._pointers[start]
            sizes = self._index._sizes[idx]
            offsets = list(accumulate(sizes))
            total_size = sum(sizes)
            np_array = np.frombuffer(
                self._bin_buffer, dtype=self._index.dtype, count=total_size, offset=ptr
            )
            sents = np.split(np_array, offsets[:-1])
            return sents

    def get(self, idx, offset=0, length=None):
        ptr, size = self._index[idx]
        if length is None:
            length = size - offset
        ptr += offset * np.dtype(self._index.dtype).itemsize
        np_array = np.frombuffer(
            self._bin_buffer, dtype=self._index.dtype, count=length, offset=ptr
        )
        return np_array

    @property
    def sizes(self):
        return self._index.sizes

    @property
    def doc_idx(self):
        return self._index.doc_idx

    def get_doc_idx(self):
        return self._index._doc_idx

    def set_doc_idx(self, doc_idx_):
        self._index._doc_idx = doc_idx_

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        return os.path.exists(index_file_path(path)) and os.path.exists(
            data_file_path(path)
        )
