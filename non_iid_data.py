import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from collections import defaultdict
import random
import math
from data_preprocess import data_set


# 6. 创建客户端的子数据集
def create_client_subsets(dataset, client_data):
    client_subsets = {}
    for client_id, data in client_data.items():
        client_subsets[client_id] = {
            "label": data["label"],
            "subset": Subset(dataset, data["indices"]),
        }
    return client_subsets


# 5. 分配客户端
def assign_clients(balanced_class_indices, num_clients, seed=42):
    random.seed(seed)
    clients = {i: {"label": None, "indices": []} for i in range(num_clients)}
    labels = list(balanced_class_indices.keys())
    num_classes = len(labels)

    # 计算每个类别需要分配给多少个客户端
    clients_per_class = math.ceil(num_clients / num_classes)

    # 分配客户端到类别
    label_client_map = defaultdict(list)
    client_id = 0
    for label in labels:
        for _ in range(clients_per_class):
            if client_id < num_clients:
                label_client_map[label].append(client_id)
                client_id += 1
            else:
                break

    # 分配数据到客户端
    client_data = {}
    for label, client_ids in label_client_map.items():
        indices = balanced_class_indices[label]
        num_clients_for_label = len(client_ids)
        if num_clients_for_label == 0:
            continue
        samples_per_client = len(indices) // num_clients_for_label
        for i, client_id in enumerate(client_ids):
            start_idx = i * samples_per_client
            end_idx = start_idx + samples_per_client
            # 对最后一个客户端分配剩余的样本
            if i == num_clients_for_label - 1:
                end_idx = len(indices)
            client_data[client_id] = {
                "label": label,
                "indices": indices[start_idx:end_idx],
            }
    return client_data


# 3. 按类别划分数据集
def split_dataset_by_class(dataset):
    class_indices = defaultdict(list)
    for idx, (data, label) in enumerate(dataset):
        class_indices[label].append(idx)
    return class_indices


def balance_class_subsets(class_indices, target_count, seed=42):
    balanced_indices = {}
    random.seed(seed)
    for label, indices in class_indices.items():
        if len(indices) > target_count:
            balanced_indices[label] = random.sample(indices, target_count)
        else:
            balanced_indices[label] = indices
    return balanced_indices


def data_init_non_iid(FL_params):
    kwargs = {"num_workers": 0, "pin_memory": True} if FL_params.cuda_state else {}
    train_dataset, testset = data_set(FL_params.data_name)
    test_loader = DataLoader(
        testset, batch_size=FL_params.test_batch_size, shuffle=True, **kwargs
    )
    train_class_indices = split_dataset_by_class(train_dataset)

    # 4. 确定目标样本数量并平衡数据
    class_counts = {
        label: len(indices) for label, indices in train_class_indices.items()
    }
    print("每个类别的样本数量:", class_counts)
    target_count = min(class_counts.values())
    print(f"目标样本数量: {target_count}")
    balanced_train_class_indices = balance_class_subsets(
        train_class_indices, target_count
    )

    # 示例：假设有 15 个客户端
    # num_clients = 100
    client_data = assign_clients(balanced_train_class_indices, FL_params.N_total_client)

    client_subsets = create_client_subsets(train_dataset, client_data)

    # 7. 验证分配结果
    for client_id, data in client_data.items():
        label = data["label"]
        num_samples = len(data["indices"])
        print(f"客户端 {client_id}: 类别 {label}, 样本数量 {num_samples}")
    client_loaders = []
    for client_id, data in client_subsets.items():
        client_loaders.append(
            DataLoader(
                data["subset"],
                batch_size=FL_params.local_batch_size,
                shuffle=True,
                **kwargs,
            )
        )
    return client_loaders, test_loader


def data_init_dirichlet(FL_params, alpha=0.5):
    """
    将数据集按照狄利克雷分布划分为多个客户端。

    参数：
    - dataset: PyTorch数据集（例如MNIST）
    - num_clients: 客户端数量
    - alpha: 狄利克雷分布的参数，控制分布的均匀性。较小的值会导致更不均匀的分布。
    - seed: 随机种子

    返回：
    - client_indices: 列表，每个元素是对应客户端的数据索引
    """
    kwargs = {"num_workers": 0, "pin_memory": True} if FL_params.cuda_state else {}
    # INFO: Download dataset
    trainset, testset = data_set(FL_params.data_name)
    test_loader = DataLoader(
        testset, batch_size=FL_params.test_batch_size, shuffle=True, **kwargs
    )
    # 获取所有标签
    labels = np.array(trainset.targets)
    num_classes = len(np.unique(labels))
    # 按类别组织数据索引
    class_indices = [np.where(labels == y)[0] for y in range(num_classes)]

    # 初始化每个客户端的数据索引列表
    client_indices = [[] for _ in range(FL_params.N_total_client)]

    # 对每个类别进行分配
    for c, indices in enumerate(class_indices):
        np.random.shuffle(indices)
        # 从狄利克雷分布获取每个客户端的比例
        proportions = np.random.dirichlet(
            alpha=np.repeat(alpha, FL_params.N_total_client)
        )
        # 计算每个客户端分配到的样本数量
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        split_indices = np.split(indices, proportions)
        for i in range(FL_params.N_total_client):
            client_indices[i].extend(split_indices[i])
    # 创建每个客户端的DataLoader
    client_loaders = []
    for i in range(FL_params.N_total_client):
        subset = Subset(trainset, client_indices[i])
        loader = DataLoader(
            subset,
            batch_size=FL_params.local_batch_size,
            shuffle=True,
        )
        client_loaders.append(loader)
    return client_loaders, test_loader
