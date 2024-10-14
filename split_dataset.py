import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np


def split_dataset_dirichlet(dataset, num_clients, alpha=0.5, seed=42):
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
    np.random.seed(seed)
    # 获取所有标签
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))
    # 按类别组织数据索引
    class_indices = [np.where(labels == y)[0] for y in range(num_classes)]

    # 初始化每个客户端的数据索引列表
    client_indices = [[] for _ in range(num_clients)]

    # 对每个类别进行分配
    for c, indices in enumerate(class_indices):
        np.random.shuffle(indices)
        # 从狄利克雷分布获取每个客户端的比例
        proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_clients))
        # 计算每个客户端分配到的样本数量
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        split_indices = np.split(indices, proportions)
        for i in range(num_clients):
            client_indices[i].extend(split_indices[i])

    return client_indices


# 示例用法
if __name__ == "__main__":
    # 定义数据转换
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # 下载MNIST训练集
    mnist_train = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    num_clients = 10  # 例如划分为10个客户端
    alpha = 0.5  # 狄利克雷分布参数

    # 划分数据集
    client_indices = split_dataset_dirichlet(mnist_train, num_clients, alpha)

    # 创建每个客户端的DataLoader
    client_loaders = []
    for i in range(num_clients):
        subset = Subset(mnist_train, client_indices[i])
        loader = DataLoader(subset, batch_size=32, shuffle=True)
        client_loaders.append(loader)
        print(f"Client {i+1}: {len(subset)} samples")

    # 可选：检查每个客户端的数据分布
    import matplotlib.pyplot as plt

    for i, loader in enumerate(client_loaders):
        labels = []
        for data, target in loader:
            labels.extend(target.numpy())
        labels = np.array(labels)
        counts = np.bincount(labels, minlength=10)
        plt.bar(range(10), counts, alpha=0.5, label=f"Client {i+1}")
    plt.xlabel("Class")
    plt.ylabel("Number of samples")
    plt.legend()
    plt.show()
