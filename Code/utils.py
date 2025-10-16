import os
import torch
import random
import numpy as np 

def ADE_FDE(y_, y, batch_first=False):
    # average displacement error
    # final displacement error
    # y_, y: S x L x N x 2
    #print("y_shape: ",y_.cpu().numpy().shape,"y shape: ",y.cpu().numpy().shape)
    if torch.is_tensor(y):
        err = (y_ - y).norm(dim=-1) #Err=||y_*y||dim=-1
        #print (err.cpu().numpy().shape)
    else:
        err = np.linalg.norm(np.subtract(y_, y), axis=-1)
        #print (err.shape)
    if len(err.shape) == 1:
        fde = err[-1]
        ade = err.mean()
    elif batch_first:
        fde = err[..., -1]
        ade = err.mean(-1)
    else:
        fde = err[..., -1, :]
        ade = err.mean(-2)
    return ade, fde

def kmeans(k, data, iters=None):
    centroids = data.copy()
    np.random.shuffle(centroids)
    centroids = centroids[:k]# Select the first k centroids

    if iters is None: iters = 100000
    for _ in range(iters):
    # while True:
        # Calculate Euclidean distances between each data point and each centroid
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        # Assign each data point to the nearest centroid
        closest = np.argmin(distances, axis=0)
        centroids_ = []
        for k in range(len(centroids)):
            cand = data[closest==k]
            if len(cand) > 0:
                centroids_.append(cand.mean(axis=0))
            else:
                # If a centroid has no assigned data points, 
                # randomly choose a data point as the new centroid
                centroids_.append(data[np.random.randint(len(data))])
        centroids_ = np.array(centroids_)
        # Check for convergence by measuring the change in centroids
        if np.linalg.norm(centroids_ - centroids) < 0.0001:
            break
        # Update centroids for the next iteration
        centroids = centroids_
    return centroids

def FPC(y, n_samples):
    """
    Final Position Clustering (FPC): 最终位置聚类算法
    从一组轨迹中选择目标点的简单算法，用于提高轨迹预测的多样性
    
    Args:
        y: 预测轨迹 shape: S x L x 2 (S: 样本数, L: 序列长度, 2: x,y坐标)
        n_samples: 要选择的聚类中心数量
    
    Returns:
        chosen: 被选中的轨迹索引数组 shape: (n_samples,)
    """
    # 处理输入维度
    if torch.is_tensor(y):
        y = y.cpu().detach().numpy()
    
    # 如果有额外维度，去除单维度
    if y.ndim > 3:
        y = np.squeeze(y)
    
    # 提取最终位置作为目标点
    goal = y[...,-1,:2]  # shape: (S, 2)
    
    # 应用K-means聚类算法获取聚类中心
    goal_ = kmeans(n_samples, goal)  # shape: (n_samples, 2)
    
    # 计算每个轨迹终点到聚类中心的距离
    dist = np.linalg.norm(goal_[:, np.newaxis, :2] - goal[np.newaxis, :, :2], axis=-1)  # shape: (n_samples, S)
    
    # 为每个聚类中心选择最近的轨迹
    chosen = np.argmin(dist, axis=1)  # shape: (n_samples,)
    
    return chosen

    
def seed(seed: int):
    rand = seed is None
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder="big")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = not rand
    torch.backends.cudnn.benchmark = rand

def get_rng_state(device):
    return (
        torch.get_rng_state(), 
        torch.cuda.get_rng_state(device) if torch.cuda.is_available and "cuda" in str(device) else None,
        np.random.get_state(),
        random.getstate(),
        )

def set_rng_state(state, device):
    torch.set_rng_state(state[0])
    if state[1] is not None: torch.cuda.set_rng_state(state[1], device)
    np.random.set_state(state[2])
    random.setstate(state[3])
