## 模型训练

### 训练流程

本文所提出的交通风险感知场景生成模型采用端到端的训练方式。训练过程中，模型以历史轨迹序列 $\{p_{1:t_{obs}}\}$ 作为输入，通过编码器提取时空特征，经由变分推理网络生成潜在变量 $z$，再由基于Mamba的状态空间解码器生成未来轨迹 $\{p_{t_{obs}+1:t_{pred}}\}$。模型优化目标为最大化观测数据的对数似然下界，即证据下界（Evidence Lower Bound, ELBO）：

$$\mathcal{L}_{ELBO} = \mathbb{E}_{q(z|p_{1:t_{obs}})}[\log p(p_{t_{obs}+1:t_{pred}}|z)] - D_{KL}[q(z|p_{1:t_{obs}})||p(z)]$$

其中，第一项为重构误差项，衡量模型生成轨迹与真实轨迹的一致性；第二项为KL散度正则化项，约束后验分布 $q(z|p_{1:t_{obs}})$ 与先验分布 $p(z)$ 的差异。训练采用Adam优化器，批量大小设置为32，学习率初始化为0.001，并采用余弦退火策略进行动态调整。

### 训练超参数配置

| 超参数 | 数值 |
|--------|------|
| 批量大小（Batch Size） | 128 |
| 学习率（Learning Rate） | 1e-4 |
| 训练轮数（Epochs） | 200 |
| 优化器 | Adam |
| $\beta_1$ | 0.9 |
| $\beta_2$ | 0.999 |
| 权重衰减（Weight Decay） | 1e-4 |
| 观察时间步数（Observation Horizon） | 8 |
| 预测时间步数（Prediction Horizon） | 25 |
| 隐藏层维度（Hidden Dimension） | 512 |
| 潜在变量维度（Latent Dimension） | 32 |
| Mamba状态维度 | 512 |
| 注意力头数（Attention Heads） | 4 |
| Dropout率 | 0.0（未启用） |
| 邻域半径（Neighborhood Radius） | 10000 |