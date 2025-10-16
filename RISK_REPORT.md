# 风险驱动轨迹生成改进报告 (RISK_REPORT)

> 本文档以学术化行文总结近期对 SocialVAE 框架在高风险/事故倾向交通轨迹生成方向上的体系化扩展，涵盖：目标函数重构、风险度量模块化、权重/不确定性自适应机制、日志与分析工具链。每个小节末尾附有“通俗解释”用于工程/非算法读者快速理解。

## 1. 总览

本阶段工作聚焦“风险显式建模 + 可控生成”二元目标，新增与改进内容概括如下：
1. 多风险组件化聚合：引入最小距离 (MinDist)、Time-to-Collision (TTC)、Post Encroachment Time (PET)、几何 Overlap 四类结构化风险度量；形成可裁剪集合 \(\mathcal{C}\)。
2. Autoscale 全局自适应：在线调节全局缩放 \(g_{risk}\)，使风险项对基准损失的相对能量收敛至目标占比 \(\gamma\)。
3. 双轨加权机制： (a) 可学习概率式组件权重（含熵正则约束分布均衡）；(b) log-sigma 不确定性加权（Bayesian 式尺度估计 + 罚项）。
4. 组件归一化策略：缓解不同物理维度/统计尺度差异导致的主导偏移，提升多指标协同训练稳定性。
5. 不确定性正则：log-sigma 罚项抑制“虚假放大噪声”行为，维持估计可辨识性。
6. 运动学（Kinematic）平滑正则：约束高阶变化（速度/加速度）以提升轨迹物理可行性与视觉平滑度。
7. 高风险样本挖掘与持久化：按分位阈值（如 90%）筛选，并保存预测张量用于事后事故态势分析与可视化。
8. 运行期控制接口：`--no-resume` 防止意外加载已完成 checkpoint；`--fpc-range` 动态限定评估搜索空间以缩短迭代周期。
9. 全链路日志增强：TensorBoard 指标、结构化 autoscale CSV、风险样本 .pt、演化绘图脚本（风险/不确定性时间序列）。
10. 快速验证配置：`Interaction_quick.py` 低 epoch / 低 batch 回归，加速功能正确性确认。

**通俗解释：** 这一节是在列清单：我们给模型加了多种风险指标，把它们的影响占比自动调稳；可以让模型自己学“哪个风险更重要”或用不确定性权重；保证不同指标不乱抢话语权；还能把高风险案例抓出来单独存。外加命令行、日志、快速测试脚本，让实验和调参更高效。

## 2. 符号约定

| 符号 | 含义 |
|------|------|
| $X$ | 历史观测轨迹 $(T_h,N,d)$ |
| $Y$ | 未来真实轨迹 $(T_f,N,d)$ |
| $\hat{Y}^{(s)}$ | 第 $s$ 个采样预测 $(s=1..S)$ |
| $z$ | VAE 潜变量 |
| $L_{rec}$ | 重建误差 (或 MSE) |
| $L_{wmse}$ | 加权 MSE / 轨迹点加权重建 |
| $L_{KL}$ | KL 散度 $\mathrm{KL}(q_\phi(z|X,Y)\|p(z))$ |
| $L_{adv}$ | 对抗 / 判别损失 |
| $L_{kin}$ | 轨迹平滑/运动学正则 |
| $R_i$ | 第 $i$ 个风险组件标量 |
| $R_{score}$ | 聚合后的风险分数 |
| $w_{risk}$ | 外部显式风险权重 (config) |
| $g_{risk}$ | Autoscale 学习的全局缩放 |
| $\gamma$ | 目标风险占比 (target frac) |
| $\lambda_{ent}$ | 组件权重熵正则系数 |
| $\lambda_{\sigma}$ | log-sigma 正则系数 |

**通俗解释：** 这张表就是变量对照表：看不懂公式里的字母，可以来这里查。比如 \(g_{risk}\) 就是风险放大的旋钮，\(\gamma\) 是希望风险占总损失的大概比例。

## 3. 基础损失结构

不含风险时：
$$
L_{base} = w_{rec} L_{rec} + w_{wmse} L_{wmse} + w_{KL} L_{KL} + w_{adv} L_{adv} + w_{kin} L_{kin}
$$

加入风险（负号表示“鼓励”更高风险分数：生成更具事故倾向场景）及正则后：
$$
\begin{aligned}
L_{total} = &\; L_{base} - w_{risk} (g_{risk} R_{score}) \\
& - \lambda_{ent} H(w) + \lambda_{\sigma} \sum_i \log \sigma_i
\end{aligned}
$$
其中熵正则与 log-sigma 罚项对应两条互斥路径：
- 组件权重模式：使用 \(w_i\) + 熵正则 \(H(w)\)；
- 不确定性模式：使用 \(\log\sigma_i\) + 罚项 \(\sum_i \log \sigma_i\)，不再出现权重熵。

**通俗解释：** 基础损失就是重建 + KL + 对抗 + 平滑等常规项。我们再“减去”风险分数（相当于鼓励模型生成更冒险的轨迹）。两种附加正则不会一起启用：要么学一组概率权重并让它们别太极端（熵），要么学每个指标的“噪声尺度”并罚它太大（log-sigma）。

### 3.1 基础损失各项公式细化与解释

下面给出当前实现中各基础分量的明确公式与工程实现要点，对应代码主要位于 `social_vae.py`：

#### 3.1.1 重建误差 / 基础 MSE (Rec)
设预测单样本（首样本）轨迹为 \(\hat{Y} \in \mathbb{R}^{T_f\times N\times 2}\)，真实轨迹为 \(Y\)。代码中的 `err` 是逐点平方残差：
$$
err_{t,n} = \lVert Y_{t,n} - \hat{Y}_{t,n} \rVert_2^2
$$
重建项取全局平均：
$$
L_{rec} = \frac{1}{T_f N} \sum_{t=1}^{T_f}\sum_{n=1}^N err_{t,n}
$$
（实现：`rec = err.mean()`）。

#### 3.1.2 时间加权 MSE (Weighted MSE, WMSE)
对将来更远时间步给予指数衰减（或加权强调近端）：
$$
w_t = \frac{e^{-\alpha (t-1)}}{\sum_{k=1}^{T_f} e^{-\alpha (k-1)}}, \quad t=1..T_f
$$
$$
L_{wmse} = \sum_{t=1}^{T_f} w_t\Big( \frac{1}{N} \sum_{n=1}^N err_{t,n} \Big)
$$
实现中先构造未归一化 `weights = exp(-α·t)`，再做加权平均（`weighted_mse_loss`）。\(\alpha\) 对应 `loss_rec_wmse_alpha`（默认 0.3 与组合模式互补）。

#### 3.1.3 KL 散度 (VAE 先验匹配)
编码器输出每步（或整段聚合）潜变量后验 \(q_\phi(z|X,Y)=\mathcal{N}(\mu, \operatorname{diag}(\sigma^2))\)，先验 \(p(z)=\mathcal{N}(0,I)\)：
$$
L_{KL} = \frac{1}{T_f N} \sum_{t,n} \frac{1}{2} \sum_{d=1}^{D_z} \Big( \mu_{t,n,d}^2 + \sigma_{t,n,d}^2 - \log \sigma_{t,n,d}^2 - 1 \Big)
$$
实现里聚合后直接 `kl.mean()` 做为 \(\bar{L}_{KL}\)。

#### 3.1.4 对抗 / 近邻吸引式损失 (L_adv)
当前实现（尚非典型 GAN 判别器形式，而是“邻居距离的 softmax 期望”）定义：
1. 预测绝对位置 \(\hat{Y}_{t,i}\) 与第 \(j\) 个邻居位置 \(N_{t,i,j}\) 之间距离：
$$ d_{t,i,j} = \lVert \hat{Y}_{t,i} - N_{t,i,j}\rVert_2 $$
2. 打分： \( s_{t,i,j} = -\sqrt{d_{t,i,j} + \varepsilon} \)
3. 邻居 softmax 权重： \( p_{t,i,j} = \frac{e^{s_{t,i,j}}}{\sum_k e^{s_{t,i,k}}} \)
4. 实现的损失：
$$
L_{adv} = \sum_{t,i}\sum_j e^{s_{t,i,j}} p_{t,i,j} = \sum_{t,i} \frac{\sum_j e^{2 s_{t,i,j}}}{\sum_k e^{s_{t,i,k}}}
$$

直觉：当预测接近多个邻居时，\(s_{t,i,j}\) 较大（负号但距离小），\(e^{2s}\) 放大近邻贡献，鼓励生成“贴近/互动”场景。它并不区分真实/虚假轨迹（非典型判别器），更像一种“邻居接近度强化”正项。若需要真正 GAN 判别器，可替换为判别器交叉熵：
$$
L_{adv}^{GAN} = \mathbb{E}_{(X,Y)}[\log D(X,Y)] + \mathbb{E}_{(X,\hat{Y})}[\log (1-D(X,\hat{Y}))]
$$
并在生成侧最小化 \(-\mathbb{E}_{(X,\hat{Y})}\log D(X,\hat{Y})\)。

#### 3.1.5 运动学平滑损失 (L_kin)
速度： \( v_{t,n} = \hat{Y}_{t,n} - \hat{Y}_{t-1,n} \)；加速度： \( a_{t,n} = v_{t,n} - v_{t-1,n} \)。
$$
L_{acc} = \frac{1}{(T_f-2)N}\sum_{t=2}^{T_f-1}\sum_n \lVert a_{t,n}\rVert_2
$$
角度： \( \theta_{t,n} = \operatorname{atan2}(v^y_{t,n}, v^x_{t,n}) \)；
包络后的角速度变化： \( \Delta \theta_{t,n} = \operatorname{wrap}(\theta_{t,n}-\theta_{t-1,n}) \in (-\pi,\pi] \)
$$
L_{ang} = \frac{1}{(T_f-2)N}\sum_{t=2}^{T_f-1}\sum_n |\Delta \theta_{t,n}|
$$
综合：
$$
L_{kin} = L_{acc} + L_{ang}
$$
实现中两部分直接求和（`acceleration_loss + angular_loss`）。

#### 3.1.6 合成基础损失 (无风险项情形)
若不合并 rec & wmse：
$$
L_{base} = w_{rec} L_{rec} + w_{wmse} L_{wmse} + w_{KL} L_{KL} + w_{adv} L_{adv} + w_{kin} L_{kin}
$$
若设置 `loss_combine_rec_wmse=True`（合成重建）：
$$
L_{rec}^{comb} = (1-\alpha)L_{rec} + \alpha L_{wmse},\quad
L_{base} = w_{rec} L_{rec}^{comb} + w_{KL} L_{KL} + w_{adv} L_{adv} + w_{kin} L_{kin}
$$
#### 3.1.7 总损失再含风险与正则
与上文 3 节公式对应：
$$
L_{total} = L_{base} - w_{risk} g_{risk} R_{score} - \lambda_{ent} H(w) + \lambda_{\sigma}\sum_i \log \sigma_i
$$
（两种聚合模式互斥：仅其一附加）。

#### 3.1.8 说明与潜在改进
- 目前的 `L_adv` 是“距离软聚合”形式，不依赖二分类判别器；若需更标准对抗性多模态增强，可并行引入判别器网络 D，替换上述项。
- 运动学项可扩展为二范数平方或 jerk（三阶差分）以强化高阶平滑。
- Weighted MSE 的衰减方向可反转（强调远期）——改为 `exp(+α t)` 并重新归一化。
- KL 可做 cyclical annealing 或 beta-schedule 改善表示学习（引入 `β(t)`）。

**通俗解释：** 这一小节把每个“名字”具体变成了可复现的数学式子：怎么加权、怎么平均、对抗项真正算的是什么、平滑项包括哪两个部分。也指出了如果要做成真正 GAN 可以怎么改，给以后优化留接口。


## 4. 风险组件定义

### 4.1 最小距离 (MinDist)
对主体-邻居未来相对距离 $D_{t,b,n}$：
1. 剪裁：$D \leftarrow \max(D, \epsilon)$  
2. soft-min：
$$
\text{softmin}_{t,b} = -\frac{1}{\beta} \log \sum_{n} e^{-\beta D_{t,b,n}}
$$
3. 映射为风险：$r_{t,b} = e^{-\text{softmin}_{t,b}}$  
4. 聚合：$ R_{\text{min\_dist}} = \frac{1}{TB} \sum_{t,b} r_{t,b}$

**通俗解释：** 看所有参与者之间未来的最小距离，越近越危险。用 soft-min 平滑，避免只盯死一个尖锐最小值导致不稳定，再转成“风险分数”平均一下。

### 4.2 TTC (Time-to-Collision)
内部函数估算相对时距 → 风险张量 $r^{\text{TTC}}_{t,b,n}$，再：
$$
R_{\text{TTC}} = \text{mean}_{t,b} \big[ \max_n r^{\text{TTC}}_{t,b,n} \big]
$$

**通俗解释：** 估计如果继续现在趋势会多久“相撞”。越可能短时间相遇就越危险，因此取各邻居里最危险一个，再时间/批次平均。

### 4.3 PET (Post Encroachment Time)
参数：$pet\_dist\_th, pet\_alpha, pet\_beta, pet\_gamma, pet\_continuous, pet\_time\_temp$。基于距离门控 + 时间衰减：
$$
R_{\text{PET}} = \mathbb{E}_{t,b} \Big[ \sum_n \alpha_{t,b,n} r^{PET}_{t,b,n} \Big], \quad \alpha_{t,b,n} = \text{softmax}_n(\gamma\, r^{PET}_{t,b,n})~(\text{可选})
$$

**通俗解释：** PET 衡量“两个主体先后占用潜在冲突区的时间差”。差越小表示擦肩或干涉越严重。我们用一些 Sigmoid/Softmax 做平滑和加权，得出整体威胁程度。

### 4.4 Overlap 风险
对 OBB/几何重叠估计 $r^{ov}_{t,b,n}$：
$$
R_{\text{overlap}} = \text{mean}_{t,b} \big[ \max_n r^{ov}_{t,b,n} \big]
$$

**通俗解释：** 直接看几何形状（或包围盒）是否“穿”到一起/重叠。重叠越明显风险越大，取最危险邻居再平均。

组件集合：$\mathcal{C} = \{R_{min}, R_{TTC}, R_{PET}, R_{overlap}\}$（可按开关裁剪）。

**通俗解释：** 上面四种是我们“风险度量工具箱”，你可以按需要打开某几项组合。

## 5. 组件聚合两种策略

### 5.1 可学习权重 (Softmax / Softplus)
原始参数 $\theta_i$：
1. Softmax：$w_i = \frac{e^{\theta_i}}{\sum_j e^{\theta_j}}$  
2. Softplus 再归一化：$a_i=\text{softplus}(\theta_i),\; w_i= a_i/\sum_j a_j$

聚合：
$$
R_{score} = \sum_i w_i R_i
$$

熵正则：
$$
H(w) = - \sum_i w_i \log(w_i+\epsilon), \quad L \leftarrow L - \lambda_{ent} H(w)
$$

**通俗解释：** 给每个风险指标一个“注意力”权重，让模型自己学哪个更重要；熵正则像是在提醒“别把全部筹码都压在一家”。

### 5.2 log-sigma 不确定性加权（互斥替代）
每组件学习 $\log\sigma_i$：
$$
R_{score} = \frac{1}{|\mathcal{C}|} \sum_i R_i e^{-\log\sigma_i} = \frac{1}{|\mathcal{C}|} \sum_i \frac{R_i}{\sigma_i}
$$
罚项：
$$
L_{log\sigma} = \lambda_{\sigma} \sum_i \log \sigma_i
$$
（鼓励不过度放大 $\sigma_i$ 以“淡化”组件风险）

**通俗解释：** 不再给显式概率权重，而是假设每个风险有“不确定性”尺度；越不确定（\(\sigma\) 大）该项被折扣，但你要为“声称不确定”付出代价（罚项）。

### 5.3 模式比较与选择实践指南

#### 5.3.1 “组件”定义澄清
本文“组件”(risk sub-component) 指聚合前的独立风险指标：当前实现为 MinDist、TTC、PET、Overlap 四项（可裁剪）。未来若添加速度差、异常加速度、逆行检测等，每一项也将自然纳入 \(\mathcal{C}\)。

**通俗解释：** 组件就是“风险传感器”输出的四个小分数；以后多加传感器，也就是多几个组件。

#### 5.3.2 两种聚合机制对比

| 维度 | python Code/main.py \
  --config config/Interaction.py \
  --test Code/data/Interation/DR_USA_Intersection_EP1/train \
  --ckpt log_formal_mamba_component_weights \
  --eval-only --fpc-finetune --quiet-test (weights + 熵) | log-sigma 不确定性模式 |
|------|---------------------------|------------------------|
| 可解释性 | 权重 \(w_i\) 直接给出“重要性” | 需同时看 \(1/\sigma_i\) 与罚项，间接一些 |
| 收敛速度 | 往往较快（参数少） | 可能稍慢（需平衡罚项） |
| 适用场景 | 指标较少，需明确“主导”排序 | 指标多且尺度差异 / 噪声差异大 |
| 防单一主导 | 倚赖熵正则 / 归一化 | 通过放大 \(\sigma\) 自动降该项权重 |
| 超参敏感性 | 熵系数 \(\lambda_{ent}\) 需调 | 罚项系数 \(\lambda_{\sigma}\) 需调 |
| 与 Autoscale 互动 | R_score 直接线性加权 | 先被 1/σ 缩放，常更平滑 |
| 产出可视化 | 权重曲线直观 | 需画 (R_i, σ_i) 双曲线 |

**简要结论：**
*想解释“哪个风险更重要”* → 首选组件权重模式；
*想自动吸收尺度/噪声差异* → 选 log-sigma 模式；
*指标未来会扩展* → 提前规划 log-sigma；
*只做初版论文/演示* → 组件权重曲线更直观。

**通俗解释：** 要看“谁最重要”就用权重；要“自动调不同量纲”就用 log-sigma。

#### 5.3.3 现阶段观测（短跑测试）
当前 2 epoch / 少 batch 试验：log_sigma_penalty 数值极小（≈1e−4），σ 尚未充分分化；Overlap 数值远高于其他组件（~4.3 vs 0.006 级别），短期更像被 scale 主导，尚不足比较长期优劣。

**通俗解释：** 现在训练太短，还看不出谁更好；只是看到 overlap 暂时最大声。

#### 5.3.4 推荐 A/B 实验流程
1. 固定 seed、数据子集、学习率与 Autoscale 参数。 
2. 配置 A：`RISK_USE_LOG_SIGMA=False`，启用 learnable weights + 适中 \(\lambda_{ent}\)（例 0.01）。  
	配置 B：`RISK_USE_LOG_SIGMA=True`，禁用权重，设 \(\lambda_{\sigma}=0.01\)（再做 0.02/0.05 网格）。
3. 训练 ≥20 epoch，收集每 2–3 epoch：ADE/FDE、risk_score_raw 均值与方差、risk_contrib 占比稳定度、权重熵或 σ 极差、top-q% 高风险样本组成。  
4. 判断：
	- 权重模式若熵→0 且单一权重≈1：加大 \(\lambda_{ent}\) 或改用 log-sigma。
	- log-sigma 若全部 σ 几乎相同：减小 \(\lambda_{\sigma}\) 或延长训练。
	- 哪个模式下 risk_contrib 波动（相对标准差）更小，说明该模式更稳定。

**通俗解释：** 做一组“只改聚合方式”的双实验，量化谁更稳、谁更能区分组件。

#### 5.3.5 诊断指标补充
| 指标 | 目的 | 判定信号 |
|------|------|----------|
| 权重熵 H(w) | 检测权重塌缩 | 过低表示单点独大 |
| σ 极差 (σ_max/σ_min) | 不确定性区分度 | ≈1 表示尚未分化 |
| risk_contrib CV | 稳定性 (变异系数) | 低 CV 更平稳 |
| top-risk 交集 Jaccard | 模式切换一致性 | 低值显示采样排名差异大 |
| 组件归一化前后方差比 | 尺度主导性 | 大幅>1 说明需归一化 |

**通俗解释：** 这些指标像“体检表”，帮助你判断要不要换模式或调超参。

#### 5.3.6 快速决策建议
| 目标 | 推荐模式 | 额外操作 |
|------|----------|-----------|
| 演示/展示可解释性 | 权重+熵 | 画权重时间曲线 |
| 多尺度/噪声混合 | log-sigma | 监控 σ 分布 & 罚项 |
| 未来快速扩展组件 | log-sigma | 提前设定 σ 初值=0 |
| 组件数少且稳定 | 权重+熵 | 适中 λ_ent 避免塌缩 |

**通俗解释：** 不同场景有“默认首选”模式，对照这张表就能选。

#### 5.3.7 过渡策略
可采用“权重模式预热 → log-sigma 模式微调”：先训练若干 epoch 让风险结构成型，再解冻 log-sigma 参数、关闭权重分支，减少早期不确定性发散。

**通俗解释：** 先让模型学会“看哪些风险”，再换一种更智能的加权方式继续精修。

#### 5.3.8 常见误区
| 误区 | 说明 | 修正 |
|------|------|------|
| 同时开权重和 log-sigma | 参数可辨识性差 | 二选一；迁移时先冻结旧分支 |
| 熵系数太大 | 所有权重近均匀，无差异 | 降低 \(\lambda_{ent}\) 或做 schedule |
| log-sigma 罚项太小 | σ 全部迅速增大，风险被过度折扣 | 提升 \(\lambda_{\sigma}\) 并加上界裁剪 |
| 只看均值不看方差 | 忽略稳定性 | 记录 risk_contrib CV/波动 |

**通俗解释：** 这些是容易踩的坑，遇到了就按“修正”那列处理。

**通俗总结：** 本小节告诉你什么时候用哪套加权策略、怎么做科学对比、出了问题看哪些指标、踩坑后怎么补救。

## 6. Autoscale 风险全局缩放
目标：
$$
w_{risk} g_{risk} \mathbb{E}[R_{score}] \approx \gamma \cdot B_t
$$
其中 $B_t$ 为基础损失 EMA，$\gamma$ 为期望占比 (例如 0.01)。

求目标：
$$
g_{target} = \frac{\gamma B_t}{w_{risk} \bar{R}_t + \varepsilon}
$$
平滑更新：
$$
g_{risk} \leftarrow (1-\beta) g_{risk} + \beta \; \text{clip}(g_{target}, g_{min}, g_{max})
$$

最终注入：
$$
L_{total} = L_{base} - w_{risk} (g_{risk} R_{score}) + \mathbf{1}_{log\sigma} L_{log\sigma} - \mathbf{1}_{entropy} \lambda_{ent} H(w)
$$

**通俗解释：** Autoscale 就像自动调音量：希望风险项声音永远保持整个合奏里固定百分比，不会时大时小；内部通过观察“平均基础损失”和“平均风险”来动态调节缩放因子。

## 7. Multi-sample 风险估计
若每 batch 采样 $S$ 次：
$$ R_i = \frac{1}{S}\sum_{s=1}^S R_i^{(s)} $$
再按上述策略聚合为 $R_{score}$（降低方差）。重建 / KL 等仅显式用首样本展示加速训练日志。

**通俗解释：** 一次多采几条未来预测，算风险再平均，能让估计更稳；但为了省时间，其它耗时项只用第一条。

## 8. Kinematic 平滑正则
示例（实际实现可略有不同）：令 $V_t = \hat{Y}_t - \hat{Y}_{t-1}$，$A_t = V_t - V_{t-1}$：
$$
L_{kin} = \frac{1}{T_f-2} \sum_{t=2}^{T_f-1} \|A_t\|_2
$$

**通俗解释：** 想让轨迹看起来“自然不抖”，就惩罚加速度跳动；它像给曲线做平滑滤波。

## 9. 高风险样本挖掘
每 epoch 收集样本的 scaled risk 或 raw risk，计算分位数 $q_{0.9}$：
$$\text{HighRiskMask} = [ R^{(k)} \ge q_{0.9} ]$$
匹配样本保存其预测轨迹：`.pt` + 文本索引，便于后处理与可视化。

**通俗解释：** 每轮训练把最危险的那一批预测“抓出来”单独留档，方便以后看“事故现场回放”。

## 10. 日志扩展
### 10.1 TensorBoard
记录：`risk_score_raw`, `risk_score`, `risk_scaled`, `risk_global_scale`, `risk_contrib`, 各组件，`compw_entropy`, `log_sigma_penalty` 等。

**通俗解释：** TensorBoard 是“仪表盘”，这些指标让你直观看到风险在训什么、调得怎么样。

### 10.2 autoscale_log.csv 列
| 列名 | 含义 |
|------|------|
| epoch | 训练 epoch |
| risk_global_scale | $g_{risk}$ |
| risk_weight | $w_{risk}$ |
| risk_autoscale_target_frac | 目标占比 $\gamma$ |
| risk_autoscale_ema_base | $B_t$ (EMA) |
| risk_autoscale_ema_raw_score | $\bar{R}_t$ (EMA) |
| compw_* | 组件权重 (或替代为 log-sigma 路径下的静态) |
| risk_score_raw / risk_score | 聚合风险分数（原始 / 规范） |
| risk_scaled | $g_{risk} R_{score}$ |
| log_sigma_penalty | $\lambda_{\sigma}\sum_i \log\sigma_i$ |
| compw_entropy / compw_entropy_lambda | 熵与其系数 |
| compw_max | 最大组件权重（监控塌缩） |
| high_risk_q90 | 风险第 90 分位值 |

**通俗解释：** CSV 是结构化“黑匣子”，每行记录一轮训练后的核心状态，方便做图、回溯或写论文图表。

### 10.3 可视化脚本
`plot_log_sigma_risk.py`：绘制 risk / log_sigma_penalty 随 epoch 变化曲线，可扩展平滑参数。

**通俗解释：** 这是“画走势图”的小工具，能快速看到风险和不确定性是升还是降。

## 11. 设计权衡
| 机制 | 优势 | 风险/副作用 | 缓解措施 |
|------|------|-------------|----------|
| Autoscale | 减少手动调参 | 震荡或失配 | 平滑系数与裁剪 |
| 可学习权重 + 熵 | 关注关键风险、避免单点主导 | 熵过大减慢收敛 | 调小 $\lambda_{ent}$ 或分段调度 |
| log-sigma | 引入不确定性尺度 | 罚项过小膨胀 | 合理设定 $\lambda_{\sigma}$，可加 warmup |
| Multi-sample | 降低估计噪声 | GPU 成本上升 | 小规模 S，必要时分阶段开启 |
| 组件归一化 | 数值尺度一致 | 统计偏差 | EMA 稳定化 + 冻结初期 |
| 高风险样本保存 | 定性/后验分析 | I/O 膨胀 | 限制每 epoch 上限 |

**通俗解释：** 这张表告诉你每个机制的“好处 vs 代价”，以及怎么缓解副作用，便于取舍和组合。

## 12. 关键公式汇总
$$
\begin{aligned}
L_{total} &= w_{rec} L_{rec} + w_{wmse} L_{wmse} + w_{KL} L_{KL} + w_{adv} L_{adv} + w_{kin} L_{kin} \\
&\quad - w_{risk} g_{risk} R_{score} - \lambda_{ent} H(w) + \lambda_{\sigma} \sum_i \log\sigma_i \\
H(w) &= -\sum_i w_i \log(w_i+\epsilon) \\
R_{score}^{(\text{weights})} &= \sum_i w_i R_i \\
R_{score}^{(\text{log-}\sigma)} &= \frac{1}{|\mathcal{C}|}\sum_i R_i e^{-\log\sigma_i} \\
g_{target} &= \frac{\gamma B_t}{w_{risk} \bar{R}_t + \varepsilon} \\
g_{risk} &\leftarrow (1-\beta) g_{risk} + \beta\;\text{clip}(g_{target}, g_{min}, g_{max})
\end{aligned}
$$

**通俗解释：** 这一块是“速查墙”：忘了某个公式长什么样时，直接滚到这里找，对写文档/论文引用也方便。

## 13. 实践建议
1. 初始可禁用 log-sigma，仅调小 $w_{risk}$ + autoscale 验证稳定性。
2. 若组件风险差异较大，开启组件归一化或熵正则，逐步增大 $\lambda_{ent}$。
3. log-sigma 模式下：先用较小 $\lambda_{\sigma}$ (如 0.01) 观察是否过度膨胀，再按需提升。
4. 关注 autoscale_log.csv 中 `risk_contrib / (loss_base)` 曲线是否稳定在目标比例附近。
5. 高风险样本可基于 Overlap 或 PET 做细分标签，有助于失败模式分类。

**通俗解释：** 给你一套“先做什么后做什么”的经验顺序，少踩坑：先稳，再细调，再加花活。

## 14. 后续扩展方向
- Per-component EMA 标准化：$\tilde{R}_i = R_i / (EMA[R_i]+\epsilon)$。
- 风险 Curriculum：逐 epoch 线性或分段上调 $w_{risk}$ 或 $\gamma$。
- log-sigma 上下界裁剪 & Warmup：防止初期梯度不稳定。
- 高风险样本附带上下文 (地图 / 邻居 ID) 元数据。
- 增加 `--skip-final-fpc`、`--no-eval-during-train` 减少迭代时间。

**通俗解释：** 这里是“未来可以再做的增强清单”，想继续优化可以从这里挑。

## 15. 结论
通过上述机制，模型从“静态权重 + 单风险注入”演进到“多组件、可学习权重 / 不确定性、动态自适应占比”的风险驱动轨迹生成框架。该框架同时提升了：
1. 训练稳定性（Autoscale + 归一化）
2. 可解释性（组件/熵/log-sigma 日志）
3. 研究可拓展性（快速验证配置 + 高风险样本挖掘）

> 如需英文版或补充具体代码行索引，可在后续版本附录中加入。

**通俗解释：** 我们已经把“多种风险”成功接进模型，并让它更好控、可解释、可分析。接下来再按需要迭代细节即可。

---
**最后更新日期**：2025-10-02
（本节更新：补充 3.1 基础损失细化公式）
**最后更新日期**：2025-10-03
