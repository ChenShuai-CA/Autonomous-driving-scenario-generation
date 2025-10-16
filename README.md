  ![示例图片](gallery/DR_USA_Roundabout_FT_2024_02_13_23_05_47_016342.gif)
  ![示例图片](Report/ESE_6500_Final_Report_页面_1.png)
  ![示例图片](Report/ESE_6500_Final_Report_页面_2.png)
  ![示例图片](Report/ESE_6500_Final_Report_页面_3.png)
  ![示例图片](Report/ESE_6500_Final_Report_页面_5.png)
  ![示例图片](Report/ESE_6500_Final_Report_页面_6.png)

## 风险驱动轨迹生成扩展

本仓库在 SocialVAE 基础上引入“风险激励”项，通过在训练损失中加入负号鼓励模型探索更接近事故临界的交互情景，用于生成更“危险/稀有”的轨迹样本。

### 1. 风险组件 (Risk Components)
当前聚合的风险由以下组件加权求和（或 softmax 归一化后求期望）：

| 组件 | 含义 | 关键超参数 |
|------|------|-----------|
| risk_min_dist | 时间步内邻接体软最小距离的反函数 | RISK_MIN_DIST_BETA |
| risk_ttc | Time-To-Collision 指标的平滑衰减 | RISK_TTC_TAU |
| risk_pet | (连续) PET（Post-Encroachment Time）软时间/邻居聚合 | RISK_PET_* |
| risk_overlap | 圆形或 OBB 近似的时序穿插/重叠软判定 | RISK_OV_* |

聚合前可选“组件归一化”：针对每个组件维护 EMA 均值/方差做 z-score，降低量纲差异导致的主导；归一化后：

```
raw_risk = Σ_i w_i * component_i
normalized_risk = mean( z_i )
```

损失中使用 normalized_risk（若启用），但 autoscale 调整使用 raw_risk 保持物理可解释性。

### 2. 自适应风险缩放 (Autoscale)
目标：保持风险项贡献占主损失的一个固定比例 target_frac（默认 1%）。训练中维护两个 EMA：

```
ema_base ← (1-a)*ema_base + a*base_loss
ema_risk ← (1-a)*ema_risk + a*raw_risk
target_scale = (target_frac * ema_base) / (risk_weight * ema_risk)
risk_global_scale ← lerp(risk_global_scale, target_scale, β)
```

最终风险减项：`- risk_weight * risk_global_scale * (normalized_risk 或 raw_risk)`。

### 3. 可学习风险组件权重 (Learnable Component Weights)
开启方式（配置文件例如 `config/Interaction.py`）：

```
RISK_LEARN_COMPONENT_WEIGHTS = True
RISK_LEARN_COMPONENT_NORM = 'softmax'   # 或 'none'
RISK_COMPW_ENTROPY_LAMBDA = 0.01        # 熵正则系数
```

实现细节：
1. 初始静态权重写入 ParameterDict 的“未约束参数” (raw logits)。
2. `'softmax'` 模式下直接对 logits 做 softmax 得到概率分布；`'none'` 模式下用 softplus 得到正权重（不归一化）。
3. 训练时每个前向都可查看当前权重 (`model.get_learned_component_weights()`)；写入 autoscale CSV 为列 `compw_<name>`。

### 4. 熵正则 (Entropy Regularization)
目的：防止 softmax 权重塌缩到单一组件，鼓励信息探索；通过最大化 Shannon 熵：

```
H(p) = - Σ_i p_i log p_i
Loss ← Loss - λ * H(p)
```

参数：`RISK_COMPW_ENTROPY_LAMBDA` (=λ)。仅当 `RISK_LEARN_COMPONENT_WEIGHTS=True` 且 λ>0 时生效。

在 autoscale CSV 及最小脚本中可看到：
```
compw_entropy, compw_entropy_lambda
```

### 5. 最小验证脚本
使用随机 + 人工高危（邻居与自车接近）构造：

```
python Code/minimal_entropy_check.py --config config/Interaction.py
```
输出示例（截取）：
```
compw_entropy: 1.33
compw_entropy_lambda: 0.01
learned component weights: {...}
```

### 6. 可视化权重演化
训练过程中 `log_rebound/autoscale_log.csv` 会累积：

```
epoch, risk_global_scale, risk_weight, risk_autoscale_ema_base, risk_autoscale_ema_raw_score, compw_risk_min_dist, ...
```

可视化脚本（待补充 `Code/plot_compw_evolution.py`）将输出：
1. 组件权重曲线 (随 epoch)。
2. 风险全局缩放 risk_global_scale 演化。
3. 熵 compw_entropy（如已写入）。

### 7. 典型调参建议
| 目标 | 建议调整 |
|------|----------|
| 提升探索更危险交互 | 增大 RISK_WEIGHT 或 RISK_GLOBAL_SCALE（若 autoscale 关闭） |
| 控制风险项不主导 | 减小 target_frac 或 risk_weight；检查 risk_global_scale 是否爆炸 |
| 权重塌缩到单一组件 | 提高 RISK_COMPW_ENTROPY_LAMBDA 或改为 softmax 模式 |
| 组件尺度差异大 | 开启组件归一化 (默认启用) |
| 想关注单一风险类型 | 暂时关闭其它组件权重或在 softmax 前对对应 logit 加偏置 |

### 8. 下一步：不确定性加权 (规划)
参见后续章节“不确定性 (log-sigma) 方案草案”。该方案提供另一种自适应平衡组件贡献的方式，替代显式归一化与熵正则。

## 地图语义 BCE 与可视化配色说明

训练期的地图语义 BCE 与可视化叠加层保持一致的“语义-颜色”约定，便于对齐观测与诊断：

- 通道含义（MAP_CHANNELS=2 时）
  - channel 0: vehicle 车道（lanelet 区域，剔除 keepout）
  - channel 1: VRU 区域（crosswalk/footway/sidewalk/pedestrian/pedestrian_area，多边形并集，剔除 keepout）
- keepout 并非单独 BCE 通道，但在可视化中以红色显示（用于直观标注“禁止/不可行”区域）。

可视化叠加层默认颜色与透明度：

- vehicle 车道：蓝色（Blues），alpha ≈ 0.8 × overlay_alpha（默认 overlay_alpha=0.28）
- VRU 区域：绿色（Greens），alpha = overlay_alpha
- keepout：红色（Reds），alpha ≈ 0.9 × overlay_alpha

叠加层开关（生成 GIF 时可通过命令行控制）：

- 开/关 VRU：`--overlay-vru` / `--no-overlay-vru`
- 开/关 vehicle：`--overlay-veh` / `--no-overlay-veh`
- 开/关 keepout：`--overlay-keepout` / `--no-overlay-keepout`
- 设置透明度：`--overlay-alpha 0.25`
- 显示/隐藏图例：`--overlay-legend` / `--no-overlay-legend`

注意事项：

- 训练 BCE 使用的通道 raster 来自 `.osm_xy` 解析后的全局栅格，与 GIF 叠加层使用同一 `world2map`（整张地图 bbox），确保像素-世界坐标一致。
- 若某场景 `.osm_xy` 缺少 VRU 标注，VRU 通道与叠加层可能为空（不会绘制），车辆车道与 keepout 仍可显示。
- 若需要更多通道（例如将 keepout 也纳入 BCE），可将 `MAP_CHANNELS` 扩展为 3，并在数据加载与损失函数中同步改动。

## 不确定性 (log-sigma) 方案（原型已集成）

### 核心想法
将每个风险组件视作一个“噪声观测”或“任务” (multi-task uncertainty weighting 思路)，引入可学习的对数标准差 `log_sigma_i`，用高斯似然近似：

```
L_risk = Σ_i ( w_base * risk_i * exp(-log_sigma_i) + log_sigma_i )
```

或若组件本身已非负且希望放大差异，可直接用：

```
Aggregate = Σ_i risk_i * exp(-log_sigma_i)
Penalty = Σ_i log_sigma_i
```

最终替换当前：`Σ_i w_i * risk_i`，并可移除：
1. 组件 z-score 归一化（由自适应 σ 调节尺度）
2. softmax 权重及熵正则（由 log_sigma 的竞争关系内在调节）

### 优点
| 维度 | 现有 (软归一+熵) | log-sigma 方案 |
|------|-----------------|---------------|
| 防止极端主导 | 依赖熵正则 | σ 自动增大主导项的对数项惩罚 |
| 解释性 | 需要同时看归一化 & 权重 | σ 直接表示“不确定/噪声” |
| 超参 | 需调 λ | 无需熵 λ（仍可保留 fallback） |
| 数值稳定 | 受 z-score 初期偏移影响 | 仅需 clamp log_sigma 范围 |

### 已实现原型开关
在 config 中（示例添加字段）：
```
RISK_USE_LOG_SIGMA = True              # 启用后忽略 learn_component_weights 聚合
RISK_LOG_SIGMA_PENALTY_W = 1.0         # Σ log_sigma_i 的惩罚系数
```
内部逻辑：若启用 log-sigma，则 risk 组件不再执行 z-score 归一化（自动跳过），聚合为：
```
agg = mean_i( comp_i * exp(-log_sigma_i) )
loss_risk_part = agg + penalty_w * Σ log_sigma_i
```
同时 raw_risk 仍保留用于 autoscale。

### 初始实现草图（已落地，并做了稳定性 clamp）

1. 在 `SocialVAE.__init__` 中添加：
```python
self.risk_log_sigmas = torch.nn.ParameterDict({
  k: torch.nn.Parameter(torch.zeros(())) for k in self.risk_component_weights.keys()
})
```
2. 聚合阶段（risk_dict 得到各 component）：
```python
terms = []
penalty = 0
for k,v in risk_components.items():
  ls = self.risk_log_sigmas[k]
  sigma = torch.exp(ls)
  terms.append(v * torch.exp(-ls))  # v / sigma
  penalty = penalty + ls            # log sigma 正则
aggregate = torch.stack(terms).mean()  # 或 sum
aggregate = aggregate + penalty_weight * penalty
```
3. 用 `aggregate` 替换现有 risk_score；保留原 raw_risk 便于回溯。
4. 可选： penalty_weight < 1 缓和对数项影响（默认 1）。

### 与 Autoscale 兼容
autoscale 仍基于“raw_risk” (简单平均或权重和)，以免 σ 的动态导致 target_frac 失真；训练中只调整最终减项的实际强度。

### 迁移策略
1. 第一个 epoch 冻结 log_sigma（避免初期数据稀疏震荡）。
2. 观测 log_sigma 是否向异常大值漂移；若是，添加 clamp：`ls.data.clamp_(min=-4, max=4)`。
3. 若希望保留 softmax 权重，可叠加：`final = Σ_i softmax(w_i) * risk_i * exp(-ls_i)`，但一般不推荐（冗余）。

### 最小评估指标
| 指标 | 期望提升 |
|------|----------|
| 组件贡献方差 | 降低 | 
| 生成样本危险性分布尾部密度 | 上升 | 
| 收敛稳定性 (loss 曲线振幅) | 改善 |

### 风险
1. 若某组件梯度极小，log_sigma 可能发散（需最小值约束）。
2. 与 KL / 其它 loss 可能存在耦合，需监控总体 loss 权重比例。

### 后续扩展
## 高风险样本与权重塌缩监控

### 高风险样本采样
训练每个 epoch 结束：
1. 收集当轮所有 batch 的 `risk_score_raw`。
2. 计算 90% 分位 (q90)。
3. 将 ≥ q90 的 batch 索引与对应 risk_score_raw 写入 `ckpt_dir/high_risk_samples/epoch_<E>_sel.txt`。
4. 在 autoscale CSV 中写入 `high_risk_q90`。

用于后续：快速定位危险交互，配合可视化脚本进一步质检。

### 权重塌缩监控 (Softmax 模式)
若 `max(compw)` ≥ 0.85 连续 3 个 epoch：
1. 自动将 `compw_entropy_lambda *= 1.5`。
2. 在控制台打印提示，并在 CSV 中刷新新系数。

此策略借助熵放大抑制权重单峰化，无需人工频繁监控。

可推广到：预测不止标量 risk_i，而是 (μ_i, σ_i) 并对其计算 KL 或 Aleatoric 不确定性；再进一步用于采样加权或 curriculum。


