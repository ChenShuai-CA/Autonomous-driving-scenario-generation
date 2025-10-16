# Gallery 使用说明（Top-K 危险场景）

本目录用于收集批量生成的“易发生事故”场景（GIF + CSV）。每次汇总会在当前目录下生成一个 `top{K}_YYYYMMDD_HHMMSS` 子目录，包含按 risk_score 降序的前 K 个结果（若不足 K，则收集全部可用样本）。

## 目录结构
- `top{K}_YYYYMMDD_HHMMSS/`
  - `top100_meta.csv`：元信息清单，字段包括
    - `gif`、`risk_score`、`src_case_id`、`window_start`、`ego_id`、`ego_type`
    - `neighbor_ids`、`neighbor_types`、`highlight_neighbor_ids`
  - 每个样本以下列文件成对出现（同名前缀）：
    - `.gif`：场景动画
    - `.csv`：该场景所有候选预测的重排名指标（ADE/out-of-lane/jerk 等）
    - `_trajs.csv`：该场景所有轨迹（ego 历史/真值/生成 + 所有邻居历史/未来）
    - `_trajs_highlight.csv`：仅导出关键高亮邻居，附带 vx, vy, ax, ay, heading_deg, speed

## 颜色与图例
- 预测轨迹颜色（重排名 Top-1/2/3）：
  - Top-1: 绿色 `#00FF37`
  - Top-2: 蓝色 `#0881c6`
  - Top-3: 紫色 `#964EEE`
- Ego 绘制：
  - 历史段为黑色折线；若被判定为危险（risk_score ≥ 阈值），Ego 的标记/矩形会以红色强调，并在左上角显示文字 “DANGEROUS”。
- 邻居绘制（按类型）：
  - 行人/骑行者（INTERACTION 中常见为 `pedestrian/bicycle`）：绿色点列 `#2ca02c`
  - 自行车：蓝色矩形 `#1f77b4`
  - 机动车：深灰矩形 `#333333`
  - 高亮邻居（贡献最大的若干个）：橙色 `#ff6600`
- 语义栅格叠加（来自 `.osm_xy` 的全局栅格）：
  - VRU 区域：绿色（Greens）
  - 车辆车道：蓝色（Blues）
  - Keepout 区域：红色（Reds）
  - 透明度可通过 `--overlay-alpha` 调整；是否显示图例可通过 `--overlay-legend/--no-overlay-legend` 控制；各通道开关见下文。

## 可视化开关（命令行）
- `--overlay-vru/--no-overlay-vru`：显示/隐藏 VRU 栅格
- `--overlay-veh/--no-overlay-veh`：显示/隐藏 车辆车道栅格
- `--overlay-keepout/--no-overlay-keepout`：显示/隐藏 keepout 栅格
- `--overlay-alpha`：叠加层透明度（默认 0.28）
- `--overlay-legend/--no-overlay-legend`：是否绘制叠加层图例

地图底图（车道线条）使用 Lanelet2 渲染；若未安装 Lanelet2 或缺少 `.osm`，将自动跳过底图绘制。语义栅格使用 `.osm_xy` 的 world2map 仿射变换进行投影，三通道分别对应 VRU/vehicle/keepout。

## 重排名与硬性筛选
- 重排名（用于选择可视化的 Top-K 预测）：
  - 复合分数：`score = ADE + λ_lane * out_frac + λ_jerk * jerk_mean`
  - 参数：`--lambda-lane`（出道惩罚系数）、`--lambda-jerk`（jerk 平滑惩罚系数）
- 硬性筛选（生成期守门）：
  - `--filter-enable` 开启后，对候选轨迹逐条检查：
    - 出道比例 `out_frac <= --filter-outfrac-max`
    - 平均 jerk `<= --filter-jerk-max`
    - 最大相邻转角（度）`<= --filter-turn-deg-max`
  - 若当前 Top-K 中有不满足者，会从“下一名次”中回填满足条件的候选；若最终仍无满足者，当前版本会回退到原始选择（确保有图可看）。如需“严格拒绝”，可进一步加入“无合格则整例跳过”的选项。

## 如何再次汇总 Top-K
- 生成结束后在工作区根目录执行（已在脚本中处理不足 K 的情况，且按 GIF 去重）：
  - `python scripts/collate_gallery.py --output-dir Output --gallery-root gallery --topk 100`
- 新目录示例：`gallery/top100_YYYYMMDD_HHMMSS/`

## 小贴士
- `top100_meta.csv` 可按 `risk_score` 排序，快速定位风险最高的样本；也可结合 `_trajs_highlight.csv` 分析关键邻居的速度/加速度/航向等。
- 若发现转角过大或频繁“回退选择”，可放宽 `--filter-turn-deg-max`（如 90/120）或适度增大 `--lambda-lane` 抑制出道。
