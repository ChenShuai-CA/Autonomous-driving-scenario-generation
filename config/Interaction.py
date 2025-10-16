# model
OB_RADIUS = 10000   # observe radius, neighborhood radius
OB_HORIZON = 8      # number of observation frames
PRED_HORIZON = 25   # number of prediction frames (修正：论文表格显示应为25，不是12)
# group name of inclusive agents; leave empty to include all agents
# non-inclusive agents will appear as neighbors only
INCLUSIVE_GROUPS = ["car"]
RNN_HIDDEN_DIM = 512

# training
LEARNING_RATE = 1e-4 
BATCH_SIZE = 128
EPOCHS = 200       # total number of epochs for training (reverted after entropy validation)
EPOCH_BATCHES = 100 # number of batches per epoch, None for data_length//batch_size
TEST_SINCE = 120   # the epoch after which performing testing during training

# testing
PRED_SAMPLES = 20   # best of N samples
FPC_SEARCH_RANGE = range(40, 50)   # FPC sampling rate

# evaluation
WORLD_SCALE = 1

# ---------------- Risk Generation Extensions -----------------
# Enable risk-aware loss (adds a negative term to encourage risky but feasible interactions)
RISK_ENABLE = True
# Static base weight (can be ramped manually by editing or future scheduler)
RISK_WEIGHT = 0.2
# Global multiplicative scale applied to aggregated risk_score before乘以RISK_WEIGHT
# 用于快速放大风险项的梯度影响（仅影响损失减项，不改变各组件之间相对比例）
RISK_GLOBAL_SCALE = 1.0
# Component weights inside risk score (linear combination)
RISK_COMPONENT_WEIGHTS = {
	"risk_min_dist": 0.35,
	"risk_ttc": 0.15,
	"risk_pet": 0.25,
	"risk_overlap": 0.25,
}
# 是否将上述组件权重改为可学习参数
RISK_LEARN_COMPONENT_WEIGHTS = True  # 训练中会为每个组件引入一个可学习正权重
# 归一化模式: 'none' 使用 softplus 得到独立正值; 'softmax' 使权重总和为1
RISK_LEARN_COMPONENT_NORM = 'softmax'
# 熵正则系数（仅在 softmax 或临时归一化下有效，鼓励分布均匀，防止塌缩）
RISK_COMPW_ENTROPY_LAMBDA = 0.01
# Soft-min temperature for distance (higher => sharper min)
RISK_MIN_DIST_BETA = 3.0
# TTC exponential decay scale (seconds)
RISK_TTC_TAU = 1.5

# PET risk parameters
RISK_PET_DIST_TH = 2.5     # distance threshold for closeness (wider to increase recall)
RISK_PET_ALPHA = 6.0       # sigmoid slope (softer boundary for more stable gradients)
RISK_PET_BETA = 0.5        # exponential decay on time difference (more tolerant to slight timing offsets)
RISK_PET_GAMMA = 2.0       # neighbor softmax temperature (0 to disable)
RISK_PET_CONTINUOUS = True # use continuous-time relative motion approximation
RISK_PET_TIME_TEMP = 4.0   # time softmax temperature for continuous PET (emphasize earliest/strongest)

# Overlap risk parameters
RISK_OV_R_SELF = 0.5       # ego radius approximation (legacy circular mode)
RISK_OV_R_NEIGH = 0.5      # neighbor radius approximation (legacy circular mode)
RISK_OV_MARGIN = 0.3       # early activation margin (both modes)
RISK_OV_K = 15.0           # softplus sharpness
RISK_OV_TIME_TEMP = 4.0    # time softmax temperature
RISK_OV_NEIGH_TEMP = 3.0   # neighbor softmax temperature

# --- New: Oriented Bounding Box (OBB) overlap mode ---
# Enable OBB overlap (if False fallback to circular)
RISK_OV_USE_OBB = True
# Vehicle (ego) full length & width in world units (user specified)
RISK_OV_SELF_LENGTH = 4.5
RISK_OV_SELF_WIDTH = 1.8
# Neighbor assumed same size (can separate later if needed)
RISK_OV_NEIGH_LENGTH = 4.5
RISK_OV_NEIGH_WIDTH = 1.8
# Minimum speed threshold to derive heading; below this use default axis (1,0)
RISK_OV_MIN_SPEED = 1e-3
# Temperature for soft-min across separation axes when computing penetration (higher sharper)
RISK_OV_AXIS_BETA = 12.0
# Whether to log extra OBB diagnostics in risk components dict
RISK_OV_DEBUG = True

# Enable/disable individual components (allows quick ablation)
RISK_ENABLE_PET = True
RISK_ENABLE_OVERLAP = True

# Multi-sample risk aggregation (during training only)
RISK_MULTI_SAMPLES = 1      # >1 to enable multi-sample latent sampling for risk aggregation
RISK_MULTI_TEMP = 4.0       # softmax temperature across sample risk scores

# --- Uncertainty (log-sigma) weighting for risk aggregation ---
# 若启用，将使用每个风险组件的可学习 log_sigma 对应似然形式: component * exp(-log_sigma) + penalty * log_sigma
# 并忽略传统组件权重学习 (component weights 仍可保留但不参与最终聚合)。
# 使用组件权重模式（关闭 log-sigma 不确定性模式）
RISK_USE_LOG_SIGMA = False
# log_sigma 项的线性惩罚系数（越大越抑制 log_sigma 无限制增大）
RISK_LOG_SIGMA_PENALTY_W = 0.01

# ---------------- Risk band (keep risk within a range) -----------------
# 通过铰链惩罚将风险维持在一个区间内，避免过低（不危险）或过高（不合理）的样本占优。
RISK_BAND_ENABLE = True
RISK_BAND_MIN = 0.20
RISK_BAND_MAX = 0.60
RISK_BAND_WEIGHT = 0.1
# 使用 raw 风险（归一化前）还是归一化后的风险参与带约束
RISK_BAND_USE_RAW = True

# ---------------- Loss Weight Overrides (optional) -----------------
# 主损失权重可调，解决 rec 与 weighted MSE 双计导致的量纲偏置问题。
# 如果不需要改动，可保持默认或在其他 config 中省略这些字段。
LOSS_W_REC = 1.0          # 原始重构项权重
LOSS_W_WMSE = 0.2         # 降低时间加权 MSE 的影响（原先=1.0 造成双计）
LOSS_W_KL = 0.1
LOSS_W_ADV = 0.01
LOSS_W_KIN = 0.05
LOSS_COMBINE_REC_WMSE = False   # 若 True 则忽略单独权重，用 alpha 融合为单一 rec
LOSS_REC_WMSE_ALPHA = 0.3       # combined_rec = (1-alpha)*rec + alpha*wmse

# ---------------- Type-aware Kinematic Constraints (examples) -----------------
# 这些阈值和权重会被 main.py 注入模型，用于在训练期对预测轨迹施加“可微的超限惩罚”。
# 当前实现按车辆一套阈值；后续可扩展为车辆/行人/骑行者不同掩码与阈值。

# 软惩罚陡峭度（越大，超过阈值后的增长越“硬”）
KIN_SOFTPLUS_K = 10.0

# 车辆类阈值（单位见注释）
KIN_VMAX_VEHICLE = 22.2   # m/s, ≈ 80 km/h
KIN_AMAX_VEHICLE = 3.0    # m/s^2
KIN_JMAX_VEHICLE = 2.0    # m/s^3
KIN_KAPPA_MAX_VEHICLE = 0.3  # 1/m, 曲率上限
KIN_MU_VEHICLE = 0.5      # 无量纲，摩擦系数（用于侧向加速度 a_lat ≤ μ g）

# 各门限子项的权重（在合成 kinematic_loss 前会乘以 LOSS_W_KIN_LIMITS 统一缩放）
LOSS_W_KIN_V = 1.0
LOSS_W_KIN_A = 1.0
LOSS_W_KIN_J = 0.5
LOSS_W_KIN_KAPPA = 0.5
LOSS_W_KIN_FRIC = 0.5
LOSS_W_KIN_LIMITS = 0.1   # 所有门限惩罚的总缩放系数

# 如需更严格的运动学约束，可适当提高各 LOSS_W_KIN_* 或降低对应阈值；
# 如需更松弛的约束，反之。

# 提前占位（后续可按类型分别配置，例如 VRU/骑行者）：
# KIN_VMAX_VRU = 3.0
# KIN_AMAX_VRU = 1.5
# KIN_JMAX_VRU = 1.0
# KIN_KAPPA_MAX_VRU = 0.8
# KIN_MU_VRU = 0.7

# ---------------- Mamba + Multi-head Attention (formal run) -----------------
# 启用 Mamba 编码器 / 解码器替换 rnn_fx / rnn_fy
USE_MAMBA_ENCODER = True
USE_MAMBA_DECODER = True
# 可保持与 RNN_HIDDEN_DIM 一致，已在 enable_mamba 内部自动加投影适配
MAMBA_D_MODEL_ENC = RNN_HIDDEN_DIM
MAMBA_D_MODEL_DEC = RNN_HIDDEN_DIM
# 启用多头邻居注意力 (仅聚合邻居，不含完整 Transformer Block)
MHA_HEADS = 4
MHA_DROPOUT = 0.0
# ---------------------------------------------------------------------------

# ---------------- Map BCE (semantic lane adherence) -----------------
# 训练期地图约束：鼓励“车辆在车道、VRU 在人行通道”但允许可控的软化/渗透
MAP_BCE_ENABLE = True
MAP_BCE_WEIGHT = 0.03          # 略加强地图BCE以减少越界
MAP_BCE_CHANNEL_VEHICLE = 0
MAP_BCE_CHANNEL_VRU = 1
# 使用 .osm_xy 解析的全局栅格
# 如无该文件，请在 quick 配置或命令行覆盖到对应场景路径
MAP_OSM_XY_PATH = "Code/data/INTERACTION/INTERACTION-Dataset-DR-multi-v1_2/maps/DR_USA_Intersection_EP1.osm_xy"
MAP_RASTER_SIZE = (224, 224)
MAP_CHANNELS = 2
# “软约束”参数：
# - VRU 对车辆车道的渗透系数：0~0.3 较为合适，>0 表示 VRU 通道在车辆道上也有少量概率（便于产生过街/冲突场景）
MAP_VRU_LEAK_TO_VEHICLE = 0.05
# - 栅格边界软化（盒滤波核大小与迭代次数），平滑 BCE 的梯度，弱化像素级“锯齿”
MAP_RASTER_SOFTEN_K = 3
MAP_RASTER_SOFTEN_ITERS = 1
# 调试：打印/记录 MAP_BCE 采样概率统计（均值/最小/最大、OOB 比例、通道占比等）。正式训练可置 False 减少开销。
MAP_BCE_DEBUG = False

