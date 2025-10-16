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
EPOCHS = 205
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
RISK_PET_DIST_TH = 2.0     # distance threshold for closeness
RISK_PET_ALPHA = 8.0       # sigmoid slope
RISK_PET_BETA = 0.7        # exponential decay on time difference
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
RISK_USE_LOG_SIGMA = True
# log_sigma 项的线性惩罚系数（越大越抑制 log_sigma 无限制增大）
RISK_LOG_SIGMA_PENALTY_W = 0.01

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
