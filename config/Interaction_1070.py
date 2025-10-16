# Config tuned for NVIDIA GTX 1070 8GB to balance speed/memory and keep training-eval behavior
_doc = """
This file mirrors config/Interaction.py and overrides a few fields for 8GB GPUs:
- BATCH_SIZE: 128 -> 48 to avoid OOM
- TEST_SINCE: 120 -> 160 to reduce early eval overhead but still create ckpt-best during training
- MAP_BCE_DEBUG: True -> False to cut logging cost
"""

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
BATCH_SIZE = 48
EPOCHS = 200       # keep total epochs, but we'll start eval late to save time
EPOCH_BATCHES = 100 # number of batches per epoch, None for data_length//batch_size
TEST_SINCE = 120   # start eval later so ckpt-best is produced during training but with less overhead

# testing
PRED_SAMPLES = 20   # best of N samples
FPC_SEARCH_RANGE = range(40, 50)   # FPC sampling rate (will be overridden by CLI during search)

# evaluation
WORLD_SCALE = 1

# ---------------- Risk Generation Extensions -----------------
RISK_ENABLE = True
RISK_WEIGHT = 0.2
RISK_GLOBAL_SCALE = 1.0
RISK_COMPONENT_WEIGHTS = {
	"risk_min_dist": 0.35,
	"risk_ttc": 0.15,
	"risk_pet": 0.25,
	"risk_overlap": 0.25,
}
RISK_LEARN_COMPONENT_WEIGHTS = True
RISK_LEARN_COMPONENT_NORM = 'softmax'
RISK_COMPW_ENTROPY_LAMBDA = 0.01
RISK_MIN_DIST_BETA = 3.0
RISK_TTC_TAU = 1.5

# PET risk parameters
RISK_PET_DIST_TH = 2.5
RISK_PET_ALPHA = 6.0
RISK_PET_BETA = 0.5
RISK_PET_GAMMA = 2.0
RISK_PET_CONTINUOUS = True
RISK_PET_TIME_TEMP = 4.0

# Overlap risk parameters
RISK_OV_R_SELF = 0.5
RISK_OV_R_NEIGH = 0.5
RISK_OV_MARGIN = 0.3
RISK_OV_K = 15.0
RISK_OV_TIME_TEMP = 4.0
RISK_OV_NEIGH_TEMP = 3.0

# --- Oriented Bounding Box (OBB) overlap mode ---
RISK_OV_USE_OBB = True
RISK_OV_SELF_LENGTH = 4.5
RISK_OV_SELF_WIDTH = 1.8
RISK_OV_NEIGH_LENGTH = 4.5
RISK_OV_NEIGH_WIDTH = 1.8
RISK_OV_MIN_SPEED = 1e-3
RISK_OV_AXIS_BETA = 12.0
RISK_OV_DEBUG = True

# Enable/disable individual components
RISK_ENABLE_PET = True
RISK_ENABLE_OVERLAP = True

# Multi-sample risk aggregation (training only)
RISK_MULTI_SAMPLES = 1
RISK_MULTI_TEMP = 4.0

# Uncertainty (log-sigma) weighting for risk aggregation
RISK_USE_LOG_SIGMA = False
RISK_LOG_SIGMA_PENALTY_W = 0.01

# ---------------- Risk band (keep risk within a range) -----------------
RISK_BAND_ENABLE = True
RISK_BAND_MIN = 0.20
RISK_BAND_MAX = 0.60
RISK_BAND_WEIGHT = 0.1
RISK_BAND_USE_RAW = True

# ---------------- Loss Weight Overrides (optional) -----------------
LOSS_W_REC = 1.0
LOSS_W_WMSE = 0.2
LOSS_W_KL = 0.1
LOSS_W_ADV = 0.01
LOSS_W_KIN = 0.05
LOSS_COMBINE_REC_WMSE = False
LOSS_REC_WMSE_ALPHA = 0.3

# ---------------- Type-aware Kinematic Constraints -----------------
KIN_SOFTPLUS_K = 10.0
KIN_VMAX_VEHICLE = 22.2
KIN_AMAX_VEHICLE = 3.0
KIN_JMAX_VEHICLE = 2.0
KIN_KAPPA_MAX_VEHICLE = 0.3
KIN_MU_VEHICLE = 0.5
LOSS_W_KIN_V = 1.0
LOSS_W_KIN_A = 1.0
LOSS_W_KIN_J = 0.5
LOSS_W_KIN_KAPPA = 0.5
LOSS_W_KIN_FRIC = 0.5
LOSS_W_KIN_LIMITS = 0.1

# ---------------- Mamba + Multi-head Attention -----------------
USE_MAMBA_ENCODER = True
USE_MAMBA_DECODER = True
MAMBA_D_MODEL_ENC = RNN_HIDDEN_DIM
MAMBA_D_MODEL_DEC = RNN_HIDDEN_DIM
MHA_HEADS = 4
MHA_DROPOUT = 0.0

# ---------------- Map BCE (semantic lane adherence) -----------------
MAP_BCE_ENABLE = True
MAP_BCE_WEIGHT = 0.03
MAP_BCE_CHANNEL_VEHICLE = 0
MAP_BCE_CHANNEL_VRU = 1
MAP_OSM_XY_PATH = "Code/data/INTERACTION/INTERACTION-Dataset-DR-multi-v1_2/maps/DR_USA_Intersection_EP1.osm_xy"
MAP_RASTER_SIZE = (224, 224)
MAP_CHANNELS = 2
MAP_VRU_LEAK_TO_VEHICLE = 0.05
MAP_RASTER_SOFTEN_K = 3
MAP_RASTER_SOFTEN_ITERS = 1
MAP_BCE_DEBUG = False  # override: reduce logging
