# Quick config enabling Mamba encoder+decoder and multi-head neighbor attention
OB_RADIUS = 10000
OB_HORIZON = 8
PRED_HORIZON = 25
INCLUSIVE_GROUPS = ["car"]
RNN_HIDDEN_DIM = 512
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
EPOCHS = 2
EPOCH_BATCHES = 20
TEST_SINCE = 2
PRED_SAMPLES = 20
FPC_SEARCH_RANGE = range(40,42)
WORLD_SCALE = 1
# ---- Risk settings (clone of quick with log-sigma path optional) ----
RISK_ENABLE = True
RISK_WEIGHT = 0.2
RISK_GLOBAL_SCALE = 1.0
RISK_COMPONENT_WEIGHTS = {"risk_min_dist":0.35,"risk_ttc":0.15,"risk_pet":0.25,"risk_overlap":0.25}
RISK_LEARN_COMPONENT_WEIGHTS = True
RISK_LEARN_COMPONENT_NORM = 'softmax'
RISK_COMPW_ENTROPY_LAMBDA = 0.01
RISK_MIN_DIST_BETA = 3.0
RISK_TTC_TAU = 1.5
RISK_PET_DIST_TH = 2.0
RISK_PET_ALPHA = 8.0
RISK_PET_BETA = 0.7
RISK_PET_GAMMA = 2.0
RISK_PET_CONTINUOUS = True
RISK_PET_TIME_TEMP = 4.0
RISK_OV_R_SELF = 0.5
RISK_OV_R_NEIGH = 0.5
RISK_OV_MARGIN = 0.3
RISK_OV_K = 15.0
RISK_OV_TIME_TEMP = 4.0
RISK_OV_NEIGH_TEMP = 3.0
RISK_OV_USE_OBB = True
RISK_OV_SELF_LENGTH = 4.5
RISK_OV_SELF_WIDTH = 1.8
RISK_OV_NEIGH_LENGTH = 4.5
RISK_OV_NEIGH_WIDTH = 1.8
RISK_OV_MIN_SPEED = 1e-3
RISK_OV_AXIS_BETA = 12.0
RISK_OV_DEBUG = False
RISK_ENABLE_PET = True
RISK_ENABLE_OVERLAP = True
RISK_MULTI_SAMPLES = 1
RISK_MULTI_TEMP = 4.0
RISK_USE_LOG_SIGMA = True
RISK_LOG_SIGMA_PENALTY_W = 0.02
# ---- Risk band (keep risk within a range) ----
RISK_BAND_ENABLE = True
RISK_BAND_MIN = 0.20
RISK_BAND_MAX = 0.60
RISK_BAND_WEIGHT = 0.1
RISK_BAND_USE_RAW = True
# ---- Loss weights ----
LOSS_W_REC = 1.0
LOSS_W_WMSE = 0.2
LOSS_W_KL = 0.1
LOSS_W_ADV = 0.01
LOSS_W_KIN = 0.05
LOSS_COMBINE_REC_WMSE = False
LOSS_REC_WMSE_ALPHA = 0.3
# ---- Mamba + Multi-head flags ----
USE_MAMBA_ENCODER = True
USE_MAMBA_DECODER = True
MAMBA_D_MODEL_ENC = RNN_HIDDEN_DIM
MAMBA_D_MODEL_DEC = RNN_HIDDEN_DIM
MHA_HEADS = 4
MHA_DROPOUT = 0.0

# ---- Map BCE (model side wired; data returns dummy raster/affine) ----
MAP_BCE_ENABLE = True
MAP_BCE_WEIGHT = 0.02
MAP_BCE_CHANNEL_VEHICLE = 0
MAP_BCE_CHANNEL_VRU = 1
# Enable MAP_BCE debug stats to log p-mean/min/max and OOB fractions during sanity runs
MAP_BCE_DEBUG = True
# Optional: use .osm_xy path if available; set to your dataset's map
# e.g., "Code/data/Interation/DR_USA_Intersection_EP1/DR_USA_Intersection_EP1.osm_xy"
MAP_OSM_XY_PATH = "Code/data/INTERACTION/INTERACTION-Dataset-DR-multi-v1_2/maps/DR_USA_Intersection_EP1.osm_xy"
# Rasterization canvas (H,W) and channels
MAP_RASTER_SIZE = (224, 224)
MAP_CHANNELS = 2
MAP_VRU_LEAK_TO_VEHICLE = 0.10
MAP_RASTER_SOFTEN_K = 3
MAP_RASTER_SOFTEN_ITERS = 1
