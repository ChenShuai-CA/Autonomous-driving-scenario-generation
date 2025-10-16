#tensorboard --logdir=C:\Users\39829\Desktop\SocialVAE\log_rebound --port 8123

import sys
import os
# IMPORTANT: Previously this script unconditionally overwrote sys.argv with a hardcoded
# default dataset/config, which prevented external CLI flags (e.g. --risk-scale,
# --max-train-batches) from taking effect. We now only inject default arguments when
# the user did not supply any (len(sys.argv) == 1). Set env DEFAULT_ARGS=0 to force
# skipping even in that case.
if len(sys.argv) == 1 and os.environ.get("DEFAULT_ARGS", "1") == "1":
    # Minimal sensible defaults for interactive/no-arg launches.
    sys.argv = [
        "main.py",
        "--train","Code/data/Interation/DR_USA_Intersection_EP1/train",
        "--test", "Code/data/Interation/DR_USA_Intersection_EP1/train",
        "--ckpt", "log_rebound",
        "--config", "config/Interaction.py"
    ]

import os, sys, time
import math
import importlib
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import csv

from social_vae import SocialVAE
from data import Dataloader
from utils import ADE_FDE, FPC, seed, get_rng_state, set_rng_state



import argparse
import json
parser = argparse.ArgumentParser()
parser.add_argument("--train", nargs='+', default=[])
parser.add_argument("--test", nargs='+', default=[])
parser.add_argument("--frameskip", type=int, default=1)
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--ckpt", type=str, default=None)
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--no-fpc", action="store_true", default=False)
parser.add_argument("--fpc-finetune", action="store_true", default=False)
parser.add_argument("--risk-scale", type=float, default=None, help="Override RISK_GLOBAL_SCALE for quick experiments")
parser.add_argument("--max-train-batches", type=int, default=None, help="Limit number of training batches per epoch (for rapid experiments)")
parser.add_argument("--fpc-range", type=str, default=None, help="Override FPC_SEARCH_RANGE, format: start,end,step (end exclusive, like Python range)")
parser.add_argument("--no-resume", action="store_true", default=False, help="Do not load ckpt-last even if it exists (start training from scratch)")
parser.add_argument("--quiet-test", action="store_true", default=False, help="Silence per-batch test progress to avoid BrokenPipe in non-interactive outputs")
parser.add_argument("--eval-only", action="store_true", default=False, help="Load checkpoint (ckpt-best if present else ckpt-last) and run a single evaluation + optional FPC search, then exit")
parser.add_argument("--force-save-best-eval", action="store_true", default=False, help="In eval-only mode: force write ckpt-best with current base (fpc=1) ADE/FDE regardless of improvement")
parser.add_argument("--max-test-batches", type=int, default=None, help="Limit number of test batches per evaluation (speeds up eval/FPC search)")
parser.add_argument("--skip-fpc-search", action="store_true", default=False, help="Skip the post-training FPC range search step")
parser.add_argument("--pred-samples-eval", type=int, default=None, help="Override PRED_SAMPLES during evaluation to reduce compute")
parser.add_argument("--amp-eval", action="store_true", default=False, help="Enable AMP mixed precision during evaluation (use with caution)")

if __name__ == "__main__":
    settings = parser.parse_args()
    # ---------------- Dynamic config override (A) ----------------
    override_path = os.environ.get('CONFIG_PATH_OVERRIDE', None)
    if override_path is not None and os.path.isfile(override_path):
        cfg_path = override_path
        print(f"[INFO] Using CONFIG_PATH_OVERRIDE={cfg_path}")
    else:
        cfg_path = settings.config
    spec = importlib.util.spec_from_file_location("config", cfg_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    # --------------------------------------------------------------

    # Optional override of FPC search range via CLI (just parse; no eval here)
    if settings.fpc_range is not None:
        try:
            parts = [int(p) for p in settings.fpc_range.split(',')]
            if len(parts) == 3:
                start, end, step = parts
            elif len(parts) == 2:
                start, end = parts; step = 1
            else:
                raise ValueError
            if step == 0:
                # Interpret as a single FPC candidate [start]
                config.FPC_SEARCH_RANGE = [start]
            else:
                config.FPC_SEARCH_RANGE = list(range(start, end, step))
            print(f"[INFO] Override FPC_SEARCH_RANGE -> {config.FPC_SEARCH_RANGE}")
        except Exception as e:
            print(f"[WARN] Failed to parse --fpc-range '{settings.fpc_range}': {e}")

    # ---------------- Device, seed, and dataset setup ----------------
    if settings.device is None:
        settings.device = ("cuda:0" if torch.cuda.is_available() else "cpu")
    seed(settings.seed)
    init_rng_state = get_rng_state(settings.device)
    rng_state = init_rng_state

    # Common dataloader kwargs
    kwargs = dict(
        batch_first=False, frameskip=settings.frameskip,
        ob_horizon=config.OB_HORIZON, pred_horizon=config.PRED_HORIZON,
        device=settings.device, seed=settings.seed
    )
    train_data, test_data = None, None
    if settings.test:
        if config.INCLUSIVE_GROUPS is not None:
            inclusive = [config.INCLUSIVE_GROUPS for _ in range(len(settings.test))]
        else:
            inclusive = None
        test_dataset = Dataloader(
            settings.test, **kwargs, inclusive_groups=inclusive,
            batch_size=config.BATCH_SIZE, shuffle=False
        )
        test_data = torch.utils.data.DataLoader(
            test_dataset,
            collate_fn=test_dataset.collate_fn,
            batch_sampler=test_dataset.batch_sampler
        )

        # Define evaluation function to be used by both train-time eval and eval-only
        def test(model, fpc=1):
            model.eval()
            ADE_list, FDE_list = [], []
            set_rng_state(init_rng_state, settings.device)
            fpc = int(fpc) if fpc else 1
            max_tb = getattr(settings, 'max_test_batches', None)
            use_amp = bool(getattr(settings, 'amp_eval', False)) and str(settings.device).startswith('cuda')
            bcount = 0
            with torch.no_grad():
                for x, y, neighbor in test_data:
                    # Honor --max-test-batches limit
                    bcount += 1
                    # n_samples x PRED_HORIZON x N x 2
                    n_pred = getattr(config, 'PRED_SAMPLES', 0)
                    if settings.pred_samples_eval is not None:
                        n_pred = int(settings.pred_samples_eval)
                    if n_pred > 0 and fpc > 1:
                        y_multi = []
                        for _ in range(fpc):
                            # optional AMP for eval
                            if use_amp:
                                with torch.cuda.amp.autocast():
                                    y_multi.append(model(x, neighbor, n_predictions=n_pred))
                            else:
                                y_multi.append(model(x, neighbor, n_predictions=n_pred))
                        y_multi = torch.cat(y_multi, 0)  # FPC * n_pred x T x N x 2
                        # FPC selection per agent
                        cand = []
                        for i in range(y_multi.size(-2)):
                            cand.append(FPC(y_multi[..., i, :].cpu().numpy(), n_samples=n_pred))
                        # Re-stack selected trajectories per agent
                        y_pred = torch.stack([y_multi[_, :, i] for i, _ in enumerate(cand)], 2)
                    else:
                        if use_amp:
                            with torch.cuda.amp.autocast():
                                y_pred = model(x, neighbor, n_predictions=n_pred)
                        else:
                            y_pred = model(x, neighbor, n_predictions=n_pred)
                    ade_b, fde_b = ADE_FDE(y_pred, y)
                    if n_pred > 0:
                        ade_b = torch.min(ade_b, dim=0)[0]
                        fde_b = torch.min(fde_b, dim=0)[0]
                    ADE_list.append(ade_b)
                    FDE_list.append(fde_b)
                    if max_tb is not None and bcount >= int(max_tb):
                        break
            ADE = torch.cat(ADE_list) if len(ADE_list) else torch.tensor([], device=settings.device)
            FDE = torch.cat(FDE_list) if len(FDE_list) else torch.tensor([], device=settings.device)
            if ADE.numel() == 0:
                # Fallback to zeros if no batches were evaluated (shouldn't happen in normal runs)
                return torch.tensor(float('nan')), torch.tensor(float('nan'))
            if torch.is_tensor(config.WORLD_SCALE) or getattr(config, 'WORLD_SCALE', 1) != 1:
                if not torch.is_tensor(config.WORLD_SCALE):
                    config.WORLD_SCALE = torch.as_tensor(config.WORLD_SCALE, device=ADE.device, dtype=ADE.dtype)
                ADE = ADE * config.WORLD_SCALE
                FDE = FDE * config.WORLD_SCALE
            return ADE.mean(), FDE.mean()

    if settings.train:
        print(settings.train)
        if config.INCLUSIVE_GROUPS is not None:
            inclusive = [config.INCLUSIVE_GROUPS for _ in range(len(settings.train))]
        else:
            inclusive = None
        train_dataset = Dataloader(
            settings.train, **kwargs, inclusive_groups=inclusive, 
            flip=True, rotate=True, scale=True,
            batch_size=config.BATCH_SIZE, shuffle=True, batches_per_epoch=config.EPOCH_BATCHES,
            include_map_meta=getattr(config, 'MAP_BCE_ENABLE', False),
            map_raster_size=getattr(config, 'MAP_RASTER_SIZE', (224,224)),
            map_channels=getattr(config, 'MAP_CHANNELS', 2),
            map_osm_xy_path=getattr(config, 'MAP_OSM_XY_PATH', None),
            map_vru_leak_to_vehicle=getattr(config, 'MAP_VRU_LEAK_TO_VEHICLE', 0.0),
            map_raster_soften_k=getattr(config, 'MAP_RASTER_SOFTEN_K', 0),
            map_raster_soften_iters=getattr(config, 'MAP_RASTER_SOFTEN_ITERS', 0)
        )
        train_data = torch.utils.data.DataLoader(train_dataset,
            collate_fn=train_dataset.collate_fn,
            batch_sampler=train_dataset.batch_sampler
        )
        batches = train_dataset.batches_per_epoch

    ###############################################################################
    #####                                                                    ######
    ##### load model                                                         ######
    #####                                                                    ######
    ###############################################################################
    # prepare risk params if present in config
    risk_params = {}
    if hasattr(config, 'RISK_ENABLE'):
        risk_params = dict(
            enable=getattr(config, 'RISK_ENABLE', False),
            weight=getattr(config, 'RISK_WEIGHT', 0.0),
            risk_global_scale=getattr(config, 'RISK_GLOBAL_SCALE', 1.0),
            component_weights=getattr(config, 'RISK_COMPONENT_WEIGHTS', {}),
            learn_component_weights=getattr(config, 'RISK_LEARN_COMPONENT_WEIGHTS', False),
            learn_component_norm=getattr(config, 'RISK_LEARN_COMPONENT_NORM', 'none'),
            beta=getattr(config, 'RISK_MIN_DIST_BETA', 2.0),
            ttc_tau=getattr(config, 'RISK_TTC_TAU', 1.5),
            # PET params
            pet_dist_th=getattr(config, 'RISK_PET_DIST_TH', 2.0),
            pet_alpha=getattr(config, 'RISK_PET_ALPHA', 8.0),
            pet_beta=getattr(config, 'RISK_PET_BETA', 0.7),
            pet_gamma=getattr(config, 'RISK_PET_GAMMA', 0.0),
            pet_continuous=getattr(config, 'RISK_PET_CONTINUOUS', True),
            pet_time_temp=getattr(config, 'RISK_PET_TIME_TEMP', 4.0),
            enable_pet=getattr(config, 'RISK_ENABLE_PET', True),
            # Overlap params
            ov_r_self=getattr(config, 'RISK_OV_R_SELF', 0.5),
            ov_r_neigh=getattr(config, 'RISK_OV_R_NEIGH', 0.5),
            ov_margin=getattr(config, 'RISK_OV_MARGIN', 0.3),
            ov_k=getattr(config, 'RISK_OV_K', 15.0),
            ov_time_temp=getattr(config, 'RISK_OV_TIME_TEMP', 4.0),
            ov_neigh_temp=getattr(config, 'RISK_OV_NEIGH_TEMP', 3.0),
            enable_overlap=getattr(config, 'RISK_ENABLE_OVERLAP', True),
            # OBB overlap extensions
            ov_use_obb=getattr(config, 'RISK_OV_USE_OBB', False),
            ov_self_length=getattr(config, 'RISK_OV_SELF_LENGTH', 4.5),
            ov_self_width=getattr(config, 'RISK_OV_SELF_WIDTH', 1.8),
            ov_neigh_length=getattr(config, 'RISK_OV_NEIGH_LENGTH', 4.5),
            ov_neigh_width=getattr(config, 'RISK_OV_NEIGH_WIDTH', 1.8),
            ov_min_speed=getattr(config, 'RISK_OV_MIN_SPEED', 1e-3),
            ov_axis_beta=getattr(config, 'RISK_OV_AXIS_BETA', 12.0),
            ov_debug=getattr(config, 'RISK_OV_DEBUG', False),
            risk_multi_samples=getattr(config, 'RISK_MULTI_SAMPLES', 1),
            risk_multi_temp=getattr(config, 'RISK_MULTI_TEMP', 4.0),
            compw_entropy_lambda=getattr(config, 'RISK_COMPW_ENTROPY_LAMBDA', None),
            use_log_sigma=getattr(config, 'RISK_USE_LOG_SIGMA', False),
            log_sigma_penalty_w=getattr(config, 'RISK_LOG_SIGMA_PENALTY_W', 0.01),
        )
    model = SocialVAE(horizon=config.PRED_HORIZON, ob_radius=config.OB_RADIUS, hidden_dim=config.RNN_HIDDEN_DIM, risk_params=risk_params)
    # Inject optional kinematic thresholds/weights from config to model (if provided)
    kin_keys = [
        'KIN_SOFTPLUS_K',
        'KIN_VMAX_VEHICLE','KIN_AMAX_VEHICLE','KIN_JMAX_VEHICLE','KIN_KAPPA_MAX_VEHICLE','KIN_MU_VEHICLE',
        'LOSS_W_KIN_V','LOSS_W_KIN_A','LOSS_W_KIN_J','LOSS_W_KIN_KAPPA','LOSS_W_KIN_FRIC','LOSS_W_KIN_LIMITS',
        # generic loss weights (optional)
        'LOSS_W_REC','LOSS_W_WMSE','LOSS_W_KL','LOSS_W_ADV','LOSS_W_KIN','LOSS_COMBINE_REC_WMSE','LOSS_REC_WMSE_ALPHA',
    ]
    for k in kin_keys:
        if hasattr(config, k):
            try:
                setattr(model, k, getattr(config, k))
            except Exception:
                pass
    # Inject risk band configs if present
    band_keys = [
        'RISK_BAND_ENABLE','RISK_BAND_MIN','RISK_BAND_MAX','RISK_BAND_WEIGHT','RISK_BAND_USE_RAW'
    ]
    for k in band_keys:
        if hasattr(config, k):
            try:
                setattr(model, k, getattr(config, k))
            except Exception:
                pass
    # Inject map BCE configs if present
    map_keys = [
        'MAP_BCE_ENABLE','MAP_BCE_WEIGHT','MAP_BCE_CHANNEL_VEHICLE','MAP_BCE_CHANNEL_VRU','MAP_BCE_DEBUG'
    ]
    for k in map_keys:
        if hasattr(config, k):
            try:
                setattr(model, k, getattr(config, k))
            except Exception:
                pass
    # Optional Mamba enabling (backwards compatible). Config may define:
    #   USE_MAMBA_ENCODER, USE_MAMBA_DECODER, MAMBA_D_MODEL_ENC, MAMBA_D_MODEL_DEC
    if all(hasattr(model, attr) for attr in ["enable_mamba", "mamba_encoder", "mamba_decoder"]):
        use_m_enc = getattr(config, 'USE_MAMBA_ENCODER', False)
        use_m_dec = getattr(config, 'USE_MAMBA_DECODER', False)
        if use_m_enc or use_m_dec:
            d_enc = getattr(config, 'MAMBA_D_MODEL_ENC', None)
            d_dec = getattr(config, 'MAMBA_D_MODEL_DEC', None)
            try:
                model.enable_mamba(encoder=use_m_enc, decoder=use_m_dec, d_model_enc=d_enc, d_model_dec=d_dec)
                print(f"[INFO] Mamba enabled: encoder={use_m_enc}, decoder={use_m_dec}")
            except Exception as e:
                print(f"[WARN] Failed to enable Mamba modules: {e}")
    # Optional multi-head neighbor attention
    if hasattr(model, 'enable_multihead_attention'):
        mha_heads = getattr(config, 'MHA_HEADS', 0)
        mha_dropout = getattr(config, 'MHA_DROPOUT', 0.0)
        if mha_heads and mha_heads > 1:
            try:
                model.enable_multihead_attention(heads=mha_heads, dropout=mha_dropout)
                print(f"[INFO] Multi-head neighbor attention enabled: heads={mha_heads}, dropout={mha_dropout}")
            except Exception as e:
                print(f"[WARN] Failed to enable multi-head attention: {e}")
    model.to(settings.device)
    # CLI override for risk global scale if provided
    if settings.risk_scale is not None and hasattr(model, 'risk_global_scale'):
        model.risk_global_scale = settings.risk_scale
        if hasattr(model, 'risk_enable') and model.risk_enable:
            print(f"[INFO] Override risk_global_scale -> {model.risk_global_scale}")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    start_epoch = 0
    # 统一定义 ckpt 路径（即便 --no-resume 也要用于后续保存）
    if settings.ckpt:
        ckpt = os.path.join(settings.ckpt, "ckpt-last")
        ckpt_best = os.path.join(settings.ckpt, "ckpt-best")
    if settings.ckpt and not settings.no_resume:
        ckpt = os.path.join(settings.ckpt, "ckpt-last")
        ckpt_best = os.path.join(settings.ckpt, "ckpt-best")
        if os.path.exists(ckpt_best):
            state_dict = torch.load(ckpt_best, map_location=settings.device)
            ade_best = state_dict["ade"]
            fde_best = state_dict["fde"]
            fpc_best = state_dict["fpc"] if "fpc" in state_dict else 1
            # 记录最佳出现的 epoch（旧 ckpt 可能未包含 best_ade_epoch 字段）
            ade_best_epoch = state_dict.get('epoch', -1)
        else:
            ade_best = 100000
            fde_best = 100000
            fpc_best = 1
            ade_best_epoch = -1
        if train_data is None: # testing mode
            ckpt = ckpt_best
        if os.path.exists(ckpt) and not settings.no_resume:
            print("Load from ckpt:", ckpt)
            state_dict = torch.load(ckpt, map_location=settings.device)
            missing, unexpected = model.load_state_dict(state_dict["model"], strict=False)
            if missing:
                print(f"[INFO] Loaded ckpt with missing params (expected after adding learnable weights): {missing}")
                # If learnable weights enabled but params absent (old ckpt), re-run set_risk_params to init them
                if getattr(model, 'risk_learn_component_weights', False):
                    model.set_risk_params(
                        enable=model.risk_enable,
                        weight=model.risk_weight,
                        component_weights=model.risk_component_weights,
                        beta=model.risk_beta,
                        ttc_tau=model.risk_ttc_tau,
                        risk_global_scale=model.risk_global_scale,
                        learn_component_weights=model.risk_learn_component_weights,
                        learn_component_norm=model.risk_learn_component_norm
                    )
            if "optimizer" in state_dict:
                try:
                    optimizer.load_state_dict(state_dict["optimizer"])
                    rng_state = [r.to("cpu") if torch.is_tensor(r) else r for r in state_dict["rng_state"]]
                except ValueError as e:
                    print(f"[INFO] Skip loading optimizer state due to structure change: {e}")
            start_epoch = state_dict["epoch"]
    elif settings.ckpt and settings.no_resume:
        print('[INFO] --no-resume specified: starting from scratch, ignoring existing ckpt-last')
    end_epoch = start_epoch+1 if train_data is None or start_epoch >= config.EPOCHS else config.EPOCHS

    # 初始化评估最佳指标（即便未加载 ckpt）
    if settings.ckpt and ('ade_best' not in locals()):
        ade_best = 1e9
        fde_best = 1e9
        fpc_best = 1
        ade_best_epoch = -1

    if settings.train and settings.ckpt:
        logger = SummaryWriter(log_dir=settings.ckpt)
    else:
        logger = None

    # Ensure autoscale CSV contains risk_L_band/map_bce_loss columns for future rows (one-time header upgrade)
    if settings.ckpt:
        autoscale_path = os.path.join(settings.ckpt, 'autoscale_log.csv')
        if os.path.exists(autoscale_path):
            try:
                with open(autoscale_path, 'r') as rf:
                    reader = csv.DictReader(rf)
                    cols = reader.fieldnames or []
                    rows = list(reader)
                need_upgrade = False
                new_cols = cols[:] if cols else ['epoch']
                if 'risk_L_band' not in new_cols:
                    new_cols = (new_cols + ['risk_L_band']) if new_cols else ['epoch','risk_L_band']
                    need_upgrade = True
                if 'map_bce_loss' not in new_cols:
                    new_cols = (new_cols + ['map_bce_loss']) if new_cols else ['epoch','map_bce_loss']
                    need_upgrade = True
                if need_upgrade:
                    with open(autoscale_path + '.tmp', 'w', newline='') as wf:
                        writer = csv.DictWriter(wf, fieldnames=new_cols)
                        writer.writeheader()
                        for r in rows:
                            r.setdefault('risk_L_band', '')
                            r.setdefault('map_bce_loss', '')
                            writer.writerow(r)
                    os.replace(autoscale_path + '.tmp', autoscale_path)
            except Exception as e:
                print(f"[WARN] Autoscale CSV header upgrade failed: {e}")

    # -------- Eval-only: load ckpt, run test once (+optional FPC search), persist logs/ckpt-best --------
    if settings.eval_only:
        if not settings.test:
            print('[ERROR] --eval-only requires --test dataset paths.'); sys.exit(1)
        chosen_ckpt = None
        if settings.ckpt:
            ckpt_best_path = os.path.join(settings.ckpt, 'ckpt-best')
            ckpt_last_path = os.path.join(settings.ckpt, 'ckpt-last')
            if os.path.exists(ckpt_best_path):
                chosen_ckpt = ckpt_best_path
            elif os.path.exists(ckpt_last_path):
                chosen_ckpt = ckpt_last_path
        if chosen_ckpt:
            try:
                state_dict = torch.load(chosen_ckpt, map_location=settings.device)
                missing, unexpected = model.load_state_dict(state_dict.get('model', state_dict), strict=False)
                if missing:
                    print(f"[INFO] Eval-only load missing params (expected for new risk weights): {missing}")
            except Exception as e:
                print(f"[WARN] Failed loading chosen ckpt for eval-only: {e}")
        # Base evaluation with fpc=1
        ade_eval, fde_eval = test(model, fpc=1)
        ade_val = float(ade_eval); fde_val = float(fde_eval)
        base_ade_val, base_fde_val = ade_val, fde_val
        print(f"[EVAL-ONLY] ADE={ade_val:.4f} FDE={fde_val:.4f} (fpc=1)")
        chosen_fpc = 1
        # Optional FPC search
        if settings.fpc_finetune and not settings.no_fpc:
            precision = 2
            trunc = lambda v: np.trunc(v*10**precision)/10**precision
            ade_list, fde_list, fpcs = [], [], []
            for fpc in getattr(config, 'FPC_SEARCH_RANGE', [1]):
                a, f = test(model, fpc)
                ade_list.append(trunc(a.item())); fde_list.append(trunc(f.item())); fpcs.append(fpc)
            idx = int(np.argmin(np.add(ade_list, fde_list)))
            ade_val, fde_val, chosen_fpc = ade_list[idx], fde_list[idx], fpcs[idx]
            print(f"[EVAL-ONLY] After FPC search: ADE={ade_val:.4f} FDE={fde_val:.4f} FPC={chosen_fpc}")
            # Append FPC meta to ckpt-best immediately (do not overwrite base ade/fde)
            if settings.ckpt:
                try:
                    ckpt_best_path = os.path.join(settings.ckpt, 'ckpt-best')
                    if os.path.exists(ckpt_best_path):
                        st_prev = torch.load(ckpt_best_path, map_location='cpu')
                        st_prev['fpc'] = chosen_fpc
                        st_prev['ade_fpc'] = ade_val
                        st_prev['fde_fpc'] = fde_val
                        torch.save(st_prev, ckpt_best_path)
                        print(f"[EVAL-ONLY] Wrote FPC meta to ckpt-best (fpc={chosen_fpc}, ade_fpc={ade_val:.4f}, fde_fpc={fde_val:.4f})")
                except Exception as e:
                    print(f"[WARN] Failed to write FPC meta to ckpt-best: {e}")

        # Save or update ckpt-best using base metrics
        if settings.ckpt:
            try:
                os.makedirs(settings.ckpt, exist_ok=True)
                ckpt_best_path = os.path.join(settings.ckpt, 'ckpt-best')
                prev_ade = None
                if os.path.exists(ckpt_best_path):
                    try:
                        st_prev = torch.load(ckpt_best_path, map_location='cpu')
                        if 'ade' in st_prev:
                            prev_ade = float(st_prev['ade'])
                    except Exception:
                        prev_ade = None
                should_save = settings.force_save_best_eval or (prev_ade is None) or (base_ade_val < prev_ade)
                if should_save:
                    epoch_meta = None
                    try:
                        if chosen_ckpt and os.path.exists(chosen_ckpt):
                            st_src = torch.load(chosen_ckpt, map_location='cpu')
                            epoch_meta = st_src.get('epoch', 'eval-only') if isinstance(st_src, dict) else 'eval-only'
                    except Exception:
                        epoch_meta = 'eval-only'
                    best_state = dict(
                        model=model.state_dict(),
                        ade=base_ade_val,
                        fde=base_fde_val,
                        epoch=(epoch_meta if epoch_meta is not None else 'eval-only')
                    )
                    # Attach FPC metadata if we searched
                    if settings.fpc_finetune and chosen_fpc is not None and isinstance(chosen_fpc, (int, float)):
                        best_state['fpc'] = chosen_fpc
                        best_state['ade_fpc'] = ade_val
                        best_state['fde_fpc'] = fde_val
                    torch.save(best_state, ckpt_best_path)
                    if prev_ade is None:
                        print(f"[EVAL-ONLY] Saved new ckpt-best (ade={base_ade_val:.4f}, fde={base_fde_val:.4f})")
                    else:
                        print(f"[EVAL-ONLY] Improved ckpt-best: {prev_ade:.4f} -> {base_ade_val:.4f}")
                else:
                    print(f"[EVAL-ONLY] Keep existing ckpt-best (ade={prev_ade:.4f})")
            except Exception as e:
                print(f"[WARN] Failed to save/update ckpt-best in eval-only: {e}")

            # Append eval-only row to eval_log.csv
            try:
                eval_log_path = os.path.join(settings.ckpt, 'eval_log.csv')
                exists_eval = os.path.exists(eval_log_path)
                fieldnames = ['timestamp','epoch','ade','fde','best_ade']
                best_ade_hist = ''
                if os.path.exists(os.path.join(settings.ckpt, 'ckpt-best')):
                    try:
                        st = torch.load(os.path.join(settings.ckpt, 'ckpt-best'), map_location='cpu')
                        best_ade_hist = float(st.get('ade', ''))
                    except Exception:
                        best_ade_hist = ''
                with open(eval_log_path, 'a', newline='') as ef:
                    writer = csv.DictWriter(ef, fieldnames=fieldnames)
                    if not exists_eval:
                        writer.writeheader()
                    ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                    writer.writerow({'timestamp': ts, 'epoch': 'eval-only', 'ade': ade_val, 'fde': fde_val, 'best_ade': best_ade_hist})
            except Exception as e:
                print(f"[WARN] Failed to write eval_log.csv (eval-only): {e}")

        # Emit compact json and exit
        result = {'mode':'eval-only','ade':ade_val,'fde':fde_val,'fpc':chosen_fpc}
        print('[EVAL-ONLY-RESULT]', json.dumps(result))
        sys.exit(0)

    # ---------------- Tee logger & helpers ----------------
    tee_file = None
    def tee_open(path):
        # 打开 train.out 追加写
        try:
            return open(path, 'a', buffering=1)
        except Exception:
            return None
    def tee_write(msg, end=''):
        # 统一写 stdout 与文件
        try:
            sys.stdout.write(msg + end)
        except Exception:
            pass
        if tee_file is not None:
            try:
                tee_file.write(msg + end)
            except Exception:
                pass
        # 避免缓存
        try:
            sys.stdout.flush()
        except Exception:
            pass
        if tee_file is not None:
            try:
                tee_file.flush()
            except Exception:
                pass
    def tee_print(*args, **kwargs):
        sep = kwargs.get('sep', ' ')
        end = kwargs.get('end', '\n')
        msg = sep.join(str(a) for a in args)
        tee_write(msg, end=end)
    # ------------------------------------------------------

    if train_data is not None:
        log_str = "\r\033[K {cur_batch:>"+str(len(str(batches)))+"}/"+str(batches)+" [{done}{remain}] -- time: {time}s - {comment}"    
        progress = 20/batches if batches > 20 else 1
        optimizer.zero_grad()
        # 权重塌缩监控状态
        collapse_patience = 3
        collapse_counter = 0
        collapse_threshold = 0.85  # 任一组件权重超过该值

    # Throughput & memory instrumentation accumulators (C)
    total_tokens = 0  # tokens ~ (batch_size * sequence_length); we'll approximate using observed horizon*agents
    total_steps = 0
    train_wall_time = 0.0
    gpu_peak_mem = 0

    for epoch in range(start_epoch+1, end_epoch+1):
        # Record tokens baseline for this epoch to compute per-epoch throughput later
        epoch_tokens_before = total_tokens
        # ---------------- Risk weight scheduling (linear warm-up) ----------------
        if hasattr(config, 'RISK_ENABLE') and getattr(config, 'RISK_ENABLE') and hasattr(model, 'risk_enable') and model.risk_enable:
            # warm-up over first 20% of total epochs
            warmup_epochs = max(1, int(0.2 * config.EPOCHS))
            target_w = getattr(config, 'RISK_WEIGHT', 0.0)
            if epoch <= warmup_epochs:
                cur_w = target_w * epoch / warmup_epochs
            else:
                cur_w = target_w
            # only update if changed to avoid extra graph noise
            if abs(cur_w - model.risk_weight) > 1e-6:
                model.risk_weight = cur_w
        # -------------------------------------------------------------------------
        ###############################################################################
        #####                                                                    ######
        ##### train                                                              ######
        #####                                                                    ######
        ###############################################################################
        losses = None
        ade, fde = None, None  # initialize to avoid NameError in summary when first 10-epoch block triggers before test
        if train_data is not None and epoch <= config.EPOCHS:
            # 初始化 train.out (仅在首个 epoch 或文件未打开时)
            if tee_file is None and settings.ckpt:
                os.makedirs(settings.ckpt, exist_ok=True)
                tee_path = os.path.join(settings.ckpt, 'train.out')
                tee_file = tee_open(tee_path)
                if tee_file:
                    tee_print(f"[TEE] Logging to {tee_path}")
            tee_print("Epoch {}/{}".format(epoch, config.EPOCHS))
            tic = time.time()
            set_rng_state(rng_state, settings.device)
            losses = {}
            last_loss_detail = None  # 保存最后一次 batch 的完整 loss 字典
            model.train()
            tee_write(log_str.format(
                cur_batch=0, done="", remain="."*int(batches*progress),
                time=round(time.time()-tic), comment=""))
            high_risk_records = []  # (risk_score_raw, batch_idx, sample_json)
            for batch, item in enumerate(train_data):
                if batch == 0:
                    torch.autograd.set_detect_anomaly(True)
                # item could be (x,y,neighbor) or (x,y,neighbor,map)
                if getattr(config, 'MAP_BCE_ENABLE', False) and len(item) == 4:
                    x,y,neighbor,map_meta = item
                    res = model(x,y,neighbor,map=map_meta)
                else:
                    res = model(*item)
                # res 结构: err, kl, L_adv_loss, avg_weighted_mse_loss, kinematic_loss, risk_score, risk_components, risk_score_raw, pred
                pred_tensor = res[-1]
                loss = model.loss(*res)
                last_loss_detail = loss
                loss["loss"].backward()
                optimizer.step()
                # ---- instrumentation update ----
                total_steps += 1
                # Estimate tokens: horizon * num_agents (pred tensor shape: T, N, 2)
                try:
                    pred_T, pred_N = pred_tensor.shape[0], pred_tensor.shape[1]
                    total_tokens += pred_T * pred_N
                except Exception:
                    pass
                if torch.cuda.is_available():
                    try:
                        gpu_peak_mem = max(gpu_peak_mem, torch.cuda.max_memory_allocated() // (1024*1024))
                    except Exception:
                        pass
                optimizer.zero_grad()
                if batch == 0:
                    torch.autograd.set_detect_anomaly(False)
                for k, v in loss.items():
                    if k == "weights":  # 跳过权重信息字典
                        continue
                    if k not in losses: 
                        losses[k] = v.item()
                    else:
                        losses[k] = (losses[k]*batch+v.item())/(batch+1)
                # 收集高风险候选（使用 risk_score_raw 若存在）
                if 'risk_score_raw' in loss:
                    try:
                        rs = float(loss['risk_score_raw'])
                        # 保存分数与 batch 以及对应预测（延迟决定是否写盘）
                        high_risk_records.append((rs, batch, pred_tensor.detach().cpu()))
                    except Exception:
                        pass
                tee_write(log_str.format(
                    cur_batch=batch+1, done="="*int((batch+1)*progress),
                    remain="."*(int(batches*progress)-int((batch+1)*progress)),
                    time=round(time.time()-tic),
                    comment=" - ".join(["{}: {:.4f}".format(k, v) for k, v in losses.items()])
                ))
                # optional early break for rapid experiments
                if settings.max_train_batches is not None and (batch+1) >= settings.max_train_batches:
                    break
            rng_state = get_rng_state(settings.device)
            train_wall_time += time.time() - tic
            tee_print()
            # --- Epoch-level autoscale CSV logging ---
            if hasattr(model, 'get_autoscale_state') and settings.ckpt:
                autoscale_row = model.get_autoscale_state()
                if autoscale_row:
                    # Always include risk_L_band in autoscale row (default 0.0); value overridden below if available
                    autoscale_row['risk_L_band'] = 0.0
                    # Always include map_bce_loss in autoscale row (default 0.0); value overridden below if available
                    autoscale_row['map_bce_loss'] = 0.0
                    # augment autoscale_row with epoch-averaged statistics first, then fallback to last batch
                    # 1) Risk scores (raw & scaled) — prefer epoch averages if available
                    for k_copy in ["risk_score_raw", "risk_score", "risk_scaled"]:
                        try:
                            if losses is not None and (k_copy in losses):
                                autoscale_row[k_copy] = float(losses[k_copy])
                            elif last_loss_detail is not None and (k_copy in last_loss_detail):
                                autoscale_row[k_copy] = float(last_loss_detail[k_copy])
                        except Exception:
                            pass
                    # 2) MAP_BCE loss — prefer epoch average if available, else last batch
                    try:
                        if losses is not None and ('map_bce_loss' in losses):
                            autoscale_row['map_bce_loss'] = float(losses['map_bce_loss'])
                        elif last_loss_detail is not None and ('map_bce_loss' in last_loss_detail):
                            mbl = last_loss_detail['map_bce_loss']
                            autoscale_row['map_bce_loss'] = float(mbl) if isinstance(mbl, (int, float)) else float(getattr(mbl, 'item', lambda: 0.0)())
                    except Exception:
                        pass
                    # 3) MAP_BCE debug stats — try epoch averages; fallback to last batch; also include map_channel0_frac
                    try:
                        for dbg_key in ['map_p_mean','map_p_min','map_p_max','map_oob_frac','map_allow_frac','map_channel0_frac']:
                            val = None
                            if losses is not None and (dbg_key in losses):
                                val = losses.get(dbg_key, None)
                            if val is None and last_loss_detail is not None and (dbg_key in last_loss_detail):
                                val = last_loss_detail.get(dbg_key, None)
                            if val is not None:
                                try:
                                    autoscale_row[dbg_key] = float(val)
                                except Exception:
                                    autoscale_row[dbg_key] = float(getattr(val, 'item', lambda: 0.0)())
                    except Exception:
                        pass
                        # Fallback: if loss dict missed MAP_BCE debug fields but model cached them, add here
                        try:
                            dbg = getattr(model, '_last_map_bce_debug', None)
                            if isinstance(dbg, dict):
                                for k, v in dbg.items():
                                    try:
                                        autoscale_row.setdefault(k, float(v))
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                    # 4) Risk band penalty — prefer epoch average if available, else last batch
                    try:
                        if losses is not None and ('risk_L_band' in losses):
                            autoscale_row['risk_L_band'] = float(losses['risk_L_band'])
                        elif last_loss_detail is not None and ('risk_L_band' in last_loss_detail):
                            rb = last_loss_detail['risk_L_band']
                            autoscale_row['risk_L_band'] = float(rb) if isinstance(rb, (int, float)) else float(getattr(rb, 'item', lambda: 0.0)())
                    except Exception:
                        pass
                        # log_sigma_penalty if present
                        if 'log_sigma_penalty' in last_loss_detail:
                            try:
                                autoscale_row['log_sigma_penalty'] = float(last_loss_detail['log_sigma_penalty'])
                            except Exception:
                                pass
                    # 附加熵正则系数（若有）
                    if hasattr(model, 'risk_compw_entropy_lambda'):
                        autoscale_row['compw_entropy_lambda'] = float(getattr(model, 'risk_compw_entropy_lambda'))
                    # 记录 compw_entropy 数值（若本 epoch 计算得到）
                    if last_loss_detail is not None and 'compw_entropy' in last_loss_detail:
                        try:
                            autoscale_row['compw_entropy'] = float(last_loss_detail['compw_entropy'])
                        except Exception:
                            pass
                    # 权重塌缩监控（仅 softmax 模式）
                    if getattr(model, 'risk_learn_component_weights', False) and getattr(model, 'risk_learn_component_norm', '') == 'softmax':
                        try:
                            compw = model.get_learned_component_weights()
                            if compw:
                                max_w = max(compw.values())
                                autoscale_row['compw_max'] = max_w
                                if max_w >= collapse_threshold:
                                    collapse_counter += 1
                                else:
                                    collapse_counter = 0
                                if collapse_counter >= collapse_patience:
                                    # 动态增大熵系数（乘法放大）
                                    if hasattr(model, 'risk_compw_entropy_lambda'):
                                        model.risk_compw_entropy_lambda *= 1.5
                                        autoscale_row['compw_entropy_lambda'] = float(model.risk_compw_entropy_lambda)
                                        collapse_counter = 0
                                        tee_print(f"[INFO] Increase entropy lambda to {model.risk_compw_entropy_lambda:.4g} due to weight collapse.")
                        except Exception as e:
                            tee_print(f"[WARN] Collapse monitor failed: {e}")
                    # 高风险样本统计（仅保存阈值, 不重复列过多）
                    if high_risk_records:
                        scores = torch.tensor([r[0] for r in high_risk_records])
                        q90 = float(torch.quantile(scores, 0.9)) if scores.numel() > 5 else float(scores.max())
                        autoscale_row['high_risk_q90'] = q90
                        # ---- 全局高风险样本输出路径 (根目录 High_Risk_Samples/<run_name>) ----
                        run_name = os.path.basename(settings.ckpt) if settings.ckpt else 'default_run'
                        global_hr_root = os.path.join('High_Risk_Samples', run_name)
                        hr_dir = os.path.join(global_hr_root, 'epoch_{:04d}'.format(epoch))
                        os.makedirs(hr_dir, exist_ok=True)
                        # 创建/更新一个顶层索引文件记录该 epoch 的统计摘要
                        try:
                            os.makedirs(global_hr_root, exist_ok=True)
                            index_path = os.path.join(global_hr_root, 'index.tsv')
                            with open(index_path, 'a') as fidx:
                                fidx.write(f"epoch\t{epoch}\tq90\t{q90:.6f}\tcount\t{len(high_risk_records)}\n")
                        except Exception as e:
                            tee_print(f"[WARN] Failed updating high risk index: {e}")
                        # 保存超过 q90 的 batch 索引到独立文本文件，并序列化预测轨迹
                        selected = [ (s,b,p) for s,b,p in high_risk_records if s >= q90 ]
                        out_path = os.path.join(hr_dir, 'selected.txt')
                        try:
                            with open(out_path, 'w') as wf:
                                for s,b,_ in selected:
                                    wf.write(f'batch={b}, risk_score_raw={s}\n')
                        except Exception as e:
                                tee_print(f"[WARN] Failed to save high risk samples: {e}")
                        # 额外保存轨迹张量（限制数量防止磁盘膨胀）
                        max_save = 10
                        for idx,(s,b,pred_tensor) in enumerate(sorted(selected, key=lambda x: -x[0])[:max_save]):
                            try:
                                torch.save({
                                    'epoch': epoch,
                                    'batch': b,
                                    'risk_score_raw': s,
                                    'pred': pred_tensor,  # 形状 (T,N,2)
                                }, os.path.join(hr_dir, f'batch_{b}_risk_{s:.4f}.pt'))
                            except Exception as e:
                                tee_print(f"[WARN] save pred tensor failed (batch={b}): {e}")
                    # ---- 附加每 epoch 吞吐率到 autoscale 记录 ----
                    epoch_tokens = total_tokens - epoch_tokens_before
                    if epoch_tokens > 0:
                        try:
                            autoscale_row['epoch_tokens'] = int(epoch_tokens)
                            elapsed_epoch = train_wall_time if 'elapsed_epoch' in locals() else None
                        except Exception:
                            pass
                    # 计算本 epoch 吞吐率（tokens/sec），使用本 epoch 训练时间
                    try:
                        epoch_time = time.time() - tic  # tic 仍指向本 epoch 开始时间
                        if epoch_time > 0 and (total_tokens - epoch_tokens_before) > 0:
                            autoscale_row['epoch_tokens_per_sec'] = (total_tokens - epoch_tokens_before) / epoch_time
                    except Exception:
                        pass
                    autoscale_path = os.path.join(settings.ckpt, 'autoscale_log.csv')
                    file_exists = os.path.exists(autoscale_path)
                    try:
                        # If file exists but missing new dynamic columns, we recreate a new file with merged columns.
                        if file_exists:
                            with open(autoscale_path, 'r') as rf:
                                header_line = rf.readline().strip()
                            existing_cols = header_line.split(',') if header_line else []
                            # Merge existing header with current keys to keep a stable superset schema
                            merged_cols = list(dict.fromkeys(existing_cols + ['epoch'] + list(autoscale_row.keys())))
                            need_rewrite = bool(set(merged_cols) - set(existing_cols)) or ('risk_L_band' not in existing_cols)
                            if need_rewrite:
                                # Need to rewrite entire file with new header
                                tmp_path = autoscale_path + '.tmp'
                                # read all rows
                                with open(autoscale_path, 'r') as rf:
                                    reader = csv.DictReader(rf)
                                    rows = list(reader)
                                # write new file with merged header
                                with open(tmp_path, 'w', newline='') as wf:
                                    writer = csv.DictWriter(wf, fieldnames=merged_cols)
                                    writer.writeheader()
                                    for old in rows:
                                        writer.writerow(old)
                                os.replace(tmp_path, autoscale_path)
                                existing_cols = merged_cols
                        else:
                            # First write: construct header from current autoscale keys (include epoch)
                            existing_cols = ['epoch'] + list(autoscale_row.keys())
                        # append new row using the established header (existing_cols)
                        with open(autoscale_path, 'a', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=existing_cols)
                            if not file_exists:
                                writer.writeheader()
                            # Fill missing columns with '' to avoid KeyError
                            row = {k: autoscale_row.get(k, '') for k in existing_cols}
                            row['epoch'] = epoch
                            writer.writerow(row)
                    except Exception as e:
                        tee_print(f"[WARN] Failed to write autoscale CSV: {e}")

            # ---------- 每10个epoch汇总输出 ----------
            if settings.ckpt and (epoch % 10 == 0):
                summary_path = os.path.join(settings.ckpt, 'summary_10ep.txt')
                try:
                    compw_max = None
                    compw_entropy = None
                    risk_score = losses.get('risk_score', None)
                    if last_loss_detail is not None and 'compw_entropy' in last_loss_detail:
                        compw_entropy = float(last_loss_detail['compw_entropy'])
                    if getattr(model, 'risk_learn_component_weights', False):
                        try:
                            cw = model.get_learned_component_weights()
                            if cw:
                                compw_max = max(cw.values())
                        except Exception:
                            pass
                    tokens_per_sec = None
                    if 'epoch_tokens_per_sec' in locals():
                        tokens_per_sec = autoscale_row.get('epoch_tokens_per_sec') if 'autoscale_row' in locals() else None
                    loss_val = losses.get('loss', None)
                    try:
                        loss_fmt = f"{loss_val:.4f}" if isinstance(loss_val, (int,float)) else str(loss_val)
                    except Exception:
                        loss_fmt = str(loss_val)
                    ade_fmt_val = float(ade) if isinstance(ade,(int,float)) and not (isinstance(ade,float) and math.isnan(ade)) else (float(ade.item()) if (hasattr(ade,'item')) else None)
                    fde_fmt_val = float(fde) if isinstance(fde,(int,float)) and not (isinstance(fde,float) and math.isnan(fde)) else (float(fde.item()) if (hasattr(fde,'item')) else None)
                    ade_fmt = ade_fmt_val if ade_fmt_val is not None else 'NaN'
                    fde_fmt = fde_fmt_val if fde_fmt_val is not None else 'NaN'
                    best_ade_fmt = None
                    if 'ade_best' in locals() and isinstance(ade_best,(int,float)) and not math.isnan(ade_best):
                        try:
                            best_ade_fmt = f"{ade_best:.4f}"
                        except Exception:
                            best_ade_fmt = ade_best
                    else:
                        best_ade_fmt = 'NaN'
                    with open(summary_path, 'a') as sf:
                        sf.write(f"epoch={epoch}\tloss={loss_fmt}\tade={ade_fmt}\tfde={fde_fmt}\tbest_ade={best_ade_fmt}\tbest_ade_epoch={ade_best_epoch}\trisk_score={risk_score}\tcompw_max={compw_max}\tcompw_entropy={compw_entropy}\ttokens_per_sec={tokens_per_sec}\n")
                    tee_print(f"[SUMMARY-10EP] epoch={epoch} appended to summary_10ep.txt")
                except Exception as e:
                    tee_print(f"[WARN] Failed writing 10-epoch summary: {e}")

        ###############################################################################
        #####                                                                    ######
        ##### test                                                               ######
        #####                                                                    ######
        ###############################################################################
        ade, fde = float('nan'), float('nan')
        perform_test = (train_data is None or epoch >= config.TEST_SINCE) and test_data is not None
        if perform_test:
            tee_print(f"[EVAL] epoch={epoch} starting evaluation ...")
            if not settings.no_fpc and not settings.fpc_finetune and losses is None and fpc_best > 1:
                fpc = fpc_best
            else:
                fpc = 1
            ade, fde = test(model, fpc)
            try:
                ade_dbg = float(ade.item()) if hasattr(ade, 'item') else float(ade)
                fde_dbg = float(fde.item()) if hasattr(fde, 'item') else float(fde)
                tee_print(f"[EVAL] epoch={epoch} done: ADE={ade_dbg:.4f}, FDE={fde_dbg:.4f}, fpc={fpc}")
            except Exception:
                tee_print(f"[EVAL] epoch={epoch} done: ADE/FDE computed")

        ###############################################################################
        #####                                                                    ######
        ##### log                                                                ######
        #####                                                                    ######
        ###############################################################################
        if losses is not None and settings.ckpt:
            if logger is not None:
                for k, v in losses.items():
                    logger.add_scalar("train/{}".format(k), v, epoch)
                if 'log_sigma_penalty' in last_loss_detail:
                    try:
                        logger.add_scalar("risk/log_sigma_penalty", float(last_loss_detail['log_sigma_penalty']), epoch)
                    except Exception:
                        pass
                if hasattr(model, 'risk_weight') and model.risk_enable:
                    logger.add_scalar("train/risk_weight_sched", model.risk_weight, epoch)
                    # Log learned component weights if enabled
                    if getattr(model, 'risk_learn_component_weights', False):
                        try:
                            compw = model.get_learned_component_weights()
                            for ck, cv in compw.items():
                                logger.add_scalar(f"risk/compw_{ck}", cv, epoch)
                        except Exception as e:
                            tee_print(f"[WARN] TensorBoard logging comp weights failed: {e}")
                if perform_test:
                    logger.add_scalar("eval/ADE", ade, epoch)
                    logger.add_scalar("eval/FDE", fde, epoch)
            state = dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                ade=ade, fde=fde, epoch=epoch, rng_state=rng_state
            )
            torch.save(state, ckpt)
            # 评估日志: 每次 perform_test 追加到 eval_log.csv
            if perform_test:
                try:
                    eval_log_path = os.path.join(settings.ckpt, 'eval_log.csv')
                    exists_eval = os.path.exists(eval_log_path)
                    with open(eval_log_path, 'a', newline='') as ef:
                        # 若文件已存在但列缺失（增加 best_ade 后），需要重写
                        existing_header = None
                        if exists_eval:
                            try:
                                with open(eval_log_path, 'r') as rf:
                                    first_line = rf.readline().strip()
                                    existing_header = first_line.split(',') if first_line else []
                            except Exception:
                                existing_header = None
                        fieldnames = ['timestamp','epoch','ade','fde','best_ade']
                        if exists_eval and existing_header and set(fieldnames) - set(existing_header):
                            # 需要把旧内容读取并重写为新列（旧列 best_ade 补空）
                            try:
                                with open(eval_log_path, 'r') as rf:
                                    reader = csv.DictReader(rf)
                                    rows = list(reader)
                                with open(eval_log_path, 'w', newline='') as wf:
                                    writer_full = csv.DictWriter(wf, fieldnames=fieldnames)
                                    writer_full.writeheader()
                                    for row in rows:
                                        row['best_ade'] = row.get('best_ade','')
                                        writer_full.writerow(row)
                                exists_eval = True
                            except Exception as e2:
                                tee_print(f"[WARN] Failed to rewrite eval_log.csv with new header: {e2}")
                        # 重新以追加模式打开
                        ef.close()
                        ef = open(eval_log_path, 'a', newline='')
                        # Single writer initialization (removed duplicate assignment)
                        writer = csv.DictWriter(ef, fieldnames=fieldnames)
                        if not exists_eval:
                            writer.writeheader()
                        ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                        # 统一数值转换，兼容 Tensor/float
                        def _to_float(x):
                            try:
                                return float(x.item()) if hasattr(x, 'item') else (float(x) if x is not None else None)
                            except Exception:
                                return None
                        ade_val = _to_float(ade)
                        fde_val = _to_float(fde)
                        best_ade_val = _to_float(ade_best)
                        writer.writerow({'timestamp': ts, 'epoch': epoch, 'ade': ade_val if ade_val is not None else '', 'fde': fde_val if fde_val is not None else '', 'best_ade': best_ade_val if best_ade_val is not None else ''})
                except Exception as e:
                    tee_print(f"[WARN] Failed to write eval_log.csv: {e}")
            # 保存 ckpt-best：比较前将张量转为 Python float
            if perform_test:
                try:
                    ade_num = float(ade.item()) if hasattr(ade, 'item') else (float(ade) if isinstance(ade, (int, float)) else None)
                    ade_best_num = float(ade_best.item()) if hasattr(ade_best, 'item') else (float(ade_best) if isinstance(ade_best, (int, float)) else None)
                    fde_num = float(fde.item()) if hasattr(fde, 'item') else (float(fde) if isinstance(fde, (int, float)) else None)
                except Exception:
                    ade_num, ade_best_num, fde_num = None, None, None
                # 若尚无 ckpt-best，则首次评估直接落盘（前提是有有效ADE）
                if not os.path.exists(ckpt_best) and (ade_num is not None) and (fde_num is not None):
                    ade_best = ade_num
                    fde_best = fde_num
                    ade_best_epoch = epoch
                    best_state = dict(
                        model=state["model"],
                        ade=ade_best, fde=fde_best, epoch=epoch
                    )
                    torch.save(best_state, ckpt_best)
                    tee_print(f"[CKPT] First eval at epoch {epoch}, saved initial ckpt-best (ADE={ade_best:.4f})")
                if (ade_num is not None) and (ade_best_num is not None) and (ade_num < ade_best_num):
                    ade_best = ade_num
                    fde_best = fde_num if fde_num is not None else fde_best
                    ade_best_epoch = epoch
                    best_state = dict(
                        model=state["model"],
                        ade=ade_best, fde=fde_best, epoch=epoch
                    )
                    torch.save(best_state, ckpt_best)
                    tee_print(f"[CKPT] New best ADE {ade_best:.4f} at epoch {epoch}, saved ckpt-best")

        # ------- Fallback: 如果本轮未进行训练(losses is None)但完成了评估，也需要记录评估并尝试更新 ckpt-best -------
        if losses is None and perform_test and settings.ckpt:
            # 写 eval_log.csv（与上面一致的列），避免重复写入（因为 losses is None，此处不会和上面重复）
            try:
                eval_log_path = os.path.join(settings.ckpt, 'eval_log.csv')
                exists_eval = os.path.exists(eval_log_path)
                fieldnames = ['timestamp','epoch','ade','fde','best_ade']
                # 若文件已存在但列缺失（增加 best_ade 后），尝试补齐
                if exists_eval:
                    try:
                        with open(eval_log_path, 'r') as rf:
                            first_line = rf.readline().strip()
                            existing_header = first_line.split(',') if first_line else []
                        if set(fieldnames) - set(existing_header):
                            with open(eval_log_path, 'r') as rf:
                                reader = csv.DictReader(rf)
                                rows = list(reader)
                            with open(eval_log_path, 'w', newline='') as wf:
                                writer_full = csv.DictWriter(wf, fieldnames=fieldnames)
                                writer_full.writeheader()
                                for row in rows:
                                    row['best_ade'] = row.get('best_ade','')
                                    writer_full.writerow(row)
                            exists_eval = True
                    except Exception:
                        pass
                with open(eval_log_path, 'a', newline='') as ef:
                    writer = csv.DictWriter(ef, fieldnames=fieldnames)
                    if not exists_eval:
                        writer.writeheader()
                    def _to_float(x):
                        try:
                            return float(x.item()) if hasattr(x, 'item') else (float(x) if x is not None else None)
                        except Exception:
                            return None
                    ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                    ade_val = _to_float(ade)
                    fde_val = _to_float(fde)
                    best_ade_val = _to_float(ade_best)
                    writer.writerow({'timestamp': ts, 'epoch': epoch, 'ade': ade_val if ade_val is not None else '', 'fde': fde_val if fde_val is not None else '', 'best_ade': best_ade_val if best_ade_val is not None else ''})
            except Exception as e:
                tee_print(f"[WARN] Failed to write eval_log.csv (no-train path): {e}")

            # 比较并尝试保存 ckpt-best
            try:
                ade_num = float(ade.item()) if hasattr(ade, 'item') else (float(ade) if isinstance(ade, (int, float)) else None)
                ade_best_num = float(ade_best.item()) if hasattr(ade_best, 'item') else (float(ade_best) if isinstance(ade_best, (int, float)) else None)
                fde_num = float(fde.item()) if hasattr(fde, 'item') else (float(fde) if isinstance(fde, (int, float)) else None)
            except Exception:
                ade_num, ade_best_num, fde_num = None, None, None
            if (ade_num is not None) and (ade_best_num is not None) and (ade_num < ade_best_num):
                ade_best = ade_num
                fde_best = fde_num if fde_num is not None else fde_best
                ade_best_epoch = epoch
                best_state = dict(
                    model=model.state_dict(),
                    ade=ade_best, fde=fde_best, epoch=epoch
                )
                torch.save(best_state, ckpt_best)
                tee_print(f"[CKPT] New best ADE {ade_best:.4f} at epoch {epoch} (no-train path), saved ckpt-best")

    if not settings.skip_fpc_search and (settings.fpc_finetune or losses is not None):
        try:
            rng_state = get_rng_state(settings.device)
        except Exception:
            pass
        tee_print(f"[FPC] starting search over {list(getattr(config,'FPC_SEARCH_RANGE',[1]))} ...")
        # FPC finetune if it is specified or after training
        precision = 2
        trunc = lambda v: np.trunc(v*10**precision)/10**precision
        ade_, fde_, fpc_ = [], [], []
        for fpc in config.FPC_SEARCH_RANGE:
            ade, fde = test(model, fpc)
            ade_.append(trunc(ade.item()))
            fde_.append(trunc(fde.item()))
            fpc_.append(fpc)
        i = np.argmin(np.add(ade_, fde_))
        ade, fde, fpc = ade_[i], fde_[i], fpc_[i]
        if settings.ckpt:
            ckpt_best = os.path.join(settings.ckpt, "ckpt-best")
            if os.path.exists(ckpt_best):
                state_dict = torch.load(ckpt_best, map_location=settings.device)
                state_dict["ade_fpc"] = ade
                state_dict["fde_fpc"] = fde
                state_dict["fpc"] = fpc
                torch.save(state_dict, ckpt_best)
        tee_print(" ADE: {:.2f}; FDE: {:.2f} ({})".format(
            ade, fde, "FPC: {}".format(fpc) if fpc > 1 else "w/o FPC", 
        ))

    # ---------------- Write A/B metrics JSON (B + C) ----------------
    try:
        label = os.environ.get('AB_LABEL', None)
        if label and settings.ckpt:
            out_path = os.path.join(settings.ckpt, f"ab_metrics_{label}.json")
            # If ade/fde not defined (e.g., no test), set to None
            summary = {
                'label': label,
                'ade': float(ade) if 'ade' in locals() and np.isscalar(ade) else None,
                'fde': float(fde) if 'fde' in locals() and np.isscalar(fde) else None,
                'total_steps': total_steps,
                'total_tokens': total_tokens,
                'tokens_per_sec': (total_tokens / train_wall_time) if train_wall_time > 0 else None,
                'gpu_peak_mem_mb': gpu_peak_mem if gpu_peak_mem else None,
                'train_wall_time_sec': train_wall_time,
                'config_path': cfg_path,
                'epochs_run': end_epoch - start_epoch,
            }
            with open(out_path, 'w') as jf:
                json.dump(summary, jf, indent=2)
            tee_print(f"[INFO] Wrote A/B metrics to {out_path}")
    except Exception as e:
        tee_print(f"[WARN] Failed to write A/B metrics JSON: {e}")
    # ---------------------------------------------------------------











