import os, torch
import importlib.util
import sys

# Usage: python Code/minimal_entropy_check.py --config config/Interaction.py

def load_config(path):
    spec = importlib.util.spec_from_file_location("config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main(cfg_path):
    cfg = load_config(cfg_path)
    from social_vae import SocialVAE
    risk_params = dict(
        enable=getattr(cfg, 'RISK_ENABLE', False),
        weight=getattr(cfg, 'RISK_WEIGHT', 0.0),
        risk_global_scale=getattr(cfg, 'RISK_GLOBAL_SCALE', 1.0),
        component_weights=getattr(cfg, 'RISK_COMPONENT_WEIGHTS', {}),
        learn_component_weights=getattr(cfg, 'RISK_LEARN_COMPONENT_WEIGHTS', False),
        learn_component_norm=getattr(cfg, 'RISK_LEARN_COMPONENT_NORM', 'none'),
        beta=getattr(cfg, 'RISK_MIN_DIST_BETA', 2.0),
        ttc_tau=getattr(cfg, 'RISK_TTC_TAU', 1.5),
        # PET
        pet_dist_th=getattr(cfg, 'RISK_PET_DIST_TH', 2.0),
        pet_alpha=getattr(cfg, 'RISK_PET_ALPHA', 8.0),
        pet_beta=getattr(cfg, 'RISK_PET_BETA', 0.7),
        pet_gamma=getattr(cfg, 'RISK_PET_GAMMA', 0.0),
        pet_continuous=getattr(cfg, 'RISK_PET_CONTINUOUS', True),
        pet_time_temp=getattr(cfg, 'RISK_PET_TIME_TEMP', 4.0),
        enable_pet=getattr(cfg, 'RISK_ENABLE_PET', True),
        # Overlap
        ov_r_self=getattr(cfg, 'RISK_OV_R_SELF', 0.5),
        ov_r_neigh=getattr(cfg, 'RISK_OV_R_NEIGH', 0.5),
        ov_margin=getattr(cfg, 'RISK_OV_MARGIN', 0.3),
        ov_k=getattr(cfg, 'RISK_OV_K', 15.0),
        ov_time_temp=getattr(cfg, 'RISK_OV_TIME_TEMP', 4.0),
        ov_neigh_temp=getattr(cfg, 'RISK_OV_NEIGH_TEMP', 3.0),
        enable_overlap=getattr(cfg, 'RISK_ENABLE_OVERLAP', True),
        ov_use_obb=getattr(cfg, 'RISK_OV_USE_OBB', False),
        ov_self_length=getattr(cfg, 'RISK_OV_SELF_LENGTH', 4.5),
        ov_self_width=getattr(cfg, 'RISK_OV_SELF_WIDTH', 1.8),
        ov_neigh_length=getattr(cfg, 'RISK_OV_NEIGH_LENGTH', 4.5),
        ov_neigh_width=getattr(cfg, 'RISK_OV_NEIGH_WIDTH', 1.8),
        ov_min_speed=getattr(cfg, 'RISK_OV_MIN_SPEED', 1e-3),
        ov_axis_beta=getattr(cfg, 'RISK_OV_AXIS_BETA', 12.0),
        ov_debug=getattr(cfg, 'RISK_OV_DEBUG', False),
        risk_multi_samples=getattr(cfg, 'RISK_MULTI_SAMPLES', 1),
        risk_multi_temp=getattr(cfg, 'RISK_MULTI_TEMP', 4.0),
        compw_entropy_lambda=getattr(cfg, 'RISK_COMPW_ENTROPY_LAMBDA', None),
    )
    model = SocialVAE(horizon=cfg.PRED_HORIZON, ob_radius=cfg.OB_RADIUS, hidden_dim=cfg.RNN_HIDDEN_DIM, risk_params=risk_params)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    torch.manual_seed(0)
    # Synthetic batch: shapes follow data loader (L_ob x N x 6) etc aggregated in forward pass expectations
    ob_hor = cfg.OB_HORIZON
    pred_hor = cfg.PRED_HORIZON
    total_h = ob_hor + pred_hor
    N = 4  # number of agents
    Nn = 3 # number of neighbors per agent (simplified)
    # x: L_ob x N x 6
    hist = torch.randn(ob_hor+1, N, 6, device=device)
    # future(y): pred_hor x N x 2
    future = torch.randn(pred_hor, N, 2, device=device)
    # neighbor: (ob_hor+pred_hor) x N x Nn x 6
    neighbor = torch.randn(total_h+1, N, Nn, 6, device=device)
    # 提高风险: 让邻居前若干时间步位置接近自车最后观测位置 (潜在重叠/距离很小)
    ego_last = hist[-1, :, :2].unsqueeze(1)  # (N,1,2)
    for t in range(min(total_h+1, ob_hor+3)):
        # 设置邻居的 x,y 接近 ego_last 并速度很小
        neighbor[t, :, :, 0:2] = ego_last.repeat(1, Nn, 1) + 0.01*torch.randn_like(neighbor[t, :, :, 0:2])
        neighbor[t, :, :, 2:4] = 0.0  # vx, vy
        neighbor[t, :, :, 4:6] = 0.0  # ax, ay
    # Forward: 训练时传入 (x, y, neighbor) 其中 y 仅用于 teacher forcing? 查看主训循环: res = model(*item) where item=(x,y,neighbor)
    out = model(hist, future, neighbor, n_predictions=1)
    loss_dict = model.loss(*out)
    keys = [k for k in loss_dict.keys() if 'compw' in k or 'risk' in k or k in ('loss','rec')]
    print("== Selected loss keys ==")
    for k in sorted(keys):
        v = loss_dict[k]
        if torch.is_tensor(v):
            v = v.detach().cpu().item()
        print(f"{k}: {v}")
    if 'compw_entropy' not in loss_dict:
        print('[Warn] compw_entropy 未出现，可能 risk_learn_component_weights 未启用或 entropy_lambda=0')
    # Explicitly show learned component weights
    if getattr(model, 'risk_learn_component_weights', False):
        print("== Learned component weights ==")
        try:
            print(model.get_learned_component_weights())
        except Exception as e:
            print("Failed to get component weights:", e)

if __name__ == '__main__':
    cfg_path = 'config/Interaction.py'
    if len(sys.argv) > 2 and sys.argv[1] == '--config':
        cfg_path = sys.argv[2]
    main(cfg_path)
