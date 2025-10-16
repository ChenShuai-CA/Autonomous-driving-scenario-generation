import argparse
import importlib.util
import torch
import numpy as np
from social_vae import SocialVAE
from data import Dataloader
from utils import ADE_FDE, seed

"""Simple risk analysis script.
Runs model on test data, collects risk component values (requires risk enabled),
computes distribution stats (mean, std, p50, p90, p95) for each component and ADE/FDE.
"""

def load_config(path):
    spec = importlib.util.spec_from_file_location("config", path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def percentile(t: torch.Tensor, q: float):
    k = (q/100.0)*(t.numel()-1)
    f = torch.floor(k).long()
    c = torch.ceil(k).long()
    if f == c:
        return torch.kthvalue(t, f+1).values
    lower = torch.kthvalue(t, f+1).values
    upper = torch.kthvalue(t, c+1).values
    return lower + (k-f)*(upper-lower)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--device', default=None)
    parser.add_argument('--limit-batches', type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    seed(getattr(cfg, 'SEED', 1))

    # dataloader (test only)
    kwargs = dict(batch_first=False, frameskip=1, ob_horizon=cfg.OB_HORIZON, pred_horizon=cfg.PRED_HORIZON,
                  device=device, seed=getattr(cfg, 'SEED', 1))
    inclusive = None
    if getattr(cfg, 'INCLUSIVE_GROUPS', None) is not None:
        inclusive = [cfg.INCLUSIVE_GROUPS]
    test_dataset = Dataloader(["Code/data/Interation/DR_USA_Intersection_EP1/train"],  # fallback path
                              **kwargs, inclusive_groups=inclusive,
                              batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset,
            collate_fn=test_dataset.collate_fn,
            batch_sampler=test_dataset.batch_sampler)

    # load model
    risk_params = {}
    if hasattr(cfg, 'RISK_ENABLE'):
        risk_params = dict(
            enable=getattr(cfg, 'RISK_ENABLE', False),
            weight=getattr(cfg, 'RISK_WEIGHT', 0.0),
            component_weights=getattr(cfg, 'RISK_COMPONENT_WEIGHTS', {}),
            beta=getattr(cfg, 'RISK_MIN_DIST_BETA', 2.0),
            ttc_tau=getattr(cfg, 'RISK_TTC_TAU', 1.5),
            pet_dist_th=getattr(cfg, 'RISK_PET_DIST_TH', 2.0),
            pet_alpha=getattr(cfg, 'RISK_PET_ALPHA', 8.0),
            pet_beta=getattr(cfg, 'RISK_PET_BETA', 0.7),
            pet_gamma=getattr(cfg, 'RISK_PET_GAMMA', 2.0),
            enable_pet=getattr(cfg, 'RISK_ENABLE_PET', True),
            ov_r_self=getattr(cfg, 'RISK_OV_R_SELF', 0.5),
            ov_r_neigh=getattr(cfg, 'RISK_OV_R_NEIGH', 0.5),
            ov_margin=getattr(cfg, 'RISK_OV_MARGIN', 0.3),
            ov_k=getattr(cfg, 'RISK_OV_K', 15.0),
            ov_time_temp=getattr(cfg, 'RISK_OV_TIME_TEMP', 4.0),
            ov_neigh_temp=getattr(cfg, 'RISK_OV_NEIGH_TEMP', 3.0),
            enable_overlap=getattr(cfg, 'RISK_ENABLE_OVERLAP', True),
            risk_multi_samples=getattr(cfg, 'RISK_MULTI_SAMPLES', 1),
            risk_multi_temp=getattr(cfg, 'RISK_MULTI_TEMP', 4.0),
        )
    model = SocialVAE(horizon=cfg.PRED_HORIZON, ob_radius=cfg.OB_RADIUS, hidden_dim=cfg.RNN_HIDDEN_DIM, risk_params=risk_params)
    model.to(device)
    ckpt = torch.load(f"{args.ckpt}/ckpt-best", map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    comps_store = {}
    ADE_all = []
    FDE_all = []
    with torch.no_grad():
        for b_idx, (x, y, neighbor) in enumerate(test_loader):
            if args.limit_batches and b_idx >= args.limit_batches:
                break
            # produce predictions (single deterministic + risk score inside model.learn is only for training)
            # so we manually compute risk over one forward pass using training path hack:
            # call learn-like pass by faking training mode? Simpler: run deterministic forward and then reuse risk_metrics.
            pred_samples = model(x, neighbor, n_predictions=getattr(cfg, 'PRED_SAMPLES', 0))  # shape depends
            if pred_samples.dim() == 4:  # n_samples x T x N x 2
                # evaluate ADE/FDE best-of-n
                ade, fde = ADE_FDE(pred_samples, y)
                ade = torch.min(ade, dim=0)[0]
                fde = torch.min(fde, dim=0)[0]
                pred = pred_samples[0]  # take first for risk evaluation
            else:
                ade, fde = ADE_FDE(pred_samples.unsqueeze(0), y)
                ade = ade[0]
                fde = fde[0]
                pred = pred_samples
            ADE_all.append(ade.detach())
            FDE_all.append(fde.detach())
            # compute risk components externally
            try:
                from risk_metrics import compute_risk_score
                risk_out = compute_risk_score(pred - x[-1,...,:2].unsqueeze(0), x[-1], neighbor,
                                              model.risk_component_weights,
                                              beta=model.risk_beta, ttc_tau=model.risk_ttc_tau,
                                              **model.risk_pet_params, **model.risk_overlap_params)
                for k, v in risk_out.items():
                    if k == 'risk_score':
                        continue
                    comps_store.setdefault(k, []).append(v.detach())
            except Exception:
                pass
    # aggregate stats
    def stackcat(lst):
        return torch.stack(lst) if isinstance(lst[0], torch.Tensor) else torch.tensor(lst)
    ade_cat = torch.cat(ADE_all)
    fde_cat = torch.cat(FDE_all)
    print("ADE mean {:.4f} FDE mean {:.4f}".format(ade_cat.mean().item(), fde_cat.mean().item()))
    for name, vals in comps_store.items():
        t = stackcat(vals).flatten()
        stats = {
            'mean': t.mean().item(),
            'std': t.std().item(),
            'p50': percentile(t, 50).item(),
            'p90': percentile(t, 90).item(),
            'p95': percentile(t, 95).item(),
        }
        print(f"{name}: " + ", ".join(f"{k}={v:.5f}" for k,v in stats.items()))

if __name__ == '__main__':
    main()
