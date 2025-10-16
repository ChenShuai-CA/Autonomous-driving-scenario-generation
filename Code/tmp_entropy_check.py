import importlib
import torch
from social_vae import SocialVAE
from data import Dataloader

spec = importlib.util.spec_from_file_location("config","config/Interaction.py")
cfg = importlib.util.module_from_spec(spec); spec.loader.exec_module(cfg)

kwargs = dict(batch_first=False, frameskip=1, ob_horizon=cfg.OB_HORIZON, pred_horizon=cfg.PRED_HORIZON,
              device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), seed=1)
paths = ['Code/data/Interation/DR_USA_Intersection_EP1/train']
train_dataset = Dataloader(paths, **kwargs, batch_size=cfg.BATCH_SIZE, shuffle=True, batches_per_epoch=1)
loader = torch.utils.data.DataLoader(train_dataset, collate_fn=train_dataset.collate_fn, batch_sampler=train_dataset.batch_sampler)

risk_params = dict(enable=cfg.RISK_ENABLE, weight=cfg.RISK_WEIGHT, risk_global_scale=cfg.RISK_GLOBAL_SCALE,
                   component_weights=cfg.RISK_COMPONENT_WEIGHTS, learn_component_weights=cfg.RISK_LEARN_COMPONENT_WEIGHTS,
                   learn_component_norm=cfg.RISK_LEARN_COMPONENT_NORM, beta=cfg.RISK_MIN_DIST_BETA, ttc_tau=cfg.RISK_TTC_TAU)

device = kwargs['device']
model = SocialVAE(horizon=cfg.PRED_HORIZON, ob_radius=cfg.OB_RADIUS, hidden_dim=cfg.RNN_HIDDEN_DIM, risk_params=risk_params).to(device)
model.risk_compw_entropy_lambda = cfg.RISK_COMPW_ENTROPY_LAMBDA
model.train()

(batch,) = list(loader)
res = model(*batch)
loss_dict = model.loss(*res)
print('Entropy key present:', 'compw_entropy' in loss_dict)
print('compw_entropy:', loss_dict.get('compw_entropy'))
print('entropy lambda:', loss_dict.get('compw_entropy_lambda'))
print('component weights (prob):', model.get_learned_component_weights())