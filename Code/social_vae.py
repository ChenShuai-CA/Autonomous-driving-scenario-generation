import torch
import torch
import numpy as np
from typing import Optional, Dict

# Optional Mamba sequence module (graceful fallback if missing)
try:  # pragma: no cover
    from mamba_layers import build_mamba_or_none
except Exception:  # pragma: no cover
    build_mamba_or_none = None

try:
    from risk_metrics import compute_risk_score
except Exception:
    compute_risk_score = None


class SocialVAE(torch.nn.Module):
    @staticmethod
    def _safe_normal(loc: torch.Tensor, std: torch.Tensor) -> torch.distributions.Normal:
        """Construct a Normal distribution robustly under AMP:
        - Cast params to float32
        - Replace NaN/Inf with finite defaults
        - Clamp std to a minimum epsilon
        - Build distribution with autocast disabled to avoid dtype promotion issues
        """
        eps = 1e-6
        # Move to float32 and sanitize
        loc32 = torch.nan_to_num(loc.float(), nan=0.0, posinf=0.0, neginf=0.0)
        std32 = torch.nan_to_num(std.float(), nan=1.0, posinf=1.0, neginf=1.0).clamp_min(eps)
        # Ensure distribution parameters are created outside autocast (FP32)
        try:
            import torch.cuda.amp as amp  # noqa: F401
            with torch.cuda.amp.autocast(enabled=False):
                return torch.distributions.Normal(loc32, std32)
        except Exception:
            return torch.distributions.Normal(loc32, std32)

    class DecoderZH(torch.nn.Module):
        def __init__(self, z_dim, hidden_dim, embed_dim, output_dim):
            super().__init__()
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(z_dim+hidden_dim, embed_dim),
                torch.nn.ReLU6(),
                torch.nn.Linear(embed_dim, embed_dim),
                torch.nn.ReLU6()
            )
            self.mu = torch.nn.Linear(embed_dim, output_dim)

        def forward(self, z, h):
            xy = self.embed(torch.cat((z, h), -1))
            loc = self.mu(xy)
            return loc


    class P_Z(torch.nn.Module):
        def __init__(self, hidden_dim_fy, embed_dim, z_dim):
            super().__init__()
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim_fy, embed_dim),
                torch.nn.ReLU6(),
                torch.nn.Linear(embed_dim, embed_dim),
                torch.nn.ReLU6()
            )
            self.mu = torch.nn.Linear(embed_dim, z_dim)
            self.std = torch.nn.Sequential(
                torch.nn.Linear(embed_dim, z_dim),
                torch.nn.Softplus()
            )
            
        def forward(self, x):
            x = self.embed(x)
            loc = self.mu(x)
            std = self.std(x)
            return SocialVAE._safe_normal(loc, std)
   

    class Q_Z(torch.nn.Module):
        def __init__(self, hidden_dim_fy, hidden_dim_by, embed_dim, z_dim):
            super().__init__()
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim_fy+hidden_dim_by, embed_dim),
                torch.nn.ReLU6(),
                torch.nn.Linear(embed_dim, embed_dim),
                torch.nn.ReLU6()
            )
            self.mu = torch.nn.Linear(embed_dim, z_dim)
            self.std = torch.nn.Sequential(
                torch.nn.Linear(embed_dim, z_dim),
                torch.nn.Softplus()
            )

        def forward(self, x, y):
            xy = self.embed(torch.cat((x, y), -1))
            loc = self.mu(xy)
            std = self.std(xy)
            return SocialVAE._safe_normal(loc, std)


    class EmbedZD(torch.nn.Module):
        def __init__(self, z_dim, d_dim, output_dim):
            super().__init__()
            self.embed_zd = torch.nn.Sequential(
                torch.nn.Linear(z_dim+d_dim, output_dim),
                torch.nn.ReLU6(),
                torch.nn.Linear(output_dim, output_dim)
            )
        def forward(self, z, d):
            code = torch.cat((z, d), -1)
            return self.embed_zd(code)


    def __init__(self, horizon, ob_radius=2, hidden_dim=256, *, risk_params: Optional[Dict]=None):
        super().__init__()
        self.ob_radius = ob_radius
        self.horizon = horizon
        hidden_dim_fx = hidden_dim
        hidden_dim_fy = hidden_dim
        hidden_dim_by = 256
        feature_dim = 256
        self_embed_dim = 128
        neighbor_embed_dim = 128
        z_dim = 32
        d_dim = 2

        self.q_z = SocialVAE.Q_Z(hidden_dim_fy, hidden_dim_by, hidden_dim_fy, z_dim)
        self.p_z = SocialVAE.P_Z(hidden_dim_fy, hidden_dim_fy, z_dim)
        self.dec = SocialVAE.DecoderZH(z_dim, hidden_dim_fy, hidden_dim_fy, d_dim)
        
        self.embed_s = torch.nn.Sequential(
            torch.nn.Linear(4, 64),             # v, a
            torch.nn.ReLU6(),
            torch.nn.Linear(64, self_embed_dim),
        )
        self.embed_n = torch.nn.Sequential(
            torch.nn.Linear(4, 64),             # dp, dv
            torch.nn.ReLU6(),
            torch.nn.Linear(64, neighbor_embed_dim),
            torch.nn.ReLU6(),
            torch.nn.Linear(neighbor_embed_dim, neighbor_embed_dim)
        )
        self.embed_k = torch.nn.Sequential(
            torch.nn.Linear(3, feature_dim),    # dist, bear angle, mpd
            torch.nn.ReLU6(),
            torch.nn.Linear(feature_dim, feature_dim),
            torch.nn.ReLU6(),
            torch.nn.Linear(feature_dim, feature_dim)
        )
        self.embed_q = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim_fx, feature_dim),
            torch.nn.ReLU6(),
            torch.nn.Linear(feature_dim, feature_dim),
            torch.nn.ReLU6(),
            torch.nn.Linear(feature_dim, feature_dim)
        )
        self.attention_nonlinearity = torch.nn.LeakyReLU(0.2)
        # --- Multi-head neighbor attention placeholders (optional) ---
        self.use_multihead_attn = False
        self.mha_heads = 1
        self.mha_q = None
        self.mha_k = None
        self.mha_v = None
        self.mha_out = None
        self.mha_drop = None

        self.rnn_fx = torch.nn.GRU(self_embed_dim+neighbor_embed_dim, hidden_dim_fx)
        self.rnn_fx_init = torch.nn.Sequential(
            torch.nn.Linear(2, hidden_dim_fx), # dp
            torch.nn.ReLU6(),
            torch.nn.Linear(hidden_dim_fx, hidden_dim_fx*self.rnn_fx.num_layers),
            torch.nn.ReLU6(),
            torch.nn.Linear(hidden_dim_fx*self.rnn_fx.num_layers, hidden_dim_fx*self.rnn_fx.num_layers),
        )
        self.rnn_by = torch.nn.GRU(self_embed_dim+neighbor_embed_dim, hidden_dim_by)

        self.embed_zd = SocialVAE.EmbedZD(z_dim, d_dim, z_dim)
        self.rnn_fy = torch.nn.GRU(z_dim, hidden_dim_fy)
        self.rnn_fy_init = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim_fx, hidden_dim_fy*self.rnn_fy.num_layers),
            torch.nn.ReLU6(),
            torch.nn.Linear(hidden_dim_fy*self.rnn_fy.num_layers, hidden_dim_fy*self.rnn_fy.num_layers)
        )
        # risk params
        self.risk_enable = False
        self.risk_weight = 0.0
        self.risk_global_scale = 1.0  # new global scale multiplier
        self.risk_component_weights = {}
        self.risk_beta = 2.0
        self.risk_ttc_tau = 1.5
        # new component params (defaults mirror config)
        self.risk_pet_params = {
            'pet_dist_th': 2.0,
            'pet_alpha': 8.0,
            'pet_beta': 0.7,
            'pet_gamma': 2.0,
            'pet_continuous': True,
            'pet_time_temp': 4.0,
            'enable_pet': True,
        }
        self.risk_overlap_params = {
            'ov_r_self': 0.5,
            'ov_r_neigh': 0.5,
            'ov_margin': 0.3,
            'ov_k': 15.0,
            'ov_time_temp': 4.0,
            'ov_neigh_temp': 3.0,
            'enable_overlap': True,
            # OBB extensions (defaults; can be overridden via risk_params)
            'ov_use_obb': False,
            'ov_self_length': 4.5,
            'ov_self_width': 1.8,
            'ov_neigh_length': 4.5,
            'ov_neigh_width': 1.8,
            'ov_min_speed': 1e-3,
            'ov_axis_beta': 12.0,
            'ov_debug': False,
        }

        # ---------------- Adaptive risk scaling (EMA-based target fraction) ----------------
        # These fields enable optional automatic adjustment of risk_global_scale so that
        #   risk_weight * risk_global_scale * E[risk_score] ≈ target_frac * base_loss
        # where base_loss = (rec + wmse + kl + adv + kin) under current weighting.
        # Tunable hyper-parameters (can later be surfaced via risk_params if needed):
        self.risk_autoscale_enable = True  # master switch
        self.risk_autoscale_target_frac = 0.01  # aim: risk term ~1% of base loss
        self.risk_autoscale_alpha = 0.01  # EMA smoothing factor
        self._risk_ema_base = None  # exponential moving average of base loss (detached)
        self._risk_ema_score = None # exponential moving average of risk_score (detached)
        self._risk_autoscale_beta = 0.1  # smoothing for updating risk_global_scale itself
        self.risk_autoscale_min = 1e-6   # clamp lower bound to avoid collapse
        self.risk_autoscale_max = 1e9    # very generous upper bound
        # -------------------------------------------------------------------------------

        # ---------------- Component normalization (per-component EMA) ----------------
        # Maintains running mean/var to z-score normalize each component before combining.
        # This helps avoid one component (e.g., overlap) dominating purely by scale.
        self.risk_compnorm_enable = True
        self.risk_compnorm_alpha = 0.01
        self._comp_ema_mean = {}
        self._comp_ema_var = {}
        # After normalization we recompute a normalized aggregate risk_score (mean of z's)
        # The raw aggregate (pre-normalization) is still logged as risk_score_raw.
        # -------------------------------------------------------------------------------

        # ---------------- Learnable component weights (optional) ----------------
        # If enabled (risk_learn_component_weights=True) we replace static
        # self.risk_component_weights dict with a ParameterDict whose positive
        # weights are produced via softplus (or softmax for normalized variant).
        # Supported normalization modes:
        #   'none'    : independent positive weights (softplus)
        #   'softmax' : weights sum to 1 (softmax on unconstrained params)
        self.risk_learn_component_weights = False
        self.risk_learn_component_norm = 'none'  # or 'softmax'
        self._risk_comp_params = torch.nn.ParameterDict()
        # -----------------------------------------------------------------------

        # ------------- Uncertainty (log-sigma) weighting prototype -------------
        # If enabled, replace linear (or softmax) weighting with learned log_sigmas
        # for each component: aggregate = mean_i( comp_i * exp(-log_sigma_i) ) + penalty_w * sum(log_sigma_i)
        self.risk_use_log_sigma = False
        self.risk_log_sigma_penalty_w = 1.0
        self._risk_log_sigmas = torch.nn.ParameterDict()  # filled lazily when enabling
        # -----------------------------------------------------------------------

        # defer applying external risk params until all supporting members created
        if risk_params:
            self.set_risk_params(**risk_params)

        # --------- Mamba integration placeholders (enabled later via enable_mamba()) ---------
        self.mamba_encoder = None  # replaces rnn_fx loop when not None
        self.mamba_decoder = None  # replaces rnn_fy step loop when not None
        self.use_mamba_encoder = False
        self.use_mamba_decoder = False
        # --------------------------------------------------------------------------------------

    def set_risk_params(self, enable: bool=False, weight: float=0.0, component_weights: Optional[Dict[str,float]]=None,
                        beta: float=2.0, ttc_tau: float=1.5, risk_global_scale: float = 1.0,
                        learn_component_weights: bool=False, learn_component_norm: str='none', compw_entropy_lambda: float = None,
                        use_log_sigma: bool=False, log_sigma_penalty_w: float=1.0, **kwargs):
        self.risk_enable = enable and (compute_risk_score is not None)
        self.risk_weight = float(weight)
        self.risk_global_scale = float(risk_global_scale)
        if component_weights:
            self.risk_component_weights = component_weights
        # configure learnable weights
        self.risk_learn_component_weights = bool(learn_component_weights)
        self.risk_learn_component_norm = str(learn_component_norm).lower()
        # initialize ParameterDict if needed
        if self.risk_learn_component_weights and len(self._risk_comp_params) == 0 and self.risk_component_weights:
            for k, v in self.risk_component_weights.items():
                # store raw unconstrained parameter; start from log(exp(w)-1) for softplus invert to keep init exact
                v0 = float(v)
                # avoid negative / zero (softplus inverse defined for >0); if user gives 0 still fine using small epsilon
                if v0 <= 0:
                    v0 = 1e-3
                # softplus inverse: log(exp(w) - 1); guard for tiny
                inv = torch.log(torch.expm1(torch.tensor(v0)) + 1e-8)
                self._risk_comp_params[k] = torch.nn.Parameter(inv)
        # configure log-sigma prototype
        self.risk_use_log_sigma = bool(use_log_sigma)
        self.risk_log_sigma_penalty_w = float(log_sigma_penalty_w)
        if self.risk_use_log_sigma and len(self._risk_log_sigmas) == 0 and self.risk_component_weights:
            for k in self.risk_component_weights.keys():
                self._risk_log_sigmas[k] = torch.nn.Parameter(torch.zeros(()))
        self.risk_beta = beta
        self.risk_ttc_tau = ttc_tau
        # store entropy regularization coefficient if provided
        if compw_entropy_lambda is not None:
            self.risk_compw_entropy_lambda = float(compw_entropy_lambda)
        # update additional params if provided
        pet_keys = [k for k in kwargs.keys() if k.startswith('pet_') or k in ('enable_pet',)]
        if pet_keys:
            for k in pet_keys:
                if k == 'enable_pet':
                    self.risk_pet_params['enable_pet'] = kwargs[k]
                elif k in self.risk_pet_params:
                    self.risk_pet_params[k] = kwargs[k]
        ov_keys = [k for k in kwargs.keys() if k.startswith('ov_') or k in ('enable_overlap',)]
        if ov_keys:
            for k in ov_keys:
                if k == 'enable_overlap':
                    self.risk_overlap_params['enable_overlap'] = kwargs[k]
                elif k in self.risk_overlap_params:
                    self.risk_overlap_params[k] = kwargs[k]
        return self

    # --- Learnable risk component weights utilities ---
    def _current_component_weights(self) -> Dict[str, torch.Tensor]:
        """Return a dict of component weights (torch.Tensors) to feed into risk aggregation.
        If learning enabled, produces positive weights via softplus / softmax; else returns static floats as tensors.
        """
        if not self.risk_learn_component_weights or len(self._risk_comp_params) == 0:
            # return static weights as tensors (device aware) so future ops are consistent
            return {k: torch.as_tensor(v, device=next(self.parameters()).device) for k, v in self.risk_component_weights.items()}
        # gather raw params
        keys = list(self._risk_comp_params.keys())
        raw = torch.stack([self._risk_comp_params[k] for k in keys])  # unconstrained
        if self.risk_learn_component_norm == 'softmax':
            w_vec = torch.softmax(raw, dim=0)
        else:
            w_vec = torch.nn.functional.softplus(raw)  # positive
        return {k: w for k, w in zip(keys, w_vec)}

    def get_learned_component_weights(self) -> Dict[str, float]:
        """Return current (post-transform) component weights as plain floats for logging."""
        if not self.risk_learn_component_weights:
            return {}
        with torch.no_grad():
            comp = self._current_component_weights()
            return {k: float(v.detach()) for k, v in comp.items()}

    def attention(self, q, k, mask):
        # q: N x d
        # k: N x Nn x d
        # mask: N x Nn
        e = (k @ q.unsqueeze(-1)).squeeze(-1)           # N x Nn
        e = self.attention_nonlinearity(e)              # N x Nn
        e[~mask] = -float("inf")
        att = torch.nn.functional.softmax(e, dim=-1)    # N x Nn
        return att.nan_to_num()

    # ---- Multi-head aggregation (if enabled) ----
    def _multihead_aggregate(self, q_vec, k_mat, v_mat, mask_row):
        """q_vec: (N,d) ; k_mat: (N,Nn,d_k_orig) ; v_mat: (N,Nn,d_v_orig); mask_row: (N,Nn)"""
        H = self.mha_heads
        d_model = self.mha_q.out_features
        head_dim = d_model // H
        # project
        q = self.mha_q(q_vec)                          # N,d
        k = self.mha_k(k_mat)                          # N,Nn,d
        v = self.mha_v(v_mat)                          # N,Nn,d
        # reshape to heads
        q = q.view(q.size(0), H, head_dim)             # N,H,hd
        k = k.view(k.size(0), k.size(1), H, head_dim).transpose(1,2)  # N,H,Nn,hd
        v = v.view(v.size(0), v.size(1), H, head_dim).transpose(1,2)  # N,H,Nn,hd
        scores = (k @ q.unsqueeze(-1)).squeeze(-1) / (head_dim ** 0.5)  # N,H,Nn
        # Expand mask for heads (broadcast-friendly) then masked_fill
        mask_h = mask_row.unsqueeze(1).expand(-1, H, -1)  # N,H,Nn
        # Use dtype-aware large negative (FP16-safe) instead of -inf to keep softmax gradients finite
        if scores.dtype == torch.float16:
            neg_large = -1e4
        else:
            neg_large = -1e9
        scores = scores.masked_fill(~mask_h, neg_large)
        # For rows that are fully masked, set scores to 0 to avoid NaN in backward
        all_masked = (~mask_h).all(dim=-1)  # N,H
        if all_masked.any():
            scores[all_masked] = 0.0
        att = torch.softmax(scores, dim=-1)
        # Zero out attention where mask is false (ensures no leakage on fully masked rows)
        att = att * mask_h.to(att.dtype)
        # Replace NaNs (can happen if a row is fully masked) with zeros
        att = att.nan_to_num(0.0)
        ctx = (att.unsqueeze(-2) @ v).squeeze(-2)      # N,H,hd
        ctx = ctx.reshape(ctx.size(0), H*head_dim)     # N,d
        out = self.mha_out(ctx)
        if self.mha_drop is not None:
            out = self.mha_drop(out)
        return out

    def enc(self, x, neighbor, *, y=None):
        """
        x: Input trajectory positions. Shape_x: (L1+1) x N x 6, where L1 is the length of the first trajectory, 
        N is the number of trajectories, and 6 represents the trajectory information 
        (e.g., x-coordinate, y-coordinate, velocity components).

        y: Additional trajectory information (optional). Shape_y: L2 x N x 2 

        neighbor: Neighbor information for each trajectory. Shape: (L1+L2+1) x N x Nn x 6, 
        where L2 is the length of the second trajectory, 
        Nn is the number of neighbors, and 6 represents neighbor information.
        """

        with torch.no_grad():
            L1 = x.size(0)-1        #Calculates the length of the first trajectory (L1) based on the size of the input x. 
            N = neighbor.size(1)    #Extracts the number of trajectories (N) from the size of the neighbor tensor.
            Nn = neighbor.size(2)   #Extracts the number of neighbors (Nn) from the size of the neighbor tensor.
            state = x               #Initializes the state variable with the input x.

            x = state[...,:2]                       # (L1+1) x N x 2
            if y is not None:
                L2 = y.size(0)
                x = torch.cat((x, y), 0)            # (L+1) x N x 2
            else:
                L2 = 0
            
            v = x[1:] - x[:-1]                      # L x N x 2
            a = v[1:] - v[:-1]                      # (L-1) x N x 2
            a = torch.cat((state[1:2,...,4:6], a))  # L x N x 2
            """
            Concatenates the initial acceleration (extracted from the state tensor) 
            with the computed acceleration along the time dimension. 
            This ensures that the acceleration tensor has the same length as the velocity tensor. 
            The resulting tensor is of the same shape 
            """

            neighbor_x = neighbor[...,:2]           # (L+1) x N x Nn x 2
            neighbor_v = neighbor[1:,...,2:4]       # L x N x Nn x 2

            dp = neighbor_x - x.unsqueeze(-2)       # (L+1) x N x Nn x 2 Computes the relative position
            dv = neighbor_v - v.unsqueeze(-2)       # L x N x Nn x 2 V between the target trajectory and its neighbors by taking the difference between their velocities. 

            # social features
            dist = dp.norm(dim=-1)                          # (L+1) x N x Nn Computes the Euclidean distance between the target trajectory and its neighbors for each time step. 
            mask = dist <= self.ob_radius
            dp0, mask0 = dp[0], mask[0]
            dp, mask = dp[1:], mask[1:]
            dist = dist[1:]
            dot_dp_v = (dp @ v.unsqueeze(-1)).squeeze(-1)   # L x N x Nn
            bearing = dot_dp_v / (dist*v.norm(dim=-1).unsqueeze(-1)) # L x N x Nn
            bearing = bearing.nan_to_num(0, 0, 0)
            dot_dp_dv = (dp.unsqueeze(-2) @ dv.unsqueeze(-1)).view(dp.size(0),N,Nn)
            tau = -dot_dp_dv / dv.norm(dim=-1)              # L x N x Nn
            tau = tau.nan_to_num(0, 0, 0).clip(0, 7)
            mpd = (dp + tau.unsqueeze(-1)*dv).norm(dim=-1)  # L x N x Nn
            features = torch.stack((dist, bearing, mpd), -1)# L x N x Nn x 3

        k = self.embed_k(features)                          # L x N x Nn x d
        s = self.embed_s(torch.cat((v, a), -1))
        n = self.embed_n(torch.cat((dp, dv), -1))           # L x N x Nn x ...

        h = self.rnn_fx_init(dp0)                           # N x Nn x d
        h = (mask0.unsqueeze(-1) * h).sum(-2)               # N x d
        h = h.view(N, -1, self.rnn_fx.num_layers)
        h = h.permute(2, 0, 1).contiguous()

        if self.mamba_encoder is None:
            for t in range(L1):
                q_vec = self.embed_q(h[-1])
                if self.use_multihead_attn and self.mha_q is not None:
                    neigh = self._multihead_aggregate(q_vec, k[t], n[t], mask[t])  # N,d_model
                else:
                    att = self.attention(q_vec, k[t], mask[t])
                    neigh = (att.unsqueeze(-2) @ n[t]).squeeze(-2)
                x_step = torch.cat((neigh, s[t]), -1).unsqueeze(0)
                _, h = self.rnn_fx(x_step, h)
        else:
            # Build the whole observed sequence then single pass through Mamba
            seq_frames = []
            for t in range(L1):
                q_vec = self.embed_q(h[-1])
                if self.use_multihead_attn and self.mha_q is not None:
                    neigh = self._multihead_aggregate(q_vec, k[t], n[t], mask[t])
                else:
                    att = self.attention(q_vec, k[t], mask[t])
                    neigh = (att.unsqueeze(-2) @ n[t]).squeeze(-2)
                seq_frames.append(torch.cat((neigh, s[t]), -1))
            seq_tensor = torch.stack(seq_frames, dim=0)  # (L1,N,D)
            # project if needed
            if hasattr(self, 'mamba_enc_in') and self.mamba_enc_in is not None:
                seq_tensor = self.mamba_enc_in(seq_tensor)
            # StackedMamba expects (T,B,D)
            out, h_all = self.mamba_encoder(seq_tensor)
            h = h_all  # (layers,N,D)
        x = h[-1]
        if y is None: return x
        mask_t = mask[L1:L1+L2].unsqueeze(-1)               # L2 x N x Nn x 1
        n_t = n[L1:L1+L2]                                   # L2 x N x Nn x d
        n_t = (mask_t * n_t).sum(-2)                        # L2 x N x d
        s_t = s[L1:L2+L2]
        x_t = torch.cat((n_t, s_t), -1)
        x_t = torch.flip(x_t, (0,))
        b, _ = self.rnn_by(x_t)                             # L2 x N x n_layer*d
        if self.rnn_by.num_layers > 1:
            b = b[...,-b.size(-1)//self.rnn_by.num_layers:]
        b = torch.flip(b, (0,))
        return x, b


    def forward(self, *args, **kwargs):
        # We put the training and testing forward function together in order to support 
        #   DistributedDataParallel better.
        # training:
        #   x: L x N x 6
        #   neighbor: L x N x Nn x 6, padding at Nn dimension using large value (e.g. 1e9)
        #   output: args to self.loss()
        # testing:
        #   x: L x N x 6
        #   neighbor: L x N x Nn x 6, padding at Nn dimension using large value (e.g. 1e9)
        #   n_predictions: int, number of predictions
        #   output: n_predictions x horizon x N x 2, for n_predictions > 0
        #         horizon x N x 2, n_predictions=0 for deterministic prediction

        # flatten only if respective GRUs exist (not replaced by Mamba)
        if self.mamba_encoder is None:
            self.rnn_fx.flatten_parameters()
        if self.mamba_decoder is None:
            self.rnn_fy.flatten_parameters()
        if self.training:
            if self.mamba_encoder is None:
                self.rnn_by.flatten_parameters()
            args = iter(args)
            x = kwargs["x"] if "x" in kwargs else next(args)
            y = kwargs["y"] if "y" in kwargs else next(args)
            neighbor = kwargs["neighbor"] if "neighbor" in kwargs else next(args)
            # propagate optional semantic map meta for MAP_BCE
            map_meta = kwargs["map"] if "map" in kwargs else None
            return self.learn(x, y, neighbor, map=map_meta)

        args = iter(args)
        x = kwargs["x"] if "x" in kwargs else next(args)
        neighbor = kwargs["neighbor"] if "neighbor" in kwargs else next(args)
        try:
            n_predictions = kwargs["n_predictions"] if "n_predictions" in kwargs else next(args)
        except:
            n_predictions = 0

        stochastic = n_predictions > 0
        if neighbor is None:
            neighbor_shape = [_ for _ in x.shape]
            neighbor_shape.insert(-1, 0)
            neighbor = torch.empty(neighbor_shape, dtype=x.dtype, device=x.device)
        C = x.dim()
        if C < 3:
            x = x.unsqueeze(1)
            neighbor = neighbor.unsqueeze(1)
            if y is not None: y = y.unsqueeze(1)
        N = x.size(1)

        neighbor = neighbor[:x.size(0)]
        h = self.enc(x, neighbor)
           
        h = self.rnn_fy_init(h)
        h = h.view(N, -1, self.rnn_fy.num_layers)
        h = h.permute(2, 0, 1)
        if stochastic: h = h.repeat(1, n_predictions, 1)
        h = h.contiguous()
        
        D = []
        for t in range(self.horizon):
            p_z = self.p_z(h[-1])
            z = p_z.sample() if stochastic else p_z.mean
            d = self.dec(z, h[-1])
            D.append(d)
            if t == self.horizon - 1: break
            zd = self.embed_zd(z, d)
            if self.mamba_decoder is None:
                _, h = self.rnn_fy(zd.unsqueeze(0), h)
            else:
                if hasattr(self, 'mamba_dec_in') and self.mamba_dec_in is not None:
                    zd_proj = self.mamba_dec_in(zd)
                else:
                    zd_proj = zd
                step_out, h_new = self.mamba_decoder(zd_proj.unsqueeze(0))
                h = h_new

        d = torch.stack(D)
        pred = torch.cumsum(d, 0)
        if stochastic:
            pred = pred.view(pred.size(0), n_predictions, -1, pred.size(-1)).permute(1, 0, 2, 3)
        pred = pred + x[-1,...,:2]
        if C < 3: pred = pred.squeeze(1)
        return pred
    
    def Adv_loss(self, neighbor, pred, x, err):
        """
        实现论文中的对抗损失：
        L_adv = ∑_{i=1}^N ∑_{j=1}^M exp(-√||e_{i,j}||_2) · (1/∑_{k=1}^M exp(-√||e_{i,k}||_2))
        其中 e_{i,j} = ŷ_i - n_{i,j}
        
        当前实现与论文公式不一致，需要修正。
        """
        if neighbor.shape[2] == 0:  # 没有邻居
            return torch.tensor(0.0, device=pred.device)
            
        # 计算ego预测轨迹的绝对位置
        ego_pred_abs = pred + x[-1,...,:2].unsqueeze(0)  # [T, N, 2]
        
        # 获取对应时间步的邻居位置
        neighbor_pos = neighbor[-pred.shape[0]:, :, :, 0:2]  # [T, N, Nn, 2]
        
        # 计算误差向量 e_{i,j} = ŷ_i - n_{i,j}
        # ego_pred_abs: [T, N, 2] -> [T, N, 1, 2]
        # neighbor_pos: [T, N, Nn, 2]
        ego_expanded = ego_pred_abs.unsqueeze(2)  # [T, N, 1, 2]
        e_ij = ego_expanded - neighbor_pos  # [T, N, Nn, 2]
        
        # 计算欧几里得距离 ||e_{i,j}||_2
        distances = torch.norm(e_ij, dim=-1)  # [T, N, Nn]
        
        # 计算 exp(-√||e_{i,j}||_2)
        exp_neg_sqrt_dist = torch.exp(-torch.sqrt(distances + 1e-8))  # [T, N, Nn]
        
        # 计算归一化权重 (softmax across neighbors)
        # 分母：∑_{k=1}^M exp(-√||e_{i,k}||_2)
        sum_exp = torch.sum(exp_neg_sqrt_dist, dim=2, keepdim=True)  # [T, N, 1]
        weights = exp_neg_sqrt_dist / (sum_exp + 1e-8)  # [T, N, Nn]
        
        # 计算最终的对抗损失
        # L_adv = ∑_{i=1}^N ∑_{j=1}^M exp(-√||e_{i,j}||_2) · weight_{i,j}
        weighted_exp = exp_neg_sqrt_dist * weights  # [T, N, Nn]
        L_adv_loss = torch.sum(weighted_exp)
        
        return L_adv_loss
    
    def kinematic_loss(self, pred):
        """
        计算运动学损失：
        1) 平滑正则（原有）：加速度变化 & 航向角变化
        2) 类型化门限（新增）：速度/加速度/jerk/曲率/摩擦圆 软约束（阈值从 config/模型属性读取）
        
        Parameters:
        pred (torch.Tensor): 预测的位移序列 [T, N, 2]
        
        Returns:
        torch.Tensor: 综合运动学损失
        """
        if pred.shape[0] < 2:
            return torch.tensor(0.0, device=pred.device)
            
        # 计算速度 (位移的一阶差分)
        velocity = pred[1:] - pred[:-1]  # [T-1, N, 2]
        
        # 计算加速度 (速度的一阶差分)
        if velocity.shape[0] < 2:
            acceleration = None
            acceleration_loss = torch.tensor(0.0, device=pred.device)
        else:
            acceleration = velocity[1:] - velocity[:-1]  # [T-2, N, 2]
            acceleration_loss = torch.norm(acceleration, dim=-1).mean()
        
        # 计算角速度变化
        if velocity.shape[0] < 2:
            angular_loss = torch.tensor(0.0, device=pred.device)
        else:
            # 计算每个时间步的角度
            angles = torch.atan2(velocity[..., 1], velocity[..., 0])  # [T-1, N]
            if angles.shape[0] < 2:
                angular_loss = torch.tensor(0.0, device=pred.device)
            else:
                # 角度差分 (注意角度的周期性)
                angle_diff = angles[1:] - angles[:-1]  # [T-2, N]
                # 处理角度的周期性 (-π, π]
                angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
                angular_loss = torch.abs(angle_diff).mean()

        base_smooth = acceleration_loss + angular_loss

        # ---------------- 新增：类型化门限软约束 ----------------
        # 使用 softplus(k*(x - x_max))/k 作为可微“超限惩罚”
        def sp(x):
            k = getattr(self, 'KIN_SOFTPLUS_K', 10.0)
            return torch.nn.functional.softplus(k * x) / k

        # 阈值读取（按类型，目前默认当作车辆处理；后续可引入 agent_type 掩码实现多类型门限）
        vmax = torch.as_tensor(getattr(self, 'KIN_VMAX_VEHICLE', 13.9), device=pred.device, dtype=pred.dtype)  # m/s
        amax = torch.as_tensor(getattr(self, 'KIN_AMAX_VEHICLE', 3.0), device=pred.device, dtype=pred.dtype)   # m/s^2
        jmax = torch.as_tensor(getattr(self, 'KIN_JMAX_VEHICLE', 2.0), device=pred.device, dtype=pred.dtype)   # m/s^3
        kmax = torch.as_tensor(getattr(self, 'KIN_KAPPA_MAX_VEHICLE', 0.3), device=pred.device, dtype=pred.dtype) # 1/m
        mu   = torch.as_tensor(getattr(self, 'KIN_MU_VEHICLE', 0.5), device=pred.device, dtype=pred.dtype)
        g = torch.as_tensor(9.81, device=pred.device, dtype=pred.dtype)

        # 速度/加速度/jerk
        v_mag = torch.norm(velocity, dim=-1)  # [T-1, N]
        L_v = sp(v_mag - vmax).mean()

        if acceleration is not None:
            a_mag = torch.norm(acceleration, dim=-1)  # [T-2, N]
            L_a = sp(a_mag - amax).mean()
            # jerk
            if acceleration.shape[0] >= 2:
                jerk = acceleration[1:] - acceleration[:-1]  # [T-3, N, 2]
                j_mag = torch.norm(jerk, dim=-1)
                L_j = sp(j_mag - jmax).mean()
            else:
                L_j = torch.tensor(0.0, device=pred.device)
        else:
            L_a = torch.tensor(0.0, device=pred.device)
            L_j = torch.tensor(0.0, device=pred.device)

        # 曲率 κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2) ；用 v 近似 x',y'，a 近似 x'',y''
        if acceleration is not None:
            vx, vy = velocity[1:,...,0], velocity[1:,...,1]    # 对齐加速度时间索引
            ax, ay = acceleration[...,0], acceleration[...,1]
            denom = (vx*vx + vy*vy).clamp_min(1e-6).pow(1.5)
            kappa = (vx*ay - vy*ax).abs() / denom  # [T-2, N]
            L_kappa = sp(kappa - kmax).mean()
            # 侧向加速度 a_lat ≈ v^2 * κ
            v_for_lat = torch.norm(velocity[1:], dim=-1)  # [T-2, N]
            a_lat = (v_for_lat*v_for_lat) * kappa
            # 摩擦圆近似：仅约束侧向 a_lat ≤ μ g（纵向已由 a_max 约束）
            L_fric = sp(a_lat - mu * g).mean()
        else:
            L_kappa = torch.tensor(0.0, device=pred.device)
            L_fric = torch.tensor(0.0, device=pred.device)

        # 权重（可从 config 注入到模型属性）
        w_v    = getattr(self, 'LOSS_W_KIN_V', 1.0)
        w_a    = getattr(self, 'LOSS_W_KIN_A', 1.0)
        w_j    = getattr(self, 'LOSS_W_KIN_J', 0.5)
        w_kap  = getattr(self, 'LOSS_W_KIN_KAPPA', 0.5)
        w_fric = getattr(self, 'LOSS_W_KIN_FRIC', 0.5)
        w_lim  = getattr(self, 'LOSS_W_KIN_LIMITS', 0.1)  # 总门限损失的缩放

        kin_limits = (w_v*L_v + w_a*L_a + w_j*L_j + w_kap*L_kappa + w_fric*L_fric)

        # 将门限损失合并进总体 kinematic_loss，保持对外接口不变
        total_kin = base_smooth + w_lim * kin_limits

        # 暴露分量用于日志（通过 loss() 输出）
        self._last_kin_components = {
            'kin_base_smooth': base_smooth.detach(),
            'kin_L_v': L_v.detach(), 'kin_L_a': L_a.detach(), 'kin_L_j': L_j.detach(),
            'kin_L_kappa': L_kappa.detach(), 'kin_L_fric': L_fric.detach(),
            'kin_limits': kin_limits.detach(),
        }
        return total_kin

    def weighted_mse_loss(self, pred, err, alpha=0.1):
        """
        Compute the weighted mean squared error.
    
        Parameters:
        alpha (float): The weight scaling factor.
        pred (torch.Tensor): The tensor of predictions.
        err (torch.Tensor): The tensor of errors (difference between predictions and ground truth).
    
        Returns:
        torch.Tensor: The computed weighted mean squared error.
        """
        weights = torch.exp(-alpha * torch.arange(pred.shape[0], device=err.device))
        weights = weights.view(pred.shape[0], 1, 1)
        weighted_err = weights * err
        avg_weighted_mse_loss = weighted_err.sum() / weights.sum()
    
        return avg_weighted_mse_loss

    def learn(self, x, y, neighbor=None, map=None):
        C = x.dim()
        if C < 3:
            x = x.unsqueeze(1)
            neighbor = neighbor.unsqueeze(1)
            if y is not None:
                y = y.unsqueeze(1)
        N = x.size(1)
        if y.size(0) != self.horizon:
            print("[Warn] Unmatched sequence length in inference and generative model. ({} vs {})".format(y.size(0), self.horizon))

        # Encode observations and target context
        h_enc, b = self.enc(x, neighbor, y=y)
        h = self.rnn_fy_init(h_enc)
        h = h.view(N, -1, self.rnn_fy.num_layers)
        h = h.permute(2, 0, 1).contiguous()

        # optional multi-sample risk aggregation
        multi_samples = getattr(self, 'risk_multi_samples', 1) if hasattr(self, 'risk_multi_samples') else 1
        multi_temp = getattr(self, 'risk_multi_temp', 4.0) if hasattr(self, 'risk_multi_temp') else 4.0
        collect_preds = []  # list of (T,N,2)
        # We'll compute reconstruction / KL only for the first sample to save compute if multi_samples>1
        first_err = None
        first_kl = None
        first_L_adv = None
        first_weighted_mse = None
        first_kinematic = None

        # reset last map bce placeholder per batch
        self._last_map_bce = torch.zeros((), device=x.device)

        for sample_idx in range(multi_samples):
            h_s = h.clone()  # starting hidden state per sample
            P, Q, D, Z = [], [], [], []
            for t in range(self.horizon):
                p_z = self.p_z(h_s[-1])
                q_z = self.q_z(h_s[-1], b[t])
                z = q_z.rsample()      # latent variable
                d = self.dec(z, h_s[-1])  # reconstructed delta
                P.append(p_z); Q.append(q_z); D.append(d); Z.append(z)
                if t == self.horizon - 1:
                    break
                zd = self.embed_zd(z, d)
                if self.mamba_decoder is None:
                    _, h_s = self.rnn_fy(zd.unsqueeze(0), h_s)
                else:
                    if hasattr(self, 'mamba_dec_in') and self.mamba_dec_in is not None:
                        zd_proj = self.mamba_dec_in(zd)
                    else:
                        zd_proj = zd
                    _, h_new = self.mamba_decoder(zd_proj.unsqueeze(0))
                    h_s = h_new
            d_stack = torch.stack(D)  # (T,N,2)
            pred_sample = torch.cumsum(d_stack, 0)
            collect_preds.append(pred_sample)

            if sample_idx == 0:
                # compute reconstruction-related only once (relative coords)
                with torch.no_grad():
                    y_rel = y - x[-1, ..., :2].unsqueeze(0)
                err = (pred_sample - y_rel).square()
                kl_list = []
                for p, q, z in zip(P, Q, Z):
                    kl_list.append(q.log_prob(z) - p.log_prob(z))
                kl = torch.stack(kl_list)
                L_adv_loss = self.Adv_loss(neighbor, pred_sample, x, err)
                avg_weighted_mse_loss = self.weighted_mse_loss(pred_sample, err, alpha=0.1)
                kinematic_loss = self.kinematic_loss(pred_sample)
                # Optional map BCE on first sample (absolute coords)
                try:
                    if map is not None and getattr(self, 'MAP_BCE_ENABLE', getattr(self, 'map_bce_enable', False)):
                        pred_abs = pred_sample + x[-1, ..., :2].unsqueeze(0)  # (T,N,2)
                        self._last_map_bce = self._compute_map_bce(pred_abs, map)
                    else:
                        self._last_map_bce = torch.zeros((), device=x.device)
                except Exception:
                    self._last_map_bce = torch.zeros((), device=x.device)

                first_err = err
                first_kl = kl
                first_L_adv = L_adv_loss
                first_weighted_mse = avg_weighted_mse_loss
                first_kinematic = kinematic_loss

        # base losses from first sample
        err = first_err
        kl = first_kl
        L_adv_loss = first_L_adv
        avg_weighted_mse_loss = first_weighted_mse
        kinematic_loss = first_kinematic

        # risk aggregation over samples
        risk_score = torch.zeros((), device=x.device)
        risk_components = {}
        risk_score_raw = None
        if self.risk_enable and neighbor is not None and compute_risk_score is not None:
            sample_scores = []
            sample_comp_list = []
            try:
                for pred_sample in collect_preds:
                    if self.risk_use_log_sigma:
                        dummy_w = {k: 1.0 for k in self.risk_component_weights.keys()}
                        risk_dict = compute_risk_score(
                            pred_sample, x[-1], neighbor, dummy_w,
                            beta=self.risk_beta, ttc_tau=self.risk_ttc_tau,
                            **self.risk_pet_params, **self.risk_overlap_params
                        )
                    else:
                        comp_w = self._current_component_weights() if hasattr(self, '_current_component_weights') else self.risk_component_weights
                        risk_dict = compute_risk_score(
                            pred_sample, x[-1], neighbor, comp_w,
                            beta=self.risk_beta, ttc_tau=self.risk_ttc_tau,
                            **self.risk_pet_params, **self.risk_overlap_params
                        )
                    sample_scores.append(risk_dict.get('risk_score', torch.zeros((), device=pred_sample.device)))
                    sample_comp_list.append({k: v for k, v in risk_dict.items() if k != 'risk_score'})

                sample_scores_tensor = torch.stack(sample_scores)
                if multi_samples > 1 and multi_temp > 0:
                    att = torch.softmax(multi_temp * sample_scores_tensor, dim=0)
                    risk_score = (att * sample_scores_tensor).sum()
                else:
                    risk_score = sample_scores_tensor.mean()

                merged = {}
                for comp_dict in sample_comp_list:
                    for k, v in comp_dict.items():
                        merged.setdefault(k, []).append(v)
                risk_components = {k: torch.stack(vlist).mean().detach() for k, vlist in merged.items()}

                # component normalization (skip if log-sigma)
                if self.risk_compnorm_enable and risk_components and not self.risk_use_log_sigma:
                    norm_values = {}
                    for k, v in risk_components.items():
                        val = v.detach()
                        if k not in self._comp_ema_mean:
                            self._comp_ema_mean[k] = val.clone()
                            self._comp_ema_var[k] = torch.zeros_like(val)
                        else:
                            a = self.risk_compnorm_alpha
                            delta = val - self._comp_ema_mean[k]
                            self._comp_ema_mean[k] = self._comp_ema_mean[k] + a * delta
                            self._comp_ema_var[k] = (1 - a) * self._comp_ema_var[k] + a * (delta ** 2)
                        mean_k = self._comp_ema_mean[k]
                        var_k = torch.clamp(self._comp_ema_var[k], 1e-12)
                        z = (val - mean_k) / var_k.sqrt()
                        norm_values[k] = z
                    if norm_values:
                        risk_score_raw = risk_score
                        z_stack = torch.stack(list(norm_values.values()))
                        risk_score = z_stack.mean()
                        for k, z in norm_values.items():
                            risk_components[f"norm_{k}"] = z.detach()
            except Exception:
                pass

        if risk_score_raw is None:
            risk_score_raw = risk_score

        # log-sigma aggregation override
        if self.risk_use_log_sigma and risk_components:
            terms = []
            penalty = torch.zeros((), device=x.device)
            for k, v in risk_components.items():
                val = v if torch.is_tensor(v) else torch.as_tensor(v, device=x.device, dtype=err.dtype)
                ls = self._risk_log_sigmas.get(k, None)
                if ls is None:
                    continue
                ls_c = torch.clamp(ls, -6.0, 3.0)
                terms.append(val * torch.exp(-ls_c))
                penalty = penalty + ls_c
            if terms:
                agg = torch.stack(terms).mean()
                penalty = self.risk_log_sigma_penalty_w * penalty
                risk_components['log_sigma_penalty'] = penalty.detach()
                risk_score = agg + penalty
                if risk_score_raw is None:
                    risk_score_raw = agg.detach()

        # Return tuple for loss computation outside
        # For completeness also return the first-sample pred (relative) as 'pred'
        pred_first = collect_preds[0] if len(collect_preds) > 0 else None
        return err, kl, L_adv_loss, avg_weighted_mse_loss, kinematic_loss, risk_score, risk_components, risk_score_raw, pred_first

    def loss(self, err: torch.Tensor, kl: torch.Tensor, L_adv_loss: torch.Tensor,
             avg_weighted_mse_loss: torch.Tensor, kinematic_loss: torch.Tensor,
             risk_score: torch.Tensor, risk_components: dict, risk_score_raw: torch.Tensor, pred: torch.Tensor):
        """Compose the overall training loss from component terms.

        Parameters
        ----------
        err : (T,N,2) squared reconstruction errors from first sample
        kl : (T, N, z_dim) or stacked tensor of KL per step & latent dim
        L_adv_loss : adversarial proximity style loss (scalar)
        avg_weighted_mse_loss : time-decayed weighted MSE (scalar)
        kinematic_loss : smoothness regularizer (scalar)
        risk_score : aggregated risk to be maximized (scalar)
        risk_components : dict of individual risk component scalars (for logging)

        Returns
        -------
        dict with keys: loss, rec, kl, L_adv_loss, Avg_weighted_mse_loss, kinematic_loss, (optional risk terms)
        """
        # Reconstruction error (mean over time, batch, dim)
        rec = err.mean()
        # KL mean
        kl_mean = kl.mean()
        # --- Configurable loss weighting ---
        # Try to pull dynamic weights from model attributes (set indirectly from config).
        w_rec = getattr(self, 'loss_w_rec', getattr(self, 'LOSS_W_REC', 1.0))
        w_wmse = getattr(self, 'loss_w_wmse', getattr(self, 'LOSS_W_WMSE', 0.2))
        w_kl = getattr(self, 'loss_w_kl', getattr(self, 'LOSS_W_KL', 0.1))
        w_adv = getattr(self, 'loss_w_adv', getattr(self, 'LOSS_W_ADV', 0.01))
        w_kin = getattr(self, 'loss_w_kin', getattr(self, 'LOSS_W_KIN', 0.05))
        combine = getattr(self, 'loss_combine_rec_wmse', getattr(self, 'LOSS_COMBINE_REC_WMSE', False))
        alpha = getattr(self, 'loss_rec_wmse_alpha', getattr(self, 'LOSS_REC_WMSE_ALPHA', 0.3))

        if combine:
            # Merge rec + wmse into a single effective reconstruction to avoid double counting.
            combined_rec = (1 - alpha) * rec + alpha * avg_weighted_mse_loss
            total = w_rec * combined_rec + w_kl * kl_mean + w_adv * L_adv_loss + w_kin * kinematic_loss
        else:
            total = w_rec * rec + w_wmse * avg_weighted_mse_loss + w_kl * kl_mean + w_adv * L_adv_loss + w_kin * kinematic_loss
        if self.risk_enable and hasattr(self, 'risk_weight'):
            # Adaptive auto-scaling: Use RAW risk (pre-normalization) for scale estimation to keep physical interpretability.
            if self.risk_autoscale_enable:
                with torch.no_grad():
                    base_detached = (w_rec * rec + w_wmse * avg_weighted_mse_loss + w_kl * kl_mean + w_adv * L_adv_loss + w_kin * kinematic_loss).detach()
                    raw_detached = risk_score_raw.detach()
                    if self._risk_ema_base is None:
                        self._risk_ema_base = base_detached
                        self._risk_ema_score = raw_detached
                    else:
                        a = self.risk_autoscale_alpha
                        self._risk_ema_base = (1 - a) * self._risk_ema_base + a * base_detached
                        self._risk_ema_score = (1 - a) * self._risk_ema_score + a * raw_detached
                    if torch.isfinite(self._risk_ema_score) and self._risk_ema_score.abs() > 1e-8 and self.risk_weight > 0:
                        target_scale = (self.risk_autoscale_target_frac * self._risk_ema_base.abs()) / (self.risk_weight * self._risk_ema_score.abs())
                        new_scale = (1 - self._risk_autoscale_beta) * self.risk_global_scale + self._risk_autoscale_beta * target_scale
                        self.risk_global_scale = float(torch.clamp(new_scale, self.risk_autoscale_min, self.risk_autoscale_max))
            scaled_risk = self.risk_global_scale * risk_score
            total = total - self.risk_weight * scaled_risk
        else:
            scaled_risk = risk_score

        # --- Risk band penalty L_band: keep risk within [min, max] ---
        band_enable = getattr(self, 'risk_band_enable', getattr(self, 'RISK_BAND_ENABLE', False))
        if band_enable:
            band_min = getattr(self, 'risk_band_min', getattr(self, 'RISK_BAND_MIN', None))
            band_max = getattr(self, 'risk_band_max', getattr(self, 'RISK_BAND_MAX', None))
            band_w   = getattr(self, 'risk_band_weight', getattr(self, 'RISK_BAND_WEIGHT', 0.0))
            band_use_raw = getattr(self, 'risk_band_use_raw', getattr(self, 'RISK_BAND_USE_RAW', True))
            if band_min is not None and band_max is not None and band_w is not None and band_w > 0:
                R_metric = risk_score_raw if band_use_raw else risk_score
                below = torch.relu(torch.as_tensor(band_min, device=R_metric.device, dtype=R_metric.dtype) - R_metric)
                above = torch.relu(R_metric - torch.as_tensor(band_max, device=R_metric.device, dtype=R_metric.dtype))
                L_band = below + above
                total = total + band_w * L_band
            else:
                L_band = torch.zeros((), device=err.device)
        else:
            L_band = torch.zeros((), device=err.device)

        # Add map semantic BCE loss if available
        map_bce_enable = getattr(self, 'MAP_BCE_ENABLE', getattr(self, 'map_bce_enable', False))
        map_bce_w = getattr(self, 'MAP_BCE_WEIGHT', getattr(self, 'map_bce_weight', 0.0))
        map_bce_loss = torch.zeros((), device=err.device)
        if map_bce_enable and map_bce_w > 0 and hasattr(self, '_last_map_bce'):
            map_bce_loss = self._last_map_bce
            total = total + map_bce_w * map_bce_loss

        # --- Entropy regularization for learnable component weights ---
        compw_entropy = None
        entropy_lambda = getattr(self, 'risk_compw_entropy_lambda', getattr(self, 'RISK_COMPW_ENTROPY_LAMBDA', 0.0))
        if self.risk_learn_component_weights and entropy_lambda > 0:
            # obtain current weights as a probability distribution
            comp_w = self._current_component_weights()
            if len(comp_w) > 0:
                w_vec = torch.stack([comp_w[k] for k in sorted(comp_w.keys())])
                # if not normalized (softplus mode), normalize to a prob dist for entropy only
                if self.risk_learn_component_norm != 'softmax':
                    w_sum = w_vec.sum() + 1e-12
                    p = w_vec / w_sum
                else:
                    p = w_vec  # already softmax probabilities
                compw_entropy = -(p * (p.clamp_min(1e-12).log())).sum()
                # maximize entropy => subtract lambda * H in a minimization objective
                total = total - entropy_lambda * compw_entropy

        out = {
            "loss": total.detach() + (total - total.detach()),  # keep graph but stable return
            "rec": rec.detach(),
            "kl": kl_mean.detach(),
            "L_adv_loss": L_adv_loss.detach(),
            "Avg_weighted_mse_loss": avg_weighted_mse_loss.detach(),
            "kinematic_loss": kinematic_loss.detach(),
        }
        out["map_bce_loss"] = map_bce_loss.detach()
        out["loss_w_map_bce"] = torch.as_tensor(map_bce_w)
        # attach optional MAP_BCE debug stats captured during _compute_map_bce
        if hasattr(self, '_last_map_bce_debug') and isinstance(self._last_map_bce_debug, dict):
            try:
                for dk, dv in self._last_map_bce_debug.items():
                    out[dk] = torch.as_tensor(dv)
            except Exception:
                pass
        # risk band diagnostics
        out["risk_L_band"] = L_band.detach()
        out["risk_band_enable"] = torch.as_tensor(1 if getattr(self, 'risk_band_enable', getattr(self, 'RISK_BAND_ENABLE', False)) else 0)
        # If kinematic sub-components available, log them (detached)
        if hasattr(self, '_last_kin_components') and isinstance(self._last_kin_components, dict):
            try:
                for kk, vv in self._last_kin_components.items():
                    out[kk] = vv if torch.is_tensor(vv) else torch.as_tensor(float(vv))
            except Exception:
                pass
        if self.risk_enable:
            out["risk_score"] = risk_score.detach()
            out["risk_score_raw"] = risk_score_raw.detach()
            out["risk_scaled"] = scaled_risk.detach()
            out["risk_weight"] = torch.as_tensor(self.risk_weight)
            out["risk_global_scale"] = torch.as_tensor(self.risk_global_scale)
            out["risk_contrib"] = (self.risk_weight * scaled_risk).detach()
            # expose adaptive scaling diagnostics if enabled
            if self.risk_autoscale_enable and self._risk_ema_base is not None:
                out["risk_autoscale_ema_base"] = self._risk_ema_base.detach().clone()
                out["risk_autoscale_ema_score"] = self._risk_ema_score.detach().clone()
                out["risk_autoscale_target_frac"] = torch.as_tensor(self.risk_autoscale_target_frac)
            for k, v in risk_components.items():
                out[f"comp_{k}"] = v
                if 'log_sigma_penalty' in risk_components:
                    out['log_sigma_penalty'] = risk_components['log_sigma_penalty']
        # Log weights for transparency
        out["loss_w_rec"] = torch.as_tensor(w_rec)
        out["loss_w_wmse"] = torch.as_tensor(w_wmse)
        out["loss_w_kl"] = torch.as_tensor(w_kl)
        out["loss_w_adv"] = torch.as_tensor(w_adv)
        out["loss_w_kin"] = torch.as_tensor(w_kin)
        out["loss_combine_rec_wmse"] = torch.as_tensor(1 if combine else 0)
        out["loss_rec_wmse_alpha"] = torch.as_tensor(alpha)
        if compw_entropy is not None:
            out["compw_entropy"] = compw_entropy.detach()
            out["compw_entropy_lambda"] = torch.as_tensor(entropy_lambda)
        return out

    def _compute_map_bce(self, pred_abs: torch.Tensor, map_meta) -> torch.Tensor:
        """Compute semantic map BCE penalty for absolute predicted positions.

        pred_abs: (T,N,2) absolute world coords
        map_meta: dict with keys:
          - raster: (C,H,W) float tensor in [0,1], probability of being allowed for each class/channel
          - world2map: (3,3) or (2,3) tensor/ndarray affine transform mapping [x,y,1] -> [u,v,1] pixel coords
          - channel: int or None; if None, defaults to vehicle/vru selection by attribute
          - agent_type: optional string ('vehicle','vru') to choose default channel

        Returns scalar BCE ~ mean(-log p_allowed) at predicted points (clamped for stability).
        """
        try:
            raster = map_meta.get('raster', None)
            W2M = map_meta.get('world2map', None)
            channel = map_meta.get('channel', None)
            agent_type = map_meta.get('agent_type', 'vehicle')
            channel_per_agent = map_meta.get('channel_per_agent', None)
            if raster is None or W2M is None:
                return torch.zeros((), device=pred_abs.device)
            if not torch.is_tensor(raster):
                raster = torch.as_tensor(raster, device=pred_abs.device, dtype=pred_abs.dtype)
            if not torch.is_tensor(W2M):
                W2M = torch.as_tensor(W2M, device=pred_abs.device, dtype=pred_abs.dtype)
            if raster.dim() == 3:
                C,H,W = raster.shape
            else:
                return torch.zeros((), device=pred_abs.device)
            # flatten points (T*N,2) -> homogeneous (T*N,3)
            T, N = pred_abs.shape[0], pred_abs.shape[1]
            pts = pred_abs.reshape(-1, 2)
            ones = torch.ones((pts.shape[0],1), device=pts.device, dtype=pts.dtype)
            homo = torch.cat([pts, ones], dim=1)  # (M,3)
            if W2M.shape == (2,3):
                uv = (W2M @ homo.t()).t()  # (M,2)
            else:
                uv1 = (W2M @ homo.t()).t()  # (M,3)
                uv = uv1[...,:2] / (uv1[...,2:].clamp_min(1e-6))
            # normalize to [-1,1] grid for grid_sample: x = (u/(W-1))*2-1 ; y = (v/(H-1))*2-1
            gx = (uv[...,0] / max(W-1,1)) * 2 - 1
            gy = (uv[...,1] / max(H-1,1)) * 2 - 1
            grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)  # (1,M,1,2)
            # oob fraction for diagnostics
            with torch.no_grad():
                oob_mask = (gx < -1) | (gx > 1) | (gy < -1) | (gy > 1)
                oob_frac = float(oob_mask.float().mean().item())

            # Determine channel assignment per point
            if channel_per_agent is not None and isinstance(channel_per_agent, torch.Tensor) and channel_per_agent.numel() == N:
                chan_idx = channel_per_agent.to(dtype=torch.long, device=pred_abs.device)  # (N,)
                # clamp to [0, C-1]
                chan_idx = chan_idx.clamp(0, C-1)
                # expand to T*N length vector for point-wise channels
                chan_pts = chan_idx.repeat_interleave(T).view(-1)  # (T*N,)
                # Sample per channel by masking indices
                p_vals = torch.empty((T*N,), device=pred_abs.device, dtype=pred_abs.dtype)
                ch0_frac = ch1_frac = None
                for c in range(C):
                    mask = (chan_pts == c)
                    if not torch.any(mask):
                        continue
                    # sub-grid for this channel
                    sub_grid = grid[:, mask, :, :]  # (1,M_c,1,2)
                    feat = raster[c:c+1].unsqueeze(0)  # (1,1,H,W)
                    samp = torch.nn.functional.grid_sample(feat, sub_grid, mode='bilinear', align_corners=True)
                    p_vals[mask] = samp.view(-1)
                    # quick two-channel stats (common case C>=2)
                    if c == 0:
                        ch0_frac = float(mask.float().mean().item())
                    elif c == 1:
                        ch1_frac = float(mask.float().mean().item())
                p = p_vals.clamp(1e-6, 1-1e-6)
            else:
                # single channel for all points (fallback to agent_type or config default)
                if channel is None:
                    if agent_type == 'vru':
                        channel = int(getattr(self, 'MAP_BCE_CHANNEL_VRU', 1))
                    else:
                        channel = int(getattr(self, 'MAP_BCE_CHANNEL_VEHICLE', 0))
                channel = max(0, min(C-1, int(channel)))
                feat = raster[channel:channel+1].unsqueeze(0)  # (1,1,H,W)
                samp = torch.nn.functional.grid_sample(feat, grid, mode='bilinear', align_corners=True)
                p = samp.view(-1).clamp(1e-6, 1-1e-6)
            bce = -(p.log()).mean()
            # Optional debug stats collection
            try:
                if getattr(self, 'MAP_BCE_DEBUG', getattr(self, 'map_bce_debug', False)):
                    with torch.no_grad():
                        p_mean = float(p.mean().item())
                        p_min = float(p.min().item())
                        p_max = float(p.max().item())
                        allow_frac = float((p > 0.5).float().mean().item())
                        debug = {
                            'map_p_mean': p_mean,
                            'map_p_min': p_min,
                            'map_p_max': p_max,
                            'map_oob_frac': oob_frac,
                            'map_allow_frac': allow_frac,
                        }
                        # include channel assignment fractions if computed
                        if 'ch0_frac' in locals() and ch0_frac is not None:
                            debug['map_channel0_frac'] = float(ch0_frac)
                        if 'ch1_frac' in locals() and ch1_frac is not None:
                            debug['map_channel1_frac'] = float(ch1_frac)
                        self._last_map_bce_debug = debug
            except Exception:
                pass
            return bce
        except Exception:
            return torch.zeros((), device=pred_abs.device)

    # ---------------- Mamba enable utility ----------------
    def enable_mamba(self, encoder: bool=False, decoder: bool=False, d_model_enc: Optional[int]=None, d_model_dec: Optional[int]=None):
        """Dynamically enable Mamba blocks. Safe to call multiple times.

        Parameters
        ----------
        encoder : bool
            Replace observation encoder recurrent loop (rnn_fx) with StackedMamba.
        decoder : bool
            Replace generative decoder recurrent loop (rnn_fy) with StackedMamba.
        d_model_enc : int, optional
            Hidden size for encoder Mamba (defaults to existing rnn_fx hidden dim).
        d_model_dec : int, optional
            Hidden size for decoder Mamba (defaults to existing rnn_fy hidden dim).
        """
        if build_mamba_or_none is None:
            return  # silently ignore if dependency not present
        if encoder and self.mamba_encoder is None:
            hid = d_model_enc or self.rnn_fx.hidden_size
            self.mamba_encoder = build_mamba_or_none(True, hid, num_layers=1, dropout=0.0)
            self.use_mamba_encoder = True
            # create encoder projection if concat(neigh,s) dim != hid
            enc_in_dim = self.embed_q[-1].out_features + self.embed_s[-1].out_features if hasattr(self.embed_s[-1], 'out_features') else None
            # Actually neigh context uses feature_dim (embed_q output) and s[t] uses self_embed_dim; embed_s last linear gives self_embed_dim
            try:
                self_embed_dim = self.embed_s[-1].out_features
                feature_dim = self.embed_q[-1].out_features
                enc_concat_dim = feature_dim + self_embed_dim
                if enc_concat_dim != hid:
                    self.mamba_enc_in = torch.nn.Linear(enc_concat_dim, hid)
                else:
                    self.mamba_enc_in = torch.nn.Identity()
            except Exception:
                self.mamba_enc_in = torch.nn.Identity()
        if decoder and self.mamba_decoder is None:
            hid = d_model_dec or self.rnn_fy.hidden_size
            self.mamba_decoder = build_mamba_or_none(True, hid, num_layers=1, dropout=0.0)
            self.use_mamba_decoder = True
            # decoder projection from zd (latent+maybe delta) dim to hid
            try:
                # use a dummy tensor through embed_zd to infer output dim if attribute missing
                if hasattr(self.embed_zd, 'out_dim'):
                    zd_dim = self.embed_zd.out_dim
                else:
                    zd_dim = self.embed_zd(torch.zeros(1, self.p_z.mu.out_features, device=next(self.parameters()).device)).shape[-1]  # risky, fallback below if fail
            except Exception:
                # Fallback: assume latent dimension equal to self.p_z.mu.out_features or 32
                zd_dim = getattr(self.p_z.mu, 'out_features', 32)
            if zd_dim != hid:
                self.mamba_dec_in = torch.nn.Linear(zd_dim, hid)
            else:
                self.mamba_dec_in = torch.nn.Identity()
    
    def enable_multihead_attention(self, heads: int = 4, dropout: float = 0.0):
        if heads < 2:
            return
        self.use_multihead_attn = True
        self.mha_heads = heads
        d_model = self.embed_q[-1].out_features  # feature_dim
        if d_model % heads != 0:
            raise ValueError(f"feature_dim {d_model} not divisible by heads {heads}")
        # projections
        self.mha_q = torch.nn.Linear(d_model, d_model, bias=False)
        self.mha_k = torch.nn.Linear(d_model, d_model, bias=False)
        neigh_dim = self.embed_n[-1].out_features
        self.mha_v = torch.nn.Linear(neigh_dim, d_model, bias=False)
        self.mha_out = torch.nn.Linear(d_model, d_model)
        self.mha_drop = torch.nn.Dropout(dropout) if dropout > 0 else None

    # --- Helper for external epoch-level logging (CSV) ---
    def get_autoscale_state(self) -> Dict[str, float]:
        """Return current autoscale diagnostics as plain Python floats for logging.
        Keys: risk_global_scale, ema_base, ema_raw_score, target_frac, weight
        """
        if not self.risk_enable:
            return {}
        out = {
            'risk_global_scale': float(self.risk_global_scale),
            'risk_weight': float(self.risk_weight),
            'risk_autoscale_target_frac': float(getattr(self, 'risk_autoscale_target_frac', 0.0)),
        }
        if getattr(self, '_risk_ema_base', None) is not None:
            out['risk_autoscale_ema_base'] = float(self._risk_ema_base)
        if getattr(self, '_risk_ema_score', None) is not None:
            out['risk_autoscale_ema_raw_score'] = float(self._risk_ema_score)
        # include learned component weights if enabled
        if self.risk_learn_component_weights:
            for k, v in self.get_learned_component_weights().items():
                out[f'compw_{k}'] = v
        return out
