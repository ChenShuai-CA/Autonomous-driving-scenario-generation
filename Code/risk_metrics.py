import torch
from typing import Dict, Tuple

@torch.jit.script_if_tracing
def safe_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return torch.sqrt(torch.clamp(torch.sum(x * x, dim=dim), min=eps))

def softmin_distance(dist: torch.Tensor, beta: float) -> torch.Tensor:
    """Compute differentiable soft-min over neighbor distances.
    Args:
        dist: (T, B, Nn) distance tensor
        beta: temperature (>0). Larger => closer to hard min.
    Returns:
        softmin: (T, B) soft-min value per time & batch
    """
    # softmin = -1/beta * log sum_i exp(-beta * d_i)
    softmin = -torch.logsumexp(-beta * dist, dim=-1) / beta
    return softmin

def ttc_risk(rel_pos: torch.Tensor, rel_vel: torch.Tensor, tau_scale: float = 1.5, max_ttc: float = 10.0) -> torch.Tensor:
    """Enhanced TTC based risk.
    rel_pos: (T,B,Nn,2) neighbor_pos - ego_pos
    rel_vel: (T,B,Nn,2) neighbor_vel - ego_vel
    Returns:
        risk_ttc: (T,B,Nn)
    """
    eps = 1e-6
    dist = safe_norm(rel_pos, dim=-1, eps=eps)
    # closing speed along line-of-sight
    rel_pos_unit = rel_pos / dist.unsqueeze(-1)
    closing_speed = -(rel_vel * rel_pos_unit).sum(dim=-1)  # positive when approaching
    approaching = closing_speed > 0
    ttc_approach = dist / (closing_speed + eps)
    ttc_approach = torch.clamp(ttc_approach, 0.0, max_ttc)
    ttc = torch.where(approaching, ttc_approach, torch.full_like(dist, max_ttc))
    # risk large when TTC small: exp(-ttc / tau)
    risk = torch.exp(-ttc / tau_scale)
    return risk

def compute_risk_components(pred: torch.Tensor, x_last: torch.Tensor, neighbor: torch.Tensor, beta: float = 2.0,
                             ttc_tau: float = 1.5, min_dist_clip: float = 0.1,
                             # PET params
                             pet_dist_th: float = 2.0, pet_alpha: float = 8.0, pet_beta: float = 0.7, pet_gamma: float = 0.0,
                             pet_continuous: bool = True, pet_time_temp: float = 4.0,
                             # Overlap params
                             ov_r_self: float = 0.5, ov_r_neigh: float = 0.5, ov_margin: float = 0.3,
                             ov_k: float = 15.0, ov_time_temp: float = 4.0, ov_neigh_temp: float = 3.0,
                             # OBB extensions
                             ov_use_obb: bool = False,
                             ov_self_length: float = 4.5, ov_self_width: float = 1.8,
                             ov_neigh_length: float = 4.5, ov_neigh_width: float = 1.8,
                             ov_min_speed: float = 1e-3, ov_axis_beta: float = 12.0,
                             ov_debug: bool = False,
                             enable_pet: bool = True, enable_overlap: bool = True) -> Dict[str, torch.Tensor]:
    """Compute risk components for a single-sample predicted trajectory.
    Args:
        pred: (T,B,2) relative displacements cumulative (already cumulative in caller) relative to x_last
        x_last: (B,6) last observed ego state
        neighbor: (L_all,B,Nn,6) includes ob + pred horizon frames; padded neighbors use large constant 1e9
        beta: temperature for soft-min
        ttc_tau: TTC risk decay scale
        min_dist_clip: small constant to avoid blow-up
    Returns:
        dict with scalar tensors (already aggregated across time & neighbors)
    """
    device = pred.device
    ob_horizon = x_last.shape[0] if x_last.dim() == 3 else None  # not used; derive via neighbor / pred lengths
    T = pred.size(0)
    # absolute ego positions
    ego_pos = pred + x_last[:, :2].unsqueeze(0)  # (T,B,2)

    # infer slicing for future neighbor frames
    total_len = neighbor.size(0)
    pred_horizon = T
    ob_len = total_len - pred_horizon
    neighbor_future = neighbor[ob_len:ob_len+pred_horizon, ..., :2]  # (T,B,Nn,2)
    neighbor_future_vel = neighbor[ob_len:ob_len+pred_horizon, ..., 2:4]

    # mask padded neighbors
    mask_valid = ~(neighbor_future.abs() > 1e8).any(dim=-1)  # (T,B,Nn)

    # distances
    dist = safe_norm(neighbor_future - ego_pos.unsqueeze(2), dim=-1)
    dist = dist.clamp(min=min_dist_clip)
    dist_masked = dist.clone()
    dist_masked[~mask_valid] = 1e6
    softmin = softmin_distance(dist_masked, beta=beta)  # (T,B)
    risk_min_dist_time = torch.exp(-softmin)  # (T,B)
    risk_min_dist = risk_min_dist_time.mean()

    # ego velocities (approx) first step pred[0], then diff
    ego_vel = torch.zeros_like(pred)
    ego_vel[0] = pred[0]
    if T > 1:
        ego_vel[1:] = pred[1:] - pred[:-1]
    rel_pos = neighbor_future - ego_pos.unsqueeze(2)
    rel_vel = neighbor_future_vel - ego_vel.unsqueeze(2)
    risk_ttc_map = ttc_risk(rel_pos, rel_vel, tau_scale=ttc_tau)  # (T,B,Nn)
    risk_ttc_map = risk_ttc_map.masked_fill(~mask_valid, 0.0)
    # aggregate: emphasize worst-case over neighbors then mean over time
    risk_ttc = risk_ttc_map.max(dim=-1)[0].mean()
    components: Dict[str, torch.Tensor] = {
        "risk_min_dist": risk_min_dist,
        "risk_ttc": risk_ttc
    }

    # ---------- PET Risk (soft proxy) ----------
    if enable_pet:
        if pet_continuous:
            # Continuous-time relative motion approximation per frame
            # Need relative position & velocity already computed (rel_pos, rel_vel)
            # For each frame compute closest approach inside the frame interval [0,1]
            v_rel = rel_vel  # (T,B,Nn,2)
            r0 = rel_pos     # (T,B,Nn,2)
            v_rel_norm2 = (v_rel * v_rel).sum(-1)  # (T,B,Nn)
            dot_rv = (r0 * v_rel).sum(-1)
            # tau* = clamp( - (r0·v_rel)/||v_rel||^2 , 0, 1 ) ; handle near-zero velocity
            tau_star = torch.zeros_like(dot_rv)
            mask_v = v_rel_norm2 > 1e-8
            tau_star[mask_v] = -dot_rv[mask_v] / (v_rel_norm2[mask_v] + 1e-9)
            tau_star = tau_star.clamp(0.0, 1.0)
            r_closest = r0 + v_rel * tau_star.unsqueeze(-1)
            d_min = safe_norm(r_closest, dim=-1)  # (T,B,Nn)
            # activation by distance threshold
            act = torch.sigmoid(pet_alpha * (pet_dist_th - d_min)) * mask_valid.float()
            # time cost prefer earlier & smaller tau* (earlier in frame + smaller time index)
            # combine global frame index t and intra-frame tau*: t_eff = t + tau*
            t_idx_full = torch.arange(T, device=device, dtype=pred.dtype).view(T,1,1)
            t_eff = t_idx_full + tau_star  # (T,B,Nn)
            # per-frame neighbor risk raw
            risk_raw = act * torch.exp(-pet_beta * t_eff)
            # time aggregation (emphasize earliest effective time)
            if pet_time_temp > 0:
                w_t = torch.softmax(pet_time_temp * risk_raw, dim=0)
                r_time = (w_t * risk_raw).sum(0)  # (B,Nn)
            else:
                r_time = risk_raw.mean(0)
            if pet_gamma > 0:
                w_n = torch.softmax(pet_gamma * r_time, dim=-1)
                risk_pet = (w_n * r_time).sum(-1).mean()
            else:
                risk_pet = r_time.mean()
            components["risk_pet"] = risk_pet
        else:
            # Legacy PET expectation/time-difference proxy
            w = torch.sigmoid(pet_alpha * (pet_dist_th - dist)) * mask_valid.float()
            w_sum_time = w.sum(0, keepdim=True) + 1e-6
            w_norm = w / w_sum_time
            t_idx = torch.linspace(0, T-1, T, device=device).view(T,1,1)
            t_bar = (w_norm * t_idx).sum(0)
            total_w = w.sum() + 1e-6
            t_ref = (w * t_idx).sum() / total_w
            pet_soft = torch.abs(t_bar - t_ref)
            r_pet_neighbors = torch.exp(-pet_beta * pet_soft)
            if pet_gamma > 0:
                att = torch.softmax(pet_gamma * r_pet_neighbors, dim=-1)
                risk_pet = (att * r_pet_neighbors).sum(-1).mean()
            else:
                risk_pet = r_pet_neighbors.mean()
            components["risk_pet"] = risk_pet

    # ---------- Overlap Risk ----------
    if enable_overlap:
        if not ov_use_obb:
            # --- Legacy circular penetration ---
            thresh = ov_r_self + ov_r_neigh + ov_margin
            pen = thresh - dist  # (T,B,Nn)
            o_frame = torch.nn.functional.softplus(ov_k * pen) / ov_k  # (T,B,Nn)
            o_frame = o_frame * mask_valid.float()
            if ov_time_temp > 0:
                w_t = torch.softmax(ov_time_temp * o_frame, dim=0)
                o_time = (w_t * o_frame).sum(0)
            else:
                o_time = o_frame.mean(0)
            if ov_neigh_temp > 0:
                w_n = torch.softmax(ov_neigh_temp * o_time, dim=-1)
                o_neigh = (w_n * o_time).sum(-1)
            else:
                o_neigh = o_time.mean(-1)
            risk_overlap = o_neigh.mean()
            components["risk_overlap"] = risk_overlap
        else:
            # --- OBB Overlap Approximation ---
            # Prepare ego absolute pos & approximate velocity per frame
            ego_abs = ego_pos  # (T,B,2)
            ego_vel = torch.zeros_like(ego_abs)
            ego_vel[0] = ego_abs[0] - x_last[:, :2]
            if T > 1:
                ego_vel[1:] = ego_abs[1:] - ego_abs[:-1]
            # neighbor positions (already have neighbor_future)
            neigh_abs = neighbor_future  # (T,B,Nn,2)
            neigh_vel = neighbor_future_vel  # (T,B,Nn,2)
            # headings
            def heading_from_vel(v):
                # v: (...,2)
                speed = safe_norm(v, dim=-1)
                hx = torch.where(speed > ov_min_speed, v[...,0]/(speed+1e-9), torch.ones_like(speed))
                hy = torch.where(speed > ov_min_speed, v[...,1]/(speed+1e-9), torch.zeros_like(speed))
                return torch.stack((hx, hy), dim=-1)
            ego_dir = heading_from_vel(ego_vel)  # (T,B,2)
            neigh_dir = heading_from_vel(neigh_vel)  # (T,B,B,Nn,2) -> Actually (T,B,Nn,2)
            # For each pair compute separation along 4 axes: ego_dir, ego_perp, neigh_dir, neigh_perp
            ego_perp = torch.stack((-ego_dir[...,1], ego_dir[...,0]), dim=-1)
            neigh_perp = torch.stack((-neigh_dir[...,1], neigh_dir[...,0]), dim=-1)
            # Centers difference d = neigh - ego
            d = neigh_abs - ego_abs.unsqueeze(2)  # (T,B,Nn,2)
            # Extents (half-length, half-width)
            eL = ov_self_length * 0.5
            eW = ov_self_width * 0.5
            nL = ov_neigh_length * 0.5
            nW = ov_neigh_width * 0.5
            # Build axis set A = [a1,a2,b1,b2]
            a1 = ego_dir.unsqueeze(2)           # (T,B,1,2)
            a2 = ego_perp.unsqueeze(2)
            b1 = neigh_dir                      # (T,B,Nn,2)
            b2 = neigh_perp
            # We need same shape for broadcasting: expand ego axes over neighbors
            a1e = a1.expand(-1,-1,neigh_abs.size(2),-1)  # (T,B,Nn,2)
            a2e = a2.expand_as(a1e)
            # Axes tensor shape (4,T,B,Nn,2)
            axes = torch.stack((a1e, a2e, b1, b2), dim=0)  # (4,T,B,Nn,2)
            # Normalize just in case (already unit, but numerical safety)
            axes = axes / (safe_norm(axes, dim=-1, eps=1e-9).unsqueeze(-1))
            # Project center distance onto each axis
            proj_d = torch.abs((d.unsqueeze(0) * axes).sum(-1))  # (4,T,B,Nn)
            # Projection radii for ego on each axis
            # For a1 axis: radius = eL; for a2 axis: eW; for b1 axis: combine components of ego axes onto b1; etc.
            # General formula: R = |a1·axis|*eL + |a2·axis|*eW
            a1_unit = a1e / (safe_norm(a1e, dim=-1, eps=1e-9).unsqueeze(-1))
            a2_unit = a2e / (safe_norm(a2e, dim=-1, eps=1e-9).unsqueeze(-1))
            # For each axis compute |dot(axis, a1)| and |dot(axis, a2)|
            dot_a1 = torch.abs((axes * a1_unit.unsqueeze(0)).sum(-1))  # (4,T,B,Nn)
            dot_a2 = torch.abs((axes * a2_unit.unsqueeze(0)).sum(-1))
            R_ego = dot_a1 * eL + dot_a2 * eW  # (4,T,B,Nn)
            # For neighbor box: axis projections using its own local axes b1,b2
            b1_unit = b1 / (safe_norm(b1, dim=-1, eps=1e-9).unsqueeze(-1))
            b2_unit = b2 / (safe_norm(b2, dim=-1, eps=1e-9).unsqueeze(-1))
            b1_unit_e = b1_unit.unsqueeze(0)
            b2_unit_e = b2_unit.unsqueeze(0)
            dot_b1 = torch.abs((axes * b1_unit_e).sum(-1))
            dot_b2 = torch.abs((axes * b2_unit_e).sum(-1))
            R_neigh = dot_b1 * nL + dot_b2 * nW
            sep = proj_d - (R_ego + R_neigh)  # (4,T,B,Nn) positive => separated
            # Soft-min across axes to approximate maximum penetration (negative separation)
            # sep_axes_min ≈ -1/β log Σ exp(-β sep)
            sep_min = -torch.logsumexp(-ov_axis_beta * sep, dim=0) / ov_axis_beta  # (T,B,Nn)
            # penetration (negative sep_min) + margin
            pen = ov_margin - sep_min  # (T,B,Nn)
            o_frame = torch.nn.functional.softplus(ov_k * pen) / ov_k
            o_frame = o_frame * mask_valid.float()
            if ov_time_temp > 0:
                w_t = torch.softmax(ov_time_temp * o_frame, dim=0)
                o_time = (w_t * o_frame).sum(0)
            else:
                o_time = o_frame.mean(0)
            if ov_neigh_temp > 0:
                w_n = torch.softmax(ov_neigh_temp * o_time, dim=-1)
                o_neigh = (w_n * o_time).sum(-1)
            else:
                o_neigh = o_time.mean(-1)
            risk_overlap = o_neigh.mean()
            components["risk_overlap"] = risk_overlap
            if ov_debug:
                # Provide additional diagnostics: mean negative sep (penetration), fraction active
                mean_pen = (pen * mask_valid.float()).clamp(min=0).mean()  # after margin
                active_frac = ((pen > 0) & mask_valid).float().mean()
                components["overlap_mean_pen"] = mean_pen.detach()
                components["overlap_active_frac"] = active_frac.detach()

    return components

def aggregate_risk(components: Dict[str, torch.Tensor], weights: Dict[str, float]) -> torch.Tensor:
    total = torch.zeros((), device=next(iter(components.values())).device)
    for k, v in components.items():
        w = weights.get(k, 0.0)
        if w != 0.0:
            total = total + w * v
    return total

def compute_risk_score(pred: torch.Tensor, x_last: torch.Tensor, neighbor: torch.Tensor,
                       weights: Dict[str, float], beta: float = 2.0, ttc_tau: float = 1.5,
                       # pass-through of new params with sane defaults
                       pet_dist_th: float = 2.0, pet_alpha: float = 8.0, pet_beta: float = 0.7, pet_gamma: float = 0.0,
                       pet_continuous: bool = True, pet_time_temp: float = 4.0,
                       ov_r_self: float = 0.5, ov_r_neigh: float = 0.5, ov_margin: float = 0.3,
                       ov_k: float = 15.0, ov_time_temp: float = 4.0, ov_neigh_temp: float = 3.0,
                       ov_use_obb: bool = False, ov_self_length: float = 4.5, ov_self_width: float = 1.8,
                       ov_neigh_length: float = 4.5, ov_neigh_width: float = 1.8,
                       ov_min_speed: float = 1e-3, ov_axis_beta: float = 12.0,
                       ov_debug: bool = False,
                       enable_pet: bool = True, enable_overlap: bool = True) -> Dict[str, torch.Tensor]:
    comps = compute_risk_components(
        pred, x_last, neighbor, beta=beta, ttc_tau=ttc_tau,
    pet_dist_th=pet_dist_th, pet_alpha=pet_alpha, pet_beta=pet_beta, pet_gamma=pet_gamma,
    pet_continuous=pet_continuous, pet_time_temp=pet_time_temp,
        ov_r_self=ov_r_self, ov_r_neigh=ov_r_neigh, ov_margin=ov_margin,
        ov_k=ov_k, ov_time_temp=ov_time_temp, ov_neigh_temp=ov_neigh_temp,
        ov_use_obb=ov_use_obb, ov_self_length=ov_self_length, ov_self_width=ov_self_width,
        ov_neigh_length=ov_neigh_length, ov_neigh_width=ov_neigh_width,
        ov_min_speed=ov_min_speed, ov_axis_beta=ov_axis_beta, ov_debug=ov_debug,
        enable_pet=enable_pet, enable_overlap=enable_overlap
    )
    score = aggregate_risk(comps, weights)
    comps["risk_score"] = score
    return comps
