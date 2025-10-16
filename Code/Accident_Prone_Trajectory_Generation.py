import sys
# 导入sys模块，用于处理命令行参数和系统相关操作

import os
import sys
import time
import csv
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate
from itertools import islice
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.animation import FuncAnimation
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import importlib
from typing import List, Optional
import xml.etree.ElementTree as ET
# 导入所需的库和模块，用于文件操作、数据处理、可视化、深度学习等

# 自定义模块的导入
import DRF
import Lanelet_Map_Viz 
from social_vae import SocialVAE
from data import Dataloader
from utils import ADE_FDE, FPC, seed, get_rng_state, set_rng_state
from risk_metrics import compute_risk_score
# 导入项目中自定义的模块和函数

try:
    import lanelet2
    use_lanelet2_lib = True
    print("Successfully imported lanelet2.")
except ImportError:
    import warnings
    use_lanelet2_lib = False
    warnings.warn("lanelet2 未安装，地图底图将被跳过。")
# 尝试导入 lanelet2 库，如果失败则发出警告

"""
x: 输入轨迹位置。Shape_x: (L1+1) x N x 6，其中 L1 是第一段轨迹的长度，
N 是轨迹的数量，6 表示轨迹信息（例如 x 坐标、y 坐标、速度分量等）。

y: 额外的轨迹信息（可选）。Shape_y: L2 x N x 2 

neighbor: 每条轨迹的邻居信息。Shape: (L1+L2+1) x N x Nn x 6，
其中 L2 是第二段轨迹的长度，
Nn 是邻居的数量，6 表示邻居信息。
"""
# 说明输入数据的格式和含义

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--train", nargs='+', default=[])
parser.add_argument("--test", nargs='+', default=[])
parser.add_argument("--frameskip", type=int, default=1)
parser.add_argument("--config", type=str, default="config/Interaction.py")
parser.add_argument("--ckpt", type=str, default="log_formal_mamba_component_weights", help="包含 ckpt-best 的目录")
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--no-fpc", action="store_true", default=False)
parser.add_argument("--fpc-finetune", action="store_true", default=False)
parser.add_argument("--csv", type=str, default="Code/data/INTERACTION/INTERACTION-Dataset-DR-multi-v1_2/val/DR_USA_Intersection_EP1_val.csv", help="用于生成的验证集 CSV 路径")
parser.add_argument("--max-cases", type=int, default=50, help="最多处理多少个 case（避免一次性过多输出）")
parser.add_argument("--rerank-enable", action="store_true", default=True, help="启用可行域惩罚+jerk重排名选择可视化样本")
parser.add_argument("--vis-topk", type=int, default=3, help="每个 case 可视化的候选条数（与FPC/重排名结合，默认3，建议5）")
parser.add_argument("--per-traj-output", action="store_true", default=False, help="将选中的每条候选分别另存为独立 GIF/CSV/NPY（文件名追加 _sel{idx}_rank{rank}）")
parser.add_argument("--lambda-lane", type=float, default=0.5, help="可行域(出道)惩罚系数")
parser.add_argument("--lambda-jerk", type=float, default=0.1, help="jerk 平滑惩罚系数")
# 仅将自车限定为某类主体（如 vehicle），过滤候选 ego；可选值：vehicle|pedestrian|bicycle|any
parser.add_argument("--ego-type-filter", type=str, default="vehicle",
                    help="限定自车类别：vehicle|pedestrian|bicycle|any，默认 vehicle")
# 邻居类型过滤与最少匹配邻居数
parser.add_argument("--neighbor-type-filter", type=str, default="any",
                    help="邻居类别过滤：vehicle|pedestrian|bicycle|vru|any（vru=行人或骑行者），默认 any")
parser.add_argument("--min-matched-neighbors", type=int, default=0,
                    help="最少匹配邻居的数量阈值；当 neighbor-type-filter=vehicle 时即最少车辆邻居数。默认 0")
parser.add_argument("--min-ego-distance", type=float, default=4.0,
                    help="自车历史段最小位移阈值（单位 m），低于该值则跳过该样本；默认 4.0m")
# 风险检测（危险/易出事故）阈值与权重
parser.add_argument("--risk-detect-enable", action="store_true", default=True,
                    help="启用基于风控组件的危险样本判定并在可视化中高亮")
parser.add_argument("--risk-threshold", type=float, default=0.30,
                    help="危险判定阈值：risk_score >= 阈值 即标记为危险；默认 0.30")
parser.add_argument("--risk-w-min-dist", type=float, default=1.0, help="risk_min_dist 权重")
parser.add_argument("--risk-w-ttc", type=float, default=1.0, help="risk_ttc 权重")
parser.add_argument("--risk-w-pet", type=float, default=0.5, help="risk_pet 权重")
parser.add_argument("--risk-w-overlap", type=float, default=1.0, help="risk_overlap 权重")
# 仅高亮参与危险的少数邻居（按未来期与ego最小距离排序）
parser.add_argument("--highlight-topk-neigh", type=int, default=3, help="危险样本时高亮的邻居个数（按未来期最小距离排序）")
parser.add_argument("--highlight-mode", type=str, default="risk", choices=["distance", "risk"],
                    help="高亮邻居选择模式：distance=最小距离，risk=基于风险贡献（默认）")
# GIF 控制：快速验证时可跳过或降采样
parser.add_argument("--no-gif", action="store_true", default=False, help="不保存GIF，仅输出CSV/NPY（用于快速验证）")
parser.add_argument("--gif-fps", type=int, default=10, help="GIF 帧率")
parser.add_argument("--gif-dpi", type=int, default=100, help="GIF 分辨率 DPI")
parser.add_argument("--gif-frame-skip", type=int, default=1, help="帧下采样步长，>1 可显著加速生成")
# 叠加层控制：是否显示 VRU/车辆/Keepout 掩模，以及图例与透明度
parser.add_argument("--overlay-vru", dest="overlay_vru", action="store_true", default=True, help="叠加 VRU 可行区栅格")
parser.add_argument("--no-overlay-vru", dest="overlay_vru", action="store_false", help="禁用 VRU 可行区叠加")
parser.add_argument("--overlay-veh", dest="overlay_veh", action="store_true", default=True, help="叠加车辆车道（vehicle channel）栅格")
parser.add_argument("--no-overlay-veh", dest="overlay_veh", action="store_false", help="禁用车辆车道叠加")
parser.add_argument("--overlay-keepout", dest="overlay_keepout", action="store_true", default=True, help="叠加 keepout 区域栅格")
parser.add_argument("--no-overlay-keepout", dest="overlay_keepout", action="store_false", help="禁用 keepout 叠加")
parser.add_argument("--overlay-alpha", type=float, default=0.28, help="叠加层透明度 [0,1]")
parser.add_argument("--overlay-legend", dest="overlay_legend", action="store_true", default=True, help="显示叠加层图例")
parser.add_argument("--no-overlay-legend", dest="overlay_legend", action="store_false", help="隐藏叠加层图例")
# 候选轨迹的硬性筛选（生成期守门）
parser.add_argument("--filter-enable", action="store_true", default=True, help="启用候选轨迹的硬性筛选")
parser.add_argument("--filter-outfrac-max", type=float, default=0.2, help="允许的最大出道比例（0~1）")
parser.add_argument("--filter-jerk-max", type=float, default=0.6, help="允许的最大平均 jerk 阈值")
parser.add_argument("--filter-turn-deg-max", type=float, default=75.0, help="允许的最大相邻转角阈值（度）")
# 定义命令行参数，用于配置训练、测试、设备、随机种子等

output_path = os.getcwd()+"/Output"
# 定义输出路径，默认在当前工作目录下的 Output 文件夹

# 检查目录是否存在
if not os.path.exists(output_path):
    # 如果目录不存在，则创建
    os.makedirs(output_path)
    print(f"Directory created at {output_path}")
else:
    print(f"Directory already exists at {output_path}")
# 检查并创建输出目录

#%%

def process_data_to_tensors(data, agent_threshold, ob_horizon, future_pre, device,
                            max_cases=50, ego_type_filter: str = "vehicle",
                            neighbor_type_filter: str = "any", min_matched_neighbors: int = 0):
    """
    将数据处理为张量格式
    :param data: 输入数据
    :param agent_threshold: 代理阈值
    :param ob_horizon: 观测时间范围
    :param future_pre: 预测时间范围
    :param device: 设备（CPU 或 GPU）
    :return: 处理后的张量列表
    """
    N=ob_horizon+future_pre+2
    tensors_list = []
    num_items_to_process = max_cases  # 限制处理的案例数量
    processed_count=0

    grouped_data = data.groupby('case_id')
    # 按 case_id 分组数据

    # 规范化过滤目标
    def norm_type(s: str) -> str:
        if s is None:
            return "unknown"
        ss = str(s).strip().lower()
        if any(k in ss for k in ["car", "vehicle", "truck", "bus", "van", "auto"]):
            return "vehicle"
        if any(k in ss for k in ["pedestrian", "walker", "human"]):
            return "pedestrian"
        if any(k in ss for k in ["bicycle", "bike", "cyclist"]):
            return "bicycle"
        if ss == "pedestrian/bicycle":
            # INTERACTION 常见合并标签，既可视作行人也可视作骑行者；这里不把它识别为 vehicle
            return "pedestrian"
        return "unknown"

    ego_filter_norm = "any" if ego_type_filter is None else ego_type_filter.strip().lower()
    neigh_filter_norm = "any" if neighbor_type_filter is None else neighbor_type_filter.strip().lower()

    for idx, (case_key, group) in tqdm(enumerate(grouped_data), desc="Processing Cases", total=num_items_to_process):
        df_selected = group[['frame_id', 'track_id', 'x', 'y', 'agent_type']]
        df_selected.columns = ['frame_ID', 'agent_ID', 'pos_x', 'pos_y', 'agent_type']
        df_selected = df_selected.sort_values(by='frame_ID')
        df_selected = df_selected.reset_index(drop=True)
        df_selected['frame_ID'] = df_selected['frame_ID'].astype(int)
        df_selected['agent_ID'] = df_selected['agent_ID'].astype(int)
        df_selected['pos_x'] = df_selected['pos_x'].round(5)
        df_selected['pos_y'] = df_selected['pos_y'].round(5)
        # 数据预处理：重命名列、排序、重置索引、数据类型转换

        agent_ids_frame_1 = df_selected.loc[df_selected['frame_ID'] == 1, 'agent_ID'].unique()
        agent_ids_max_frame = df_selected.loc[df_selected['frame_ID'] == df_selected['frame_ID'].max(), 'agent_ID'].unique()
        # 获取第一帧和最后一帧的代理 ID

        if set(agent_ids_frame_1).issubset(set(agent_ids_max_frame)):
            unique_agent_ids = df_selected['agent_ID'].nunique()
            # 检查第一帧的代理是否是最后一帧的子集

            if unique_agent_ids >= agent_threshold:
                windows = [df_selected[df_selected['frame_ID'].isin(range(start_frame, start_frame + ob_horizon + future_pre + 2))]
                           for start_frame in range(df_selected['frame_ID'].min(), df_selected['frame_ID'].max() - ob_horizon - future_pre + 1)]
                # 滑动窗口提取时间范围内的数据

                for scene_data in windows:            
                    Possible_ego_ids = scene_data['agent_ID'].value_counts() == ob_horizon + future_pre + 2
                    Possible_ego_ids_list = Possible_ego_ids[Possible_ego_ids].index.tolist()
                    # 根据 agent_type 过滤可能的自车 ID
                    if ego_filter_norm != "any":
                        id2type = scene_data.drop_duplicates('agent_ID').set_index('agent_ID')['agent_type'].to_dict()
                        filtered_ids = []
                        for _id in Possible_ego_ids_list:
                            at = norm_type(id2type.get(_id, 'unknown'))
                            if ego_filter_norm == at:
                                filtered_ids.append(_id)
                        # 打印过滤统计
                        print(f"[ego-type-filter] case={idx} target={ego_filter_norm} before={len(Possible_ego_ids_list)} after={len(filtered_ids)}")
                        Possible_ego_ids_list = filtered_ids
                    # 找到可能的自车 ID

                    for Possible_ego in Possible_ego_ids_list:
                        ego_full = scene_data[scene_data['agent_ID'] == Possible_ego].reset_index(drop=True)
                        ego_type_str = str(ego_full.loc[0, 'agent_type']) if 'agent_type' in ego_full.columns and len(ego_full) > 0 else 'unknown'

                        # 使用前向差分（当前帧减上一帧），与训练时基于位置差分的速度方向一致
                        ego_full['vx'] = ego_full['pos_x'].diff().fillna(0)
                        ego_full['vy'] = ego_full['pos_y'].diff().fillna(0)
                        # 加速度为速度的一阶差分
                        ego_full['ax'] = ego_full['vx'].diff().fillna(0)
                        ego_full['ay'] = ego_full['vy'].diff().fillna(0)

                        ego_data = ego_full.head(N-2).drop(columns=['agent_ID', 'frame_ID', 'agent_type'])
                        hist_ego = np.array(ego_data.head(ob_horizon)).reshape(ob_horizon, 1, 6)
                        ground_truth_ego = np.array(ego_data.reset_index(drop=True).tail(future_pre)[['pos_x', 'pos_y']]).reshape(future_pre, 1, 2)
                        # 提取自车的历史轨迹和真实轨迹

                        # 构建邻居并对齐到固定的 agent_ID 顺序，以保证时间维度的一致槽位
                        neighbor_df = scene_data[scene_data['agent_ID'] != Possible_ego].reset_index(drop=True).copy()
                        grouped = neighbor_df.groupby('agent_ID')
                        # 邻居同样采用前向差分
                        neighbor_df['vx'] = grouped['pos_x'].diff()
                        neighbor_df['vy'] = grouped['pos_y'].diff()
                        neighbor_df['ax'] = grouped['vx'].diff()
                        neighbor_df['ay'] = grouped['vy'].diff()

                        # 仅保留长度足够的邻居
                        neighbor_df = neighbor_df.sort_values(by=['agent_ID', 'frame_ID'])
                        neighbor_df = neighbor_df.groupby('agent_ID').filter(lambda x: len(x) >= N)

                        # 邻居类型过滤（按 agent_ID 粒度过滤），并统计过滤前后数量
                        if neigh_filter_norm != "any":
                            id2type_nei = neighbor_df.drop_duplicates('agent_ID').set_index('agent_ID')['agent_type'].to_dict()
                            keep_ids = []
                            for aid, at in id2type_nei.items():
                                atn = norm_type(at)
                                match = False
                                if neigh_filter_norm == 'vehicle':
                                    match = (atn == 'vehicle')
                                elif neigh_filter_norm == 'pedestrian':
                                    match = (atn == 'pedestrian')
                                elif neigh_filter_norm == 'bicycle':
                                    match = (atn == 'bicycle')
                                elif neigh_filter_norm == 'vru':
                                    match = (atn in ('pedestrian','bicycle'))
                                if match:
                                    keep_ids.append(aid)
                            before_cnt = len(id2type_nei)
                            neighbor_df = neighbor_df[neighbor_df['agent_ID'].isin(keep_ids)]
                            after_cnt = len(set(keep_ids))
                            print(f"[neighbor-filter] case={idx} target={neigh_filter_norm} before={before_cnt} after={after_cnt}")

                        # 阈值检查：匹配邻居至少 min_matched_neighbors 个
                        matched_ids_now = sorted(neighbor_df['agent_ID'].unique().tolist())
                        if min_matched_neighbors > 0 and len(matched_ids_now) < min_matched_neighbors:
                            # 不满足邻居数量要求，跳过该自车候选
                            #print(f"[neighbor-filter] skip ego {Possible_ego}: matched={len(matched_ids_now)} < {min_matched_neighbors}")
                            continue

                        # 计算对齐所需的帧序列与邻居ID序列
                        frames_all = sorted(scene_data['frame_ID'].unique())
                        frames_use = frames_all[:-2]  # 与自车使用 N-2 帧对齐
                        neighbor_ids = matched_ids_now
                        # 为每个邻居存储类型（如 pedestrian/bicycle / car 等）
                        types_map = neighbor_df.drop_duplicates('agent_ID').set_index('agent_ID')['agent_type'].to_dict()
                        neighbor_types = [str(types_map.get(aid, 'unknown')) for aid in neighbor_ids]

                        # 构建 (T, Nn, 6) 数组，列为 [pos_x, pos_y, vx, vy, ax, ay]
                        time_rows = []
                        for f in frames_use:
                            fdf = neighbor_df[neighbor_df['frame_ID'] == f]
                            if fdf.empty:
                                row_arr = np.zeros((len(neighbor_ids), 6), dtype=np.float64)
                            else:
                                fsel = fdf[['agent_ID', 'pos_x', 'pos_y', 'vx', 'vy', 'ax', 'ay']].set_index('agent_ID')
                                row_arr = np.zeros((len(neighbor_ids), 6), dtype=np.float64)
                                for j, aid in enumerate(neighbor_ids):
                                    if aid in fsel.index:
                                        row_arr[j, :] = fsel.loc[aid].values
                            time_rows.append(row_arr)
                        neighbor_np = np.stack(time_rows, axis=0)  # (T, Nn, 6)
                        neighbor_np = np.expand_dims(neighbor_np, axis=1)  # (T, 1, Nn, 6)

                        neighbor = torch.from_numpy(neighbor_np).double().to(device)
                        x = torch.from_numpy(hist_ego).double().to(device)
                        y = torch.from_numpy(ground_truth_ego).double().to(device)

                        meta = {
                            'ego_id': int(Possible_ego),
                            'ego_type': ego_type_str,
                            'neighbor_ids': neighbor_ids,
                            'neighbor_types': neighbor_types,
                            'src_case_id': case_key,
                            'window_start_frame': int(scene_data['frame_ID'].min()),
                            'sample_index': int(processed_count),
                        }

                        tensors_list.append((x, y, neighbor, meta))
                        processed_count += 1
                        # 将处理后的张量添加到列表中

                        return_list = tensors_list
                        if processed_count >= num_items_to_process:
                            # 已经收集到足够的样本，提前结束
                            break
                    # 打印当前已收集样本数量（可选调试信息）
                    # print("Collected:", len(tensors_list))
                if processed_count >= num_items_to_process:
                    break
        if processed_count >= num_items_to_process:
            print("Break due to sample limit")
            break
    return tensors_list

# --------- Rerank helpers (feasible-area penalty + jerk) ----------
def build_lanelet_paths(laneletmap):
    if laneletmap is None:
        return []
    paths = []
    try:
        for ll in laneletmap.laneletLayer:
            points = [(pt.x, pt.y) for pt in ll.polygon2d()]
            if len(points) >= 3:
                paths.append(Path(points, closed=True))
    except Exception:
        pass
    return paths

def count_out_of_lane(points_xy, lane_paths):
    if not lane_paths:
        return 0
    # points_xy: (T,2) np.array
    # 如果任意一个 lane polygon 包含该点即视为在道内
    out = 0
    for p in points_xy:
        inside = False
        for path in lane_paths:
            if path.contains_point((p[0], p[1])):
                inside = True
                break
        if not inside:
            out += 1
    return out

def jerk_mean(points_xy):
    # points_xy: (T,2)
    if points_xy.shape[0] < 4:
        return 0.0
    v = np.diff(points_xy, axis=0)
    if v.shape[0] < 2:
        return 0.0
    a = np.diff(v, axis=0)
    if a.shape[0] < 2:
        return 0.0
    j = np.diff(a, axis=0)
    jn = np.linalg.norm(j, axis=1)
    return float(jn.mean()) if jn.size > 0 else 0.0

def rerank_indices(y_pred_tensor, y_true_tensor, lane_paths, lambda_lane=0.5, lambda_jerk=0.1):
    """Return list of dicts sorted by composite score: ADE + λ_lane*(out_of_lane/T) + λ_jerk*jerk_mean"""
    S = y_pred_tensor.shape[0]
    T = y_pred_tensor.shape[1]
    ranked = []
    y_true = y_true_tensor.detach().cpu().numpy()[:,0,:]  # (T,2)
    for s in range(S):
        pred = y_pred_tensor[s,:,0,:].detach().cpu().numpy()  # (T,2)
        ade = float(np.linalg.norm(pred - y_true, axis=1).mean())
        out_cnt = count_out_of_lane(pred, lane_paths) if lane_paths is not None else 0
        out_frac = out_cnt / max(1, T)
        jerk = jerk_mean(pred)
        score = ade + lambda_lane * out_frac + lambda_jerk * jerk
        ranked.append({
            'idx': s,
            'score': float(score),
            'ade': float(ade),
            'out_cnt': int(out_cnt),
            'out_frac': float(out_frac),
            'jerk': float(jerk),
        })
    ranked.sort(key=lambda d: d['score'])
    return ranked

def traj_max_turn_degree(points_xy: np.ndarray) -> float:
    """计算相邻段的最大转角（度）。"""
    if points_xy.shape[0] < 3:
        return 0.0
    v = np.diff(points_xy, axis=0)
    # 避免零向量
    eps = 1e-8
    norms = np.maximum(np.linalg.norm(v, axis=1, keepdims=True), eps)
    vn = v / norms
    # 相邻夹角
    dots = (vn[:-1] * vn[1:]).sum(axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    ang = np.degrees(np.arccos(dots))
    return float(np.max(ang)) if ang.size > 0 else 0.0

def out_of_lane_fraction_with_fallback(points_xy: np.ndarray, lane_paths, map_overlay: Optional[dict]) -> float:
    """优先用 lane 多边形计算出道比例；若不可用，退化为基于 veh 栅格统计。"""
    T = max(1, points_xy.shape[0])
    if lane_paths:
        out = count_out_of_lane(points_xy, lane_paths)
        return out / T
    # fallback: vehicle raster
    try:
        if map_overlay is None:
            return 0.0
        veh_img = map_overlay.get('veh_img', None)
        W2M = map_overlay.get('world2map', None)
        if veh_img is None or W2M is None:
            return 0.0
        H, W = veh_img.shape[:2]
        sx = float(W2M[0,0]); tx = float(W2M[0,2])
        sy = float(W2M[1,1]); ty = float(W2M[1,2])
        u = points_xy[:,0] * sx + tx
        v = points_xy[:,1] * sy + ty
        # 取最近的像素
        uu = np.clip(np.round(u).astype(int), 0, W-1)
        vv = np.clip(np.round(v).astype(int), 0, H-1)
        vals = veh_img[vv, uu]
        inside = (vals > 0.5)
        return float((~inside).sum()) / T
    except Exception:
        return 0.0
# 返回处理后的张量列表

#%%
def Vehicle_Viz(ax, centers,angles,width, height,style="r-"):
    """

    :param ax: Matplotlib axis object.
    :param center: The coordinates (x, y) of the center of the vehicle.
    :param width: The width of the vehicle.
    :param height: The height of the vehicle.
    :param angle: Heading angle.
    """
   
    for center, angle in zip(centers, angles):
            # Calculate the half width and half height
            half_width = width / 2
            half_height = height / 2
    
            # Define the four corners of the rectangle based on the center, width, and height
            rectangle = np.array([
                [center[0] - half_width, center[1] - half_height],
                [center[0] + half_width, center[1] - half_height],
                [center[0] + half_width, center[1] + half_height],
                [center[0] - half_width, center[1] + half_height],
                [center[0] - half_width, center[1] - half_height]
            ])
    
            # Convert angle to radians
            theta = np.radians(angle)
    
            # Create rotation matrix
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    
            # Rotate each corner of the rectangle around the center
            rotated_rectangle = np.dot(rectangle - center, rotation_matrix) + center
    
            # Draw the rotated rectangle
            ax.plot(rotated_rectangle[:, 0], rotated_rectangle[:, 1], style)

def calculate_headings(x, y):
    """
    Calculate the bearing angles between consecutive points and extend the last bearing.

    :param x: Array of x-coordinates.
    :param y: Array of y-coordinates.
    :return: Array of heading angles with the same length as x and y.
    """
    # Calculate the differences between consecutive points
    dx = np.diff(x)
    dy = np.diff(y)

    # Calculate the bearing angles
    headings = np.arctan2(dy, dx)

    # Convert headings from radians to degrees
    headings = -np.degrees(headings)

    # Extend the last bearing
    headings = np.append(headings, headings[-1])

    return headings

def plot_trajectory(fig,ax,x,y,y_pred,neighbor,mode,EN_vehicle=True,Length=4,Width=1.8,Style='b-'):
    
    Pos_npred = []  # 初始化预测位置列表
    color_list = ["#00FF37", "#0881c6","#964EEE", '#9AC9DB', '#F8AC8C', '#C82423',
                      '#FF8884', '#8ECFC9',"#F3D266","#B1CE46","#a1a9d0","#F6CAE5",
                      '#F1D77E', '#d76364','#2878B5', '#9AC9DB', '#F8AC8C', '#C82423',
                      '#FF8884', '#8ECFC9',"#F3D266","#B1CE46","#a1a9d0","#F6CAE5",]
          
        
    if mode=="Ego_Pred":
        # Drawing the trajectories
        plt.plot(x[:,0,0].cpu().detach().numpy(),x[:,0,1].cpu().detach().numpy(),
                 color='k', marker='o', markersize=6, markeredgecolor='black', markerfacecolor='k')
        
        plt.plot(y[:,0,0].cpu().detach().numpy(),y[:,0,1].cpu().detach().numpy(),
                 color='k', marker='*', markersize=10, markeredgecolor='black', markerfacecolor='g')
        
        if EN_vehicle:      
            headings=calculate_headings(x[:,0,0].cpu().detach().numpy(),x[:,0,1].cpu().detach().numpy())          
            Vehicle_Viz(ax, list(zip(x[:,0,0].cpu().detach().numpy(),x[:,0,1].cpu().detach().numpy())),headings,Length,Width,Style)
        
        # Loop for predictions
        for N in range(y_pred.cpu().detach().numpy().shape[0]):
            Pos_npred.append([y_pred[N,:,0,0].cpu().detach().numpy(),y_pred[N,:,0,1].cpu().detach().numpy()])
            plt.plot(y_pred[N,:,0,0].cpu().detach().numpy(),y_pred[N,:,0,1].cpu().detach().numpy(),
                     color=color_list[N], marker='o', markersize=6, markeredgecolor='black', markerfacecolor=color_list[N])
    
        plt.title('Trajectory Plot')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend()
        plt.grid(True)
        #plt.show()
        
        
    if mode=="Scenario_Pred":
        # Drawing the trajectories
        
        plt.plot(x[:,0,0].cpu().detach().numpy(),x[:,0,1].cpu().detach().numpy(),
                 color='k', marker='o', markersize=6, markeredgecolor='black', markerfacecolor='k')
        
        plt.plot(y[:,0,0].cpu().detach().numpy(),y[:,0,1].cpu().detach().numpy(),
                 color='k', marker='*', markersize=1, markeredgecolor='black', markerfacecolor='g')
        
        if EN_vehicle:      
            headings=calculate_headings(x[:,0,0].cpu().detach().numpy(),x[:,0,1].cpu().detach().numpy())          
            Vehicle_Viz(ax, list(zip(x[:,0,0].cpu().detach().numpy(),x[:,0,1].cpu().detach().numpy())),headings,Length,Width,Style)
    
        # Loop for predictions
        for N in range(y_pred.cpu().detach().numpy().shape[0]):
            Pos_npred.append([y_pred[N,:,0,0].cpu().detach().numpy(),y_pred[N,:,0,1].cpu().detach().numpy()])
            plt.plot(y_pred[N,:,0,0].cpu().detach().numpy(),y_pred[N,:,0,1].cpu().detach().numpy(),
                     color=color_list[N], marker='o', markersize=1, markeredgecolor='black', markerfacecolor=color_list[N])
            
        neighbor_array= neighbor.cpu().detach().numpy().squeeze() 
      
        for i in range(neighbor_array.shape[1]):
            slice = neighbor_array[:, i, :]
            slice = slice[~(slice == 0).all(axis=1)]
            plt.plot(slice[:,0],slice[:,1], alpha=0.5)
            
            headings=calculate_headings(slice[:,0],slice[:,1])          
            Vehicle_Viz(ax, list(zip(slice[:,0],slice[:,1])),headings,Length,Width,Style)
           
    
            #print(f"Slice {i}:\n", slice)
        
        x_data = x[:, 0, 0].cpu().detach().numpy()
        y_data = x[:, 0, 1].cpu().detach().numpy()
        
        plt.title('Trajectory Plot')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend()
        plt.grid(True)
        #plt.show()

    return Pos_npred,ax



def plot_trajectory_animation(x, y, y_pred, neighbor, laneletmap, mode, output_path, name, EN_vehicle=True, Length=4, Width=1.8, Style='b-', meta=None, danger_flag: bool=False, highlight_indices: Optional[List[int]] = None, map_overlay: Optional[dict] = None):
    # 基于 agent_type 的样式映射
    def agent_style(agent_type: str):
        s = str(agent_type).lower()
        # INTERACTION 验证集中常见为 'pedestrian/bicycle'，此处统一按行人风格绘制为圆点
        if 'pedestrian' in s or 'walker' in s or 'pedestrian/bicycle' in s:
            return dict(kind='dot', color='#2ca02c', length=0.5, width=0.5, label='pedestrian/bicycle')
        if 'bicycle' in s or 'cyclist' in s or 'bike' in s:
            return dict(kind='rect', color='#1f77b4', length=2.0, width=0.6, label='bicycle')
        # 其它均按机动车处理
        return dict(kind='rect', color='#333333', length=4.0, width=1.8, label='vehicle')
    fig, ax = plt.subplots(figsize=(10.24, 7.68))
    # 预测轨迹颜色顺序（对应重排名后的 top-1/2/3）：#00FF37、#0881c6、#964EEE
    color_list = ["#00FF37", "#0881c6", "#964EEE", '#9AC9DB', '#F8AC8C', '#C82423',
                  '#FF8884', '#8ECFC9',"#F3D266","#B1CE46","#a1a9d0","#F6CAE5"] * 2
    
    # Assuming x and y are tensors with shape [num_frames, num_vehicles, 2]
    Ego_his_x = x[:,0,0].cpu().detach().numpy()  # Convert the tensor to numpy for plotting
    Ego_his_y = x[:,0,1].cpu().detach().numpy()
    Ego_future_x = y[:,0,0].cpu().detach().numpy()
    Ego_future_y = y[:,0,1].cpu().detach().numpy()
    y_pred=y_pred.cpu().detach().numpy()
    
    # 将邻居张量规范为 (T, Nn, 6) 形状：原始为 (T, 1, Nn, 6)。
    _nei_arr = neighbor.cpu().detach().numpy()
    if _nei_arr.ndim == 4:
        # (T, 1, Nn, 6) -> (T, Nn, 6)
        neighbor_array = _nei_arr[:, 0, :, :]
    elif _nei_arr.ndim == 3 and _nei_arr.shape[-1] == 6:
        # 已经是 (T, Nn, 6)
        neighbor_array = _nei_arr
    elif _nei_arr.ndim == 2 and _nei_arr.shape[-1] == 6:
        # (T, 6) 表示仅有 1 个邻居 -> (T, 1, 6)
        neighbor_array = _nei_arr[:, None, :]
    else:
        # 兜底：尽力推断为 (T, Nn, 6)
        try:
            T = _nei_arr.shape[0]
            neighbor_array = _nei_arr.reshape(T, -1, 6)
        except Exception:
            # 无法解析则置空，后续分支将跳过邻居绘制
            neighbor_array = np.zeros((0, 0, 6), dtype=float)
    neighbor_types = []
    if meta is not None and isinstance(meta, dict):
        neighbor_types = meta.get('neighbor_types', [])
    ego_type = meta.get('ego_type', 'vehicle') if isinstance(meta, dict) else 'vehicle'
    neighbor_ids = meta.get('neighbor_ids', []) if isinstance(meta, dict) else []
    ego_id = meta.get('ego_id') if isinstance(meta, dict) else None
    

    def _draw_overlays(ax_handle):
        try:
            if map_overlay is None:
                return
            world2map = map_overlay.get('world2map', None)
            if world2map is None:
                return
            # choose a base image to get size
            base_img = None
            for key in ['vru_img', 'veh_img', 'keepout_img']:
                im = map_overlay.get(key, None)
                if im is not None:
                    base_img = im
                    break
            if base_img is None:
                return
            H, W = base_img.shape[:2]
            # invert affine for extent
            sx = float(world2map[0, 0]); tx = float(world2map[0, 2])
            sy = float(world2map[1, 1]); ty = float(world2map[1, 2])
            xmin = (0.0 - tx) / (sx if abs(sx) > 1e-12 else 1.0)
            xmax = ((W - 1.0) - tx) / (sx if abs(sx) > 1e-12 else 1.0)
            ymin = (0.0 - ty) / (sy if abs(sy) > 1e-12 else 1.0)
            ymax = ((H - 1.0) - ty) / (sy if abs(sy) > 1e-12 else 1.0)
            extent = [xmin, xmax, ymin, ymax]
            alpha_all = float(map_overlay.get('alpha', 0.28))
            # Draw order: keepout (bottom) -> vehicle -> VRU (top)
            if bool(map_overlay.get('enable_keepout', True)):
                k_img = map_overlay.get('keepout_img', None)
                if k_img is not None and np.asarray(k_img).sum() > 0:
                    ax_handle.imshow(k_img, cmap='Reds', alpha=alpha_all*0.9, origin='lower', extent=extent, zorder=1, interpolation='nearest')
            if bool(map_overlay.get('enable_veh', True)):
                veh_img = map_overlay.get('veh_img', None)
                if veh_img is not None and np.asarray(veh_img).sum() > 0:
                    ax_handle.imshow(veh_img, cmap='Blues', alpha=alpha_all*0.8, origin='lower', extent=extent, zorder=2, interpolation='nearest')
            if bool(map_overlay.get('enable_vru', True)):
                vru_img = map_overlay.get('vru_img', None)
                if vru_img is not None and np.asarray(vru_img).sum() > 0:
                    ax_handle.imshow(vru_img, cmap='Greens', alpha=alpha_all, origin='lower', extent=extent, zorder=3, interpolation='nearest')
        except Exception as _e:
            print(f"[WARN] overlay draw failed: {_e}")

    def _add_overlay_legend(ax_handle):
        try:
            if map_overlay is None or not bool(map_overlay.get('legend', True)):
                return
            from matplotlib.patches import Patch
            handles = []
            alpha_all = float(map_overlay.get('alpha', 0.28))
            any_added = False
            if bool(map_overlay.get('enable_keepout', True)) and map_overlay.get('keepout_img', None) is not None and np.asarray(map_overlay['keepout_img']).sum() > 0:
                handles.append(Patch(facecolor='#e74c3c', alpha=min(alpha_all*0.9,1.0), label='keepout'))
                any_added = True
            if bool(map_overlay.get('enable_veh', True)) and map_overlay.get('veh_img', None) is not None and np.asarray(map_overlay['veh_img']).sum() > 0:
                handles.append(Patch(facecolor='#3498db', alpha=min(alpha_all*0.8,1.0), label='vehicle lane'))
                any_added = True
            if bool(map_overlay.get('enable_vru', True)) and map_overlay.get('vru_img', None) is not None and np.asarray(map_overlay['vru_img']).sum() > 0:
                handles.append(Patch(facecolor='#2ecc71', alpha=alpha_all, label='VRU area'))
                any_added = True
            if any_added:
                # 将叠加层图例追加，不替换既有 agent 图例
                leg = ax_handle.legend(handles=handles, loc='upper right', framealpha=0.8)
                leg.set_zorder(10)
        except Exception as _e:
            print(f"[WARN] overlay legend failed: {_e}")

    def init():
        ax.clear()
        ax.set_title('Trajectory Plot')
        #ax.set_xlim(np.min(Ego_x)-10, np.max(Ego_x)+10)
        #ax.set_ylim(np.min(Ego_his_y)-10, np.max(Ego_his_y)+10)
        if laneletmap is not None:
            Lanelet_Map_Viz.draw_lanelet_map(laneletmap, ax)
        # 绘制叠加层（VRU/vehicle/keepout）
        _draw_overlays(ax)
        ax.grid(True)
        
        # 绘制邻居完整轨迹（静态底图部分）并准备图例
        legend_used = {}
        hi_set = set(highlight_indices or [])
        for i in range(neighbor_array.shape[1]):
            slice = neighbor_array[:, i, :]
            slice = slice[~(slice == 0).all(axis=1)]
            if slice.shape[0] == 0:
                continue
            atype = neighbor_types[i] if i < len(neighbor_types) else 'vehicle'
            sty = agent_style(atype)
            # 非高亮邻居使用原色，高亮邻居在更新阶段着色更醒目
            if sty['kind'] == 'dot':
                ax.plot(slice[:, 0], slice[:, 1], linestyle='-', color=sty['color'], alpha=0.6, linewidth=1)
                ax.scatter(slice[:, 0], slice[:, 1], s=8, c=sty['color'], alpha=0.8)
            else:
                ax.plot(slice[:, 0], slice[:, 1], linestyle='-', color=sty['color'], alpha=0.7, linewidth=1)
                headings = calculate_headings(slice[:, 0], slice[:, 1])
                Vehicle_Viz(ax, list(zip(slice[:, 0], slice[:, 1])), headings, sty['length'], sty['width'], sty['color'])
            legend_used[sty['label']] = sty['color']
        ax.set_xlabel('X/m')
        ax.set_ylabel('Y/m')
        ax.set_title('Accident Prone Trajectory Generation')

        # 添加图例
        if legend_used:
            from matplotlib.lines import Line2D
            handles = []
            for label, color in legend_used.items():
                if label == 'pedestrian/bicycle':
                    handles.append(Line2D([0], [0], marker='o', color=color, label=label, markersize=6, linestyle='None'))
                else:
                    handles.append(Line2D([0], [0], color=color, label=label, linewidth=2))
            ax.legend(handles=handles, loc='best')
        # 危险样本标注
        if danger_flag:
            ax.text(0.02, 0.98, 'DANGEROUS', transform=ax.transAxes, fontsize=12, color='red',
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))
        return ax,

    def update(frame):
        ax.clear()
        ax.grid(True)
        if laneletmap is not None:
            Lanelet_Map_Viz.draw_lanelet_map(laneletmap, ax)
        _draw_overlays(ax)

        ego_sty = agent_style(ego_type)
        if frame <= len(Ego_his_x):
            ax.plot(Ego_his_x[0:frame], Ego_his_y[0:frame], linestyle='-', color='k', markersize=1)
            if EN_vehicle:
                if ego_sty['kind'] == 'dot':
                    c = '#ff0000' if danger_flag else ego_sty['color']
                    ax.scatter(Ego_his_x[max(0, frame-1):frame], Ego_his_y[max(0, frame-1):frame], s=20, c=c)
                    # 标注自车ID
                    if ego_id is not None and frame > 0:
                        ax.text(Ego_his_x[frame-1] + 0.5, Ego_his_y[frame-1] + 0.5, f"{ego_id}", color=c, fontsize=9)
                else:
                    headings = calculate_headings(Ego_his_x, Ego_his_y)
                    Vehicle_Viz(ax,
                                list(zip(Ego_his_x[max(0, frame-1):frame], Ego_his_y[max(0, frame-1):frame])),
                                headings[max(0, frame-1):frame], ego_sty['length'], ego_sty['width'], ('#ff0000' if danger_flag else ego_sty['color']))
                    if ego_id is not None and frame > 0:
                        ax.text(Ego_his_x[frame-1] + 0.5, Ego_his_y[frame-1] + 0.5, f"{ego_id}", color=('#ff0000' if danger_flag else ego_sty['color']), fontsize=9)
        else:
            ax.plot(Ego_his_x, Ego_his_y, linestyle='-', color='k', markersize=1)
            ax.plot(Ego_future_x[0:frame-len(Ego_his_x)], Ego_future_y[0:frame-len(Ego_his_x)], color='pink', markersize=1)
            if EN_vehicle:
                headings = calculate_headings(Ego_future_x, Ego_future_y)
                if ego_sty['kind'] == 'dot':
                    c = '#ff0000' if danger_flag else ego_sty['color']
                    ax.scatter(Ego_future_x[max(0, frame-1-len(Ego_his_x)):frame-len(Ego_his_x)],
                               Ego_future_y[max(0, frame-1-len(Ego_his_x)):frame-len(Ego_his_x)], s=20, c=c)
                    if ego_id is not None and (frame - len(Ego_his_x)) > 0:
                        idx = frame - len(Ego_his_x) - 1
                        ax.text(Ego_future_x[idx] + 0.5, Ego_future_y[idx] + 0.5, f"{ego_id}", color=c, fontsize=9)
                else:
                    Vehicle_Viz(
                        ax,
                        list(zip(Ego_future_x[max(0, frame-1-len(Ego_his_x)):frame-len(Ego_his_x)],
                                 Ego_future_y[max(0, frame-1-len(Ego_his_x)):frame-len(Ego_his_x)])),
                        headings[max(0, frame-1-len(Ego_his_x)):frame-len(Ego_his_x)],
                        ego_sty['length'], ego_sty['width'], ('#ff0000' if danger_flag else 'pink')
                    )
                    if ego_id is not None and (frame - len(Ego_his_x)) > 0:
                        idx = frame - len(Ego_his_x) - 1
                        ax.text(Ego_future_x[idx] + 0.5, Ego_future_y[idx] + 0.5, f"{ego_id}", color=('#ff0000' if danger_flag else 'pink'), fontsize=9)
            for N in range(y_pred.shape[0]):
                plt.plot(y_pred[N,0:frame-len(Ego_his_x),0,0],
                         y_pred[N,0:frame-len(Ego_his_x),0,1],
                         color=color_list[N], marker='o', markersize=1, markerfacecolor=color_list[N])
                headings=calculate_headings(y_pred[N,0:frame,0,0],y_pred[N,0:frame,0,1])
            
                Vehicle_Viz(ax, 
                            list(zip(y_pred[N,max(0,frame-1-len(Ego_his_x)):frame-len(Ego_his_x),0,0],
                                     y_pred[N,max(0,frame-1-len(Ego_his_x)):frame-len(Ego_his_x),0,1])),
                            headings[max(0,frame-1-len(Ego_his_x)):frame-len(Ego_his_x)],
                            Length,
                            Width,
                            color_list[N])
        
        
        hi_set = set(highlight_indices or [])
        for i in range(neighbor_array.shape[1]):
            slice = neighbor_array[:, i, :]
            slice = slice[~(slice == 0).all(axis=1)]
            if slice.shape[0] == 0:
                continue
            atype = neighbor_types[i] if i < len(neighbor_types) else 'vehicle'
            sty = agent_style(atype)
            if sty['kind'] == 'dot':
                # 仅对高亮邻居使用强调色
                c = '#ff6600' if (danger_flag and (i in hi_set)) else sty['color']
                ax.plot(slice[0:frame, 0], slice[0:frame, 1], linestyle='-', color=c, linewidth=1)
                ax.scatter(slice[frame-1:frame, 0], slice[frame-1:frame, 1], s=16, c=c)
                # 邻居ID标注（仅高亮）
                if (i in hi_set) and frame > 0 and i < len(neighbor_ids):
                    ax.text(slice[min(frame-1, slice.shape[0]-1), 0] + 0.5,
                            slice[min(frame-1, slice.shape[0]-1), 1] + 0.5,
                            f"{neighbor_ids[i]}", color=c, fontsize=9)
            else:
                headings = calculate_headings(slice[:, 0], slice[:, 1])[max(0, frame-1):frame]
                c = '#ff6600' if (danger_flag and (i in hi_set)) else sty['color']
                Vehicle_Viz(ax, list(zip(slice[max(0, frame-1):frame, 0], slice[max(0, frame-1):frame, 1])),
                            headings, sty['length'], sty['width'], c)
                ax.plot(slice[0:frame, 0], slice[0:frame, 1], linestyle='-', color=c, linewidth=1)
                if (i in hi_set) and frame > 0 and i < len(neighbor_ids):
                    ax.text(slice[min(frame-1, slice.shape[0]-1), 0] + 0.5,
                            slice[min(frame-1, slice.shape[0]-1), 1] + 0.5,
                            f"{neighbor_ids[i]}", color=c, fontsize=9)
        
        ax.set_xlabel('X/m')
        ax.set_ylabel('Y/m')
        ax.set_title('Accident Prone Trajectory Generation')
        # 叠加层图例（可选）
        _add_overlay_legend(ax)
        # 危险样本标注
        if danger_flag:
            ax.text(0.02, 0.98, 'DANGEROUS', transform=ax.transAxes, fontsize=12, color='red',
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))
        return ax,

    total_frames = len(Ego_his_x)+len(Ego_future_x)
    # 支持帧下采样
    frame_skip = int(getattr(parser.parse_args(), 'gif_frame_skip', 1)) if 'parser' in globals() else 1
    frames = list(range(0, total_frames, max(1, frame_skip)))
    # 如果启用 no-gif，直接返回，不落盘（提前，不创建动画对象，避免 UserWarning）
    args = parser.parse_args()
    if getattr(args, 'no_gif', False):
        plt.close(fig)
        return
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False)
    ani.save(os.path.join(output_path, name), writer='pillow', fps=int(getattr(args, 'gif_fps', 10)), dpi=int(getattr(args, 'gif_dpi', 100)))

#%%

if __name__ == "__main__":
    
    
    parser.add_argument("--lat_origin", type=float,
                        help="Latitude of the reference point for the projection of the lanelet map (float)",
                        default=0.0, nargs="?")
    parser.add_argument("--lon_origin", type=float,
                        help="Longitude of the reference point for the projection of the lanelet map (float)",
                        default=0.0, nargs="?")
    
    settings = parser.parse_args()
    spec = importlib.util.spec_from_file_location("config", settings.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    if settings.device is None:
        settings.device = "cuda" if torch.cuda.is_available() else "cpu"
    settings.device = torch.device(settings.device)
    
    seed(settings.seed)
    init_rng_state = get_rng_state(settings.device)
    rng_state = init_rng_state

    ###############################################################################
    #####                                                                    ######
    ##### prepare datasets                                                   ######
    #####                                                                    ######
    ###############################################################################
    kwargs = dict(
            batch_first=False, frameskip=settings.frameskip,
            ob_horizon=config.OB_HORIZON, pred_horizon=config.PRED_HORIZON,
            device=settings.device, seed=settings.seed)
    train_data, test_data = None, None
    if settings.test:
        #print(settings.test)
        if config.INCLUSIVE_GROUPS is not None:
            inclusive = [config.INCLUSIVE_GROUPS for _ in range(len(settings.test))]
        else:
            inclusive = None
            
        #settings.test ---> File location
        
        test_dataset = Dataloader(
            settings.test, **kwargs, inclusive_groups=inclusive,
            batch_size=config.BATCH_SIZE, shuffle=False
        )
        
        test_data = torch.utils.data.DataLoader(test_dataset, 
            collate_fn=test_dataset.collate_fn,
            batch_sampler=test_dataset.batch_sampler
        )
             


    ###############################################################################
    #####                                                                    ######
    ##### load model                                                         ######
    #####                                                                    ######
    ###############################################################################
    model = SocialVAE(horizon=config.PRED_HORIZON, ob_radius=config.OB_RADIUS, hidden_dim=config.RNN_HIDDEN_DIM)
    model.to(settings.device)
    # 将训练期的结构/风险/损失配置应用到模型，以便与 ckpt-best 的权重结构对齐
    # 1) 启用 Mamba 与多头邻居注意力（需在 load_state_dict 前创建相应子模块）
    try:
        if hasattr(config, 'USE_MAMBA_ENCODER') or hasattr(config, 'USE_MAMBA_DECODER'):
            enc_on = getattr(config, 'USE_MAMBA_ENCODER', False)
            dec_on = getattr(config, 'USE_MAMBA_DECODER', False)
            d_enc = getattr(config, 'MAMBA_D_MODEL_ENC', None)
            d_dec = getattr(config, 'MAMBA_D_MODEL_DEC', None)
            model.enable_mamba(encoder=enc_on, decoder=dec_on, d_model_enc=d_enc, d_model_dec=d_dec)
        if getattr(config, 'MHA_HEADS', 1) and getattr(config, 'MHA_HEADS', 1) > 1:
            model.enable_multihead_attention(heads=config.MHA_HEADS, dropout=getattr(config, 'MHA_DROPOUT', 0.0))
        # 新增模块是在 .to() 之后创建的，需再次迁移到目标设备
        model.to(settings.device)
    except Exception as e:
        print("[WARN] 启用 Mamba/MHA 失败：", e)
    # 2) 风险参数（含可学习组件权重 / log-sigma / OBB 等）
    try:
        risk_kwargs = dict(
            enable=getattr(config, 'RISK_ENABLE', False),
            weight=getattr(config, 'RISK_WEIGHT', 0.0),
            risk_global_scale=getattr(config, 'RISK_GLOBAL_SCALE', 1.0),
            component_weights=getattr(config, 'RISK_COMPONENT_WEIGHTS', {}),
            learn_component_weights=getattr(config, 'RISK_LEARN_COMPONENT_WEIGHTS', False),
            learn_component_norm=getattr(config, 'RISK_LEARN_COMPONENT_NORM', 'none'),
            compw_entropy_lambda=getattr(config, 'RISK_COMPW_ENTROPY_LAMBDA', 0.0),
            beta=getattr(config, 'RISK_MIN_DIST_BETA', 2.0),
            ttc_tau=getattr(config, 'RISK_TTC_TAU', 1.5),
            # PET
            pet_dist_th=getattr(config, 'RISK_PET_DIST_TH', 2.0),
            pet_alpha=getattr(config, 'RISK_PET_ALPHA', 8.0),
            pet_beta=getattr(config, 'RISK_PET_BETA', 0.7),
            pet_gamma=getattr(config, 'RISK_PET_GAMMA', 2.0),
            pet_continuous=getattr(config, 'RISK_PET_CONTINUOUS', True),
            pet_time_temp=getattr(config, 'RISK_PET_TIME_TEMP', 4.0),
            enable_pet=getattr(config, 'RISK_ENABLE_PET', True),
            # Overlap/OBB
            enable_overlap=getattr(config, 'RISK_ENABLE_OVERLAP', True),
            ov_use_obb=getattr(config, 'RISK_OV_USE_OBB', False),
            ov_self_length=getattr(config, 'RISK_OV_SELF_LENGTH', 4.5),
            ov_self_width=getattr(config, 'RISK_OV_SELF_WIDTH', 1.8),
            ov_neigh_length=getattr(config, 'RISK_OV_NEIGH_LENGTH', 4.5),
            ov_neigh_width=getattr(config, 'RISK_OV_NEIGH_WIDTH', 1.8),
            ov_min_speed=getattr(config, 'RISK_OV_MIN_SPEED', 1e-3),
            ov_axis_beta=getattr(config, 'RISK_OV_AXIS_BETA', 12.0),
            ov_debug=getattr(config, 'RISK_OV_DEBUG', False),
            # log-sigma
            use_log_sigma=getattr(config, 'RISK_USE_LOG_SIGMA', False),
            log_sigma_penalty_w=getattr(config, 'RISK_LOG_SIGMA_PENALTY_W', 0.01),
        )
        model.set_risk_params(**risk_kwargs)
    except Exception as e:
        print("[WARN] 应用风险配置失败：", e)
    # 3) 主损失权重/组合开关（供 loss() 读取）
    for name in [
        'LOSS_W_REC','LOSS_W_WMSE','LOSS_W_KL','LOSS_W_ADV','LOSS_W_KIN',
        'LOSS_COMBINE_REC_WMSE','LOSS_REC_WMSE_ALPHA'
    ]:
        if hasattr(config, name):
            setattr(model, name, getattr(config, name))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    start_epoch = 0
   
    if settings.ckpt:
        # 直接按 POSIX 路径拼接，避免 Windows 风格替换
        ckpt_dir = settings.ckpt if os.path.isabs(settings.ckpt) else os.path.join(os.getcwd(), settings.ckpt)
        ckpt = os.path.join(ckpt_dir, "ckpt-last")
        ckpt_best = os.path.join(ckpt_dir, "ckpt-best")
        if os.path.exists(ckpt_best):
            state_dict = torch.load(ckpt_best, map_location=settings.device)
            ade_best = state_dict["ade"]
            fde_best = state_dict["fde"]
            fpc_best = state_dict["fpc"] if "fpc" in state_dict else 1
        else:
            ade_best = 100000
            fde_best = 100000
            fpc_best = 1
        if train_data is None: # testing mode
            ckpt = ckpt_best
        if os.path.exists(ckpt):
            print("Load from ckpt:", ckpt)
            state_dict = torch.load(ckpt, map_location=settings.device)
            # 若训练结构启用了 Mamba/MHA/风险可学习权重，则需在 load 前已创建对应子模块（上面已处理）
            # 为最大兼容性，如仍有轻微不匹配，放宽 strict
            try:
                model.load_state_dict(state_dict["model"], strict=True)
            except Exception as e:
                print("[WARN] 严格加载失败，尝试非严格加载：", e)
                missing, unexpected = model.load_state_dict(state_dict["model"], strict=False)
                if missing:
                    print("Missing keys:", missing)
                if unexpected:
                    print("Unexpected keys:", unexpected)
            # 再次确保所有权重与新建子模块在同一设备
            model.to(settings.device)
            if "optimizer" in state_dict:
                optimizer.load_state_dict(state_dict["optimizer"])
                rng_state = [r.to("cpu") if torch.is_tensor(r) else r for r in state_dict["rng_state"]]
            start_epoch = state_dict["epoch"]
     
    #end_epoch = start_epoch+1 if train_data is None or start_epoch >= config.EPOCHS else config.EPOCHS
    #ade, fde = test(model, fpc)
    model.eval()
   
   
#%%
    # Define the file path
    agent_threshold = 5
    ob_horizon = 10
    future_pre = 25
       
    agent_threshold = 5
    Mode_select=["train","val"]
    current_directory = os.getcwd()

    # 使用命令行参数指定的 CSV
    csv_path = settings.csv if os.path.isabs(settings.csv) else os.path.join(current_directory, settings.csv)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到指定的 CSV: {csv_path}")

    print(f"Loading data from: {csv_path}")
    data = pd.read_csv(csv_path)
    
    #print(data.head())
    print(data.info())
    grouped_data = data.groupby('case_id')
    
    # 处理数据
    if hasattr(settings, 'ego_type_filter'):
        print(f"[ego-type-filter] active: {settings.ego_type_filter}")
        ego_filter = settings.ego_type_filter
    else:
        ego_filter = "vehicle"
    if hasattr(settings, 'neighbor_type_filter'):
        print(f"[neighbor-type-filter] active: {settings.neighbor_type_filter}")
        neigh_filter = settings.neighbor_type_filter
    else:
        neigh_filter = "any"
    min_nei = getattr(settings, 'min_matched_neighbors', 0)
    print(f"[neighbor-min] threshold: {min_nei}")
    tensor_data=process_data_to_tensors(
        data, agent_threshold, ob_horizon, future_pre, settings.device,
        max_cases=settings.max_cases,
        ego_type_filter=ego_filter,
        neighbor_type_filter=neigh_filter,
        min_matched_neighbors=min_nei,
    )
    #print(tensor_data)
    try:
        print(f"[DATA] collected samples: {len(tensor_data)}")
    except Exception:
        print("[DATA] collected samples: <unknown>")


#%%
    model.double()
    # 预初始化元信息变量，避免静态分析器未定义报错；循环内会覆盖
    meta = {}
    md_all = {}

    # ================= VRU 可行区叠加层（基于 .osm_xy 的全局 world2map + 栅格） =================
    def _load_osm_xy_polygons(path: str):
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            # nodes
            nodes = {}
            for n in root.findall('.//node'):
                nid = int(n.attrib['id'])
                x = float(n.attrib.get('x', n.attrib.get('lon', '0.0')))
                y = float(n.attrib.get('y', n.attrib.get('lat', '0.0')))
                nodes[nid] = (x, y)
            # ways
            ways = {}
            for w in root.findall('.//way'):
                wid = int(w.attrib['id'])
                nds = [int(nd.attrib['ref']) for nd in w.findall('nd')]
                ways[wid] = nds
            lanelet_polys, keepout_polys, vru_polys = [], [], []
            for r in root.findall('.//relation'):
                tags = {t.attrib['k']: t.attrib['v'] for t in r.findall('tag')}
                rtype = tags.get('type', '')
                if rtype == 'lanelet':
                    left_id = right_id = None
                    for m in r.findall('member'):
                        if m.attrib.get('role') == 'left' and m.attrib.get('type') == 'way':
                            left_id = int(m.attrib['ref'])
                        elif m.attrib.get('role') == 'right' and m.attrib.get('type') == 'way':
                            right_id = int(m.attrib['ref'])
                    if left_id in ways and right_id in ways:
                        left_pts = [nodes[nid] for nid in ways[left_id] if nid in nodes]
                        right_pts = [nodes[nid] for nid in ways[right_id] if nid in nodes]
                        if len(left_pts) >= 2 and len(right_pts) >= 2:
                            poly = np.array(left_pts + right_pts[::-1], dtype=np.float32)
                            subtype = tags.get('subtype', '').lower()
                            if subtype in ('crosswalk','footway','sidewalk','pedestrian','pedestrian_area'):
                                vru_polys.append(poly)
                            else:
                                lanelet_polys.append(poly)
                elif rtype == 'multipolygon':
                    subtype = tags.get('subtype', '').lower()
                    if subtype == 'keepout':
                        for m in r.findall('member'):
                            if m.attrib.get('role') == 'outer' and m.attrib.get('type') == 'way':
                                wid = int(m.attrib['ref'])
                                if wid in ways:
                                    pts = [nodes[nid] for nid in ways[wid] if nid in nodes]
                                    if len(pts) >= 3:
                                        keepout_polys.append(np.array(pts, dtype=np.float32))
                    elif subtype in ('crosswalk','footway','sidewalk','pedestrian','pedestrian_area'):
                        for m in r.findall('member'):
                            if m.attrib.get('role') == 'outer' and m.attrib.get('type') == 'way':
                                wid = int(m.attrib['ref'])
                                if wid in ways:
                                    pts = [nodes[nid] for nid in ways[wid] if nid in nodes]
                                    if len(pts) >= 3:
                                        vru_polys.append(np.array(pts, dtype=np.float32))
            return lanelet_polys, keepout_polys, vru_polys
        except Exception as e:
            print(f"[WARN] parse .osm_xy failed: {e}")
            return None, None, None

    def _build_global_map_raster(bounds, lanelet_polys, keepout_polys, vru_polys, raster_size=(224,224)):
        try:
            if bounds is None or lanelet_polys is None:
                return None, None, None, None, None
            minx, miny, maxx, maxy = bounds
            H, W = int(raster_size[0]), int(raster_size[1])
            sx = (W - 1) / max(1e-6, (maxx - minx))
            sy = (H - 1) / max(1e-6, (maxy - miny))
            tx = -minx * sx
            ty = -miny * sy
            world2map = np.array([[sx, 0.0, tx], [0.0, sy, ty]], dtype=np.float32)
            # grid points (pixel centers)
            uu = np.arange(W, dtype=np.float32)
            vv = np.arange(H, dtype=np.float32)
            U, V = np.meshgrid(uu, vv)
            pts = np.stack([U.ravel(), V.ravel()], axis=1)
            def tf_poly(poly):
                xy = np.asarray(poly, dtype=np.float32)
                u = xy[:,0]*world2map[0,0] + xy[:,1]*world2map[0,1] + world2map[0,2]
                v = xy[:,0]*world2map[1,0] + xy[:,1]*world2map[1,1] + world2map[1,2]
                return np.stack([u, v], axis=1)
            try:
                from matplotlib.path import Path as MplPath
            except Exception:
                MplPath = None
            def pip_numpy(poly_uv: np.ndarray, pts_uv: np.ndarray) -> np.ndarray:
                x = pts_uv[:,0]; y = pts_uv[:,1]
                n = poly_uv.shape[0]
                inside = np.zeros(pts_uv.shape[0], dtype=bool)
                xj, yj = poly_uv[-1,0], poly_uv[-1,1]
                for i in range(n):
                    xi, yi = poly_uv[i,0], poly_uv[i,1]
                    intersect = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
                    inside ^= intersect
                    xj, yj = xi, yi
                return inside
            # vehicle mask（lanelet 区域减去 keepout）
            allowed = np.zeros((H*W,), dtype=bool)
            for poly in (lanelet_polys or []):
                p_uv = tf_poly(poly)
                if MplPath is not None:
                    allowed |= MplPath(p_uv, closed=True).contains_points(pts)
                else:
                    allowed |= pip_numpy(p_uv, pts)
            # keepout union mask
            keep_union = np.zeros((H*W,), dtype=bool)
            for poly in (keepout_polys or []):
                p_uv = tf_poly(poly)
                if MplPath is not None:
                    ko = MplPath(p_uv, closed=True).contains_points(pts)
                else:
                    ko = pip_numpy(p_uv, pts)
                keep_union |= ko
            allowed &= ~keep_union
            veh_mask = allowed.reshape(H, W).astype(np.float32)
            keepout_mask = keep_union.reshape(H, W).astype(np.float32)
            # VRU mask
            if vru_polys is not None and len(vru_polys) > 0:
                allowed_vru = np.zeros((H*W,), dtype=bool)
                for poly in (vru_polys or []):
                    p_uv = tf_poly(poly)
                    if MplPath is not None:
                        allowed_vru |= MplPath(p_uv, closed=True).contains_points(pts)
                    else:
                        allowed_vru |= pip_numpy(p_uv, pts)
                allowed_vru &= ~keep_union
                vru_mask = allowed_vru.reshape(H, W).astype(np.float32)
            else:
                vru_mask = np.zeros_like(veh_mask, dtype=np.float32)
            return vru_mask, veh_mask, keepout_mask, world2map, (minx, miny, maxx, maxy)
        except Exception as e:
            print(f"[WARN] build map raster failed: {e}")
            return None, None, None, None, None

    def _compute_map_bounds(lanelet_polys, keepout_polys, vru_polys):
        xs, ys = [], []
        for coll in [lanelet_polys, vru_polys, keepout_polys]:
            if coll is None:
                continue
            for poly in coll:
                if poly is None or len(poly) == 0:
                    continue
                xs.append(np.asarray(poly)[:,0]); ys.append(np.asarray(poly)[:,1])
        if len(xs) == 0:
            return None
        minx = float(np.min([x.min() for x in xs])); maxx = float(np.max([x.max() for x in xs]))
        miny = float(np.min([y.min() for y in ys])); maxy = float(np.max([y.max() for y in ys]))
        m = 1.0
        return (minx - m, miny - m, maxx + m, maxy + m)

    map_overlay = None
    try:
        osm_xy_path = getattr(config, 'MAP_OSM_XY_PATH', None)
        raster_size = getattr(config, 'MAP_RASTER_SIZE', (224, 224))
        if osm_xy_path and os.path.exists(osm_xy_path):
            ll_polys, ko_polys, vru_polys = _load_osm_xy_polygons(osm_xy_path)
            bounds = _compute_map_bounds(ll_polys, ko_polys, vru_polys)
            vru_img, veh_img, keepout_img, world2map_np, _ = _build_global_map_raster(bounds, ll_polys, ko_polys, vru_polys, raster_size=raster_size)
            if world2map_np is not None:
                map_overlay = {
                    'vru_img': vru_img,               # HxW float32 array in [0,1]
                    'veh_img': veh_img,
                    'keepout_img': keepout_img,
                    'world2map': world2map_np,         # 2x3 float32
                    'enable_vru': bool(getattr(settings, 'overlay_vru', True)),
                    'enable_veh': bool(getattr(settings, 'overlay_veh', True)),
                    'enable_keepout': bool(getattr(settings, 'overlay_keepout', True)),
                    'alpha': float(getattr(settings, 'overlay_alpha', 0.28)),
                    'legend': bool(getattr(settings, 'overlay_legend', True)),
                }
                shape_str = tuple(vru_img.shape) if vru_img is not None else (None,)
                print(f"[OVERLAY] Built raster for GIFs: vru={shape_str} veh={(None if veh_img is None else veh_img.shape)} keepout={(None if keepout_img is None else keepout_img.shape)}")
            else:
                print("[OVERLAY] No raster available (parse failed or empty).")
        else:
            print("[OVERLAY] MAP_OSM_XY_PATH not set or file missing; skip overlay.")
    except Exception as e:
        print(f"[OVERLAY] Failed to prepare overlay: {e}")
    
    for num in range(len(tensor_data)):
        # for num in range(10):
        # num=8
        x = tensor_data[num][0]
        y = tensor_data[num][1]
        neighbor = tensor_data[num][2]
        meta = tensor_data[num][3] if len(tensor_data[num]) >= 4 else None
        md_all = meta if isinstance(meta, dict) else {}
        """
        Exp_L2= []
        for i in range(10):
            neighbor_array=neighbor[-25-1:-1,:,i,0:2].cpu().detach()
            ego_pre_array=y.cpu().detach()
            print(ego_pre_array-neighbor_array)
            Exp_L2.append((np.exp(-np.linalg.norm((ego_pre_array-neighbor_array), axis=2))))
        
        delta = [item / np.sum(Exp_L2) for item in Exp_L2]
        
        L_adv=[]
        
        for i in range(neighbor.shape[2]):
           
           neighbor_array=neighbor[-25-1:-1,:,i,0:2].cpu().detach()
           ego_pre_array=y.cpu().detach()
           
           L_adv.append(np.sum(delta[i]*np.linalg.norm((ego_pre_array-neighbor_array), axis=2)))
           a=L_adv
            #Exp_L2.append(sum(np.exp(-np.linalg.norm((ego_pre_array-neighbor_array), axis=1))))
        L_adv=np.sum(L_adv)"""

        distance = sum(np.sqrt(np.diff(x[:,0,0].cpu().detach().numpy())**2 + np.diff(x[:,0,1].cpu().detach().numpy())**2))
        
        if distance>settings.min_ego_distance:   
            print(f"Processing case {num}: distance = {distance:.2f}m, generating predictions...")
            y_pred = model(x, neighbor, n_predictions=config.PRED_SAMPLES)
            try:
                print(f"[MODEL] forward done: y_pred shape={tuple(y_pred.shape)} PRED_SAMPLES={getattr(config,'PRED_SAMPLES',None)}")
            except Exception:
                print("[MODEL] forward done")
        else:
            try:
                print(f"[SKIP] case {num}: distance {distance:.2f} <= min {settings.min_ego_distance}")
            except Exception:
                print("[SKIP] case {num}: distance below threshold")
            # 未达到最小位移阈值时跳过后续生成/保存流程，进入下一个样本
            continue

        # 通过最小位移阈值后，继续执行后续流程（地图、rerank、保存等）
        # 提前加载 lanelet 地图并构建 lane 多边形，供重排名使用
        laneletmap = None
        lane_paths = []
        try:
            dataset_root = os.path.dirname(os.path.dirname(csv_path))  # .../INTERACTION-Dataset-DR-multi-v1_2
            map_file_path = os.path.join(dataset_root, "maps", "DR_USA_Intersection_EP1.osm")
            if use_lanelet2_lib and os.path.exists(map_file_path):
                lat_origin = settings.lat_origin
                lon_origin = settings.lon_origin
                projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(lat_origin, lon_origin))
                laneletmap = lanelet2.io.load(map_file_path, projector)
                lane_paths = build_lanelet_paths(laneletmap)
            else:
                if not use_lanelet2_lib:
                    print("[WARN] lanelet2 不可用，重排名将不使用车道出道惩罚。")
                else:
                    print(f"[WARN] 地图文件不存在：{map_file_path}，重排名将不使用车道出道惩罚。")
        except Exception as _e:
            print(f"[WARN] 加载 lanelet 地图或构建 lane 多边形失败：{_e}")

        # 从多个预测中选出前 K 条用于可视化：优先使用 rerank（可行域+jerk），否则退回 FPC
        try:
            # 基于参数 --vis-topk 与 fpc_best、PRED_SAMPLES、实际可用候选数共同约束
            req_k = int(max(1, getattr(settings, 'vis_topk', 3)))
            k_vis = req_k
            # 与 fpc_best 取最小，防止展示数量超过 FPC 最佳值
            try:
                if 'fpc_best' in locals() and isinstance(fpc_best, (int, float)):
                    k_vis = min(k_vis, int(max(1, fpc_best)))
            except Exception:
                pass
            # 与 PRED_SAMPLES 取最小
            try:
                if hasattr(config, 'PRED_SAMPLES') and isinstance(config.PRED_SAMPLES, (int, float)):
                    k_vis = min(k_vis, int(max(1, config.PRED_SAMPLES)))
            except Exception:
                pass
            # 与 y_pred 的第一维（候选条数）取最小
            try:
                k_vis = min(k_vis, int(y_pred.shape[0]))
            except Exception:
                pass
        except Exception:
            k_vis = 3
        ranked_for_csv = []
        sel_idx_for_csv = []
        if settings.rerank_enable:
            try:
                print("[RERANK] start ...")
                ranked = rerank_indices(y_pred, y, lane_paths, lambda_lane=settings.lambda_lane, lambda_jerk=settings.lambda_jerk)
                # 打印每条候选的指标
                print("[RERANK] lambda_lane=%.3f lambda_jerk=%.3f" % (settings.lambda_lane, settings.lambda_jerk))
                for r in ranked:
                    print("  idx=%02d  score=%.4f  ade=%.4f  out_cnt=%d(%.2f)  jerk=%.4f" % (
                        r['idx'], r['score'], r['ade'], r['out_cnt'], r['out_frac'], r['jerk']))
                sel_idx = [r['idx'] for r in ranked[:k_vis]]
                print("[RERANK] selected indices:", sel_idx)
                ranked_for_csv = ranked
                sel_idx_for_csv = sel_idx
            except Exception as e:
                print(f"[ERROR] rerank failed: {e}")
                continue
        else:
            sel_idx = list(FPC(y_pred, k_vis))
            # 仍然计算一份 ranked 以导出 CSV（便于对比），但不用于选择
            ranked_for_csv = rerank_indices(y_pred, y, lane_paths, lambda_lane=settings.lambda_lane, lambda_jerk=settings.lambda_jerk)
            sel_idx_for_csv = sel_idx
        # 针对选中的候选进行硬性筛选（可选）
        if getattr(settings, 'filter_enable', True):
            max_outfrac = float(getattr(settings, 'filter_outfrac_max', 0.2))
            max_jerk = float(getattr(settings, 'filter_jerk_max', 0.6))
            max_turn = float(getattr(settings, 'filter_turn_deg_max', 75.0))
            kept = []
            dropped = []
            y_true_np = y.detach().cpu().numpy()[:,0,:]
            ranked_np = []
            for r in ranked_for_csv:
                ranked_np.append((r['idx'], r))
            # 逐个检查选择的候选
            for idx_ in sel_idx:
                pred = y_pred[idx_, :, 0, :].detach().cpu().numpy()
                # 出道比例（lane 多边形优先，退回 veh 栅格）
                out_frac = out_of_lane_fraction_with_fallback(pred, lane_paths, map_overlay)
                # jerk
                jv = jerk_mean(pred)
                # 最大转角
                maxdeg = traj_max_turn_degree(pred)
                if (out_frac <= max_outfrac) and (jv <= max_jerk) and (maxdeg <= max_turn):
                    kept.append(idx_)
                else:
                    dropped.append((idx_, out_frac, jv, maxdeg))
            # 不足 K 时，从剩余未选中的按综合分数补齐（同时应用滤网）
            if len(kept) < len(sel_idx):
                remaining = [r['idx'] for r in ranked_for_csv if r['idx'] not in set(sel_idx)]
                for idx_ in remaining:
                    pred = y_pred[idx_, :, 0, :].detach().cpu().numpy()
                    out_frac = out_of_lane_fraction_with_fallback(pred, lane_paths, map_overlay)
                    jv = jerk_mean(pred)
                    maxdeg = traj_max_turn_degree(pred)
                    if (out_frac <= max_outfrac) and (jv <= max_jerk) and (maxdeg <= max_turn):
                        kept.append(idx_)
                        if len(kept) >= len(sel_idx):
                            break
            if dropped:
                for idx_, ofr, jv, md in dropped:
                    print(f"[FILTER] drop idx={idx_:02d} out_frac={ofr:.2f} jerk={jv:.2f} max_turn={md:.1f}")
            # 若仍不足，则回退使用原始 sel_idx（确保有图可看）
            final_sel = kept if len(kept) > 0 else sel_idx
            print(f"[FILTER] kept={len(kept)} of {len(sel_idx)}; final_sel={final_sel}")
        else:
            final_sel = sel_idx
        y_pred = y_pred[final_sel, :, :, :]

        # ------- 风险检测（对 top-1 的生成轨迹计算 risk_score） -------
        danger_flag = False
        risk_score_val = None
        risk_detail = {}
        highlight_indices: List[int] = []
        if getattr(settings, 'risk_detect_enable', True):
            try:
                # 将 top-1 的生成坐标转换为绝对位置增量格式（pred: (T,B,2)）
                pred_abs = y_pred[0, :, 0, :]  # (T,2) absolute positions
                x_last = x[-1:, 0, :].double().to(settings.device)  # (1,6)
                x_last_pos = x[-1, 0, 0:2].double().to(settings.device)  # (2,)
                pred_top1 = (pred_abs.double().to(settings.device) - x_last_pos).unsqueeze(1)  # (T,1,2)
                neighbor_full = neighbor.double().to(settings.device)
                weights = {
                    'risk_min_dist': float(getattr(settings, 'risk_w_min_dist', 1.0)),
                    'risk_ttc': float(getattr(settings, 'risk_w_ttc', 1.0)),
                    'risk_pet': float(getattr(settings, 'risk_w_pet', 0.5)),
                    'risk_overlap': float(getattr(settings, 'risk_w_overlap', 1.0)),
                }
                comps = compute_risk_score(
                    pred_top1, x_last, neighbor_full,
                    weights=weights,
                    beta=getattr(config, 'RISK_MIN_DIST_BETA', 2.0),
                    ttc_tau=getattr(config, 'RISK_TTC_TAU', 1.5),
                    pet_dist_th=getattr(config, 'RISK_PET_DIST_TH', 2.0),
                    pet_alpha=getattr(config, 'RISK_PET_ALPHA', 8.0),
                    pet_beta=getattr(config, 'RISK_PET_BETA', 0.7),
                    pet_gamma=getattr(config, 'RISK_PET_GAMMA', 0.0),
                    pet_continuous=getattr(config, 'RISK_PET_CONTINUOUS', True),
                    pet_time_temp=getattr(config, 'RISK_PET_TIME_TEMP', 4.0),
                    ov_use_obb=getattr(config, 'RISK_OV_USE_OBB', False),
                    ov_self_length=getattr(config, 'RISK_OV_SELF_LENGTH', 4.5),
                    ov_self_width=getattr(config, 'RISK_OV_SELF_WIDTH', 1.8),
                    ov_neigh_length=getattr(config, 'RISK_OV_NEIGH_LENGTH', 4.5),
                    ov_neigh_width=getattr(config, 'RISK_OV_NEIGH_WIDTH', 1.8),
                    ov_min_speed=getattr(config, 'RISK_OV_MIN_SPEED', 1e-3),
                    ov_axis_beta=getattr(config, 'RISK_OV_AXIS_BETA', 12.0),
                    enable_pet=getattr(config, 'RISK_ENABLE_PET', True),
                    enable_overlap=getattr(config, 'RISK_ENABLE_OVERLAP', True),
                )
                risk_score_val = float(comps['risk_score'].detach().cpu().item())
                risk_detail = {k: float(v.detach().cpu().item()) for k, v in comps.items() if hasattr(v, 'detach')}
                danger_flag = risk_score_val >= float(getattr(settings, 'risk_threshold', 0.30))
                print(f"[RISK] score={risk_score_val:.4f} threshold={settings.risk_threshold} -> danger={danger_flag}")
                # 选择高亮邻居（按未来期与 ego 的距离/风险贡献）
                ob_len = x.shape[0]
                T_pred = y_pred.shape[1]
                ego_gen_np = y_pred[0, :, 0, :].detach().cpu().numpy()
                nei_full_np = neighbor.detach().cpu().numpy()
                if nei_full_np.ndim == 4 and nei_full_np.shape[0] >= ob_len + T_pred:
                    nei_future_pos = nei_full_np[ob_len:ob_len+T_pred, 0, :, 0:2]
                    nei_future_vel = nei_full_np[ob_len:ob_len+T_pred, 0, :, 2:4]
                    Nn = nei_future_pos.shape[1]
                    k = int(max(0, getattr(settings, 'highlight_topk_neigh', 3)))
                    mode = getattr(settings, 'highlight_mode', 'risk')
                    if mode == 'distance':
                        mins = []
                        for j in range(Nn):
                            xy = nei_future_pos[:, j, :]
                            if np.allclose(xy, 0.0):
                                mins.append((j, np.inf))
                                continue
                            mask = ~(np.isclose(xy[:,0], 0.0) & np.isclose(xy[:,1], 0.0))
                            if mask.sum() == 0:
                                mins.append((j, np.inf))
                                continue
                            d = np.linalg.norm(ego_gen_np[mask] - xy[mask], axis=1)
                            mins.append((j, float(d.min()) if d.size > 0 else np.inf))
                        mins.sort(key=lambda t: t[1])
                        highlight_indices = [j for j, _ in mins[:k] if np.isfinite(_)]
                    else:
                        # 风险贡献近似排序
                        eps = 1e-6
                        beta_v = float(getattr(config, 'RISK_MIN_DIST_BETA', 2.0))
                        ttc_tau = float(getattr(config, 'RISK_TTC_TAU', 1.5))
                        ov_r_self = float(getattr(config, 'OV_R_SELF', 0.5)) if hasattr(config, 'OV_R_SELF') else 0.5
                        ov_r_neigh = float(getattr(config, 'OV_R_NEIGH', 0.5)) if hasattr(config, 'OV_R_NEIGH') else 0.5
                        ov_margin = float(getattr(config, 'OV_MARGIN', 0.3)) if hasattr(config, 'OV_MARGIN') else 0.3
                        ov_k = float(getattr(config, 'OV_K', 15.0)) if hasattr(config, 'OV_K') else 15.0
                        w_min = float(getattr(settings, 'risk_w_min_dist', 1.0))
                        w_ttc = float(getattr(settings, 'risk_w_ttc', 1.0))
                        w_pet = float(getattr(settings, 'risk_w_pet', 0.5))
                        w_ov = float(getattr(settings, 'risk_w_overlap', 1.0))
                        dist = np.linalg.norm(nei_future_pos - ego_gen_np[:, None, :], axis=2)
                        dmin = np.min(dist, axis=0)
                        r_min = np.exp(-dmin)
                        rel_pos = nei_future_pos - ego_gen_np[:, None, :]
                        dist_safe = np.maximum(np.linalg.norm(rel_pos, axis=2), eps)
                        rel_pos_unit = rel_pos / dist_safe[..., None]
                        ego_vel_np = np.zeros_like(ego_gen_np)
                        ego_vel_np[0] = ego_gen_np[0] - x[-1, 0, 0:2].detach().cpu().numpy()
                        if T_pred > 1:
                            ego_vel_np[1:] = ego_gen_np[1:] - ego_gen_np[:-1]
                        rel_vel = nei_future_vel - ego_vel_np[:, None, :]
                        closing_speed = -(rel_vel * rel_pos_unit).sum(axis=2)
                        approaching = closing_speed > 0
                        ttc = np.full_like(dist_safe, 10.0)
                        ttc[approaching] = dist_safe[approaching] / (closing_speed[approaching] + eps)
                        ttc = np.clip(ttc, 0.0, 10.0)
                        r_ttc_t = np.exp(-ttc / max(ttc_tau, 1e-6))
                        r_ttc = r_ttc_t.max(axis=0)
                        thresh = ov_r_self + ov_r_neigh + ov_margin
                        pen = thresh - dist
                        r_ov = (np.log1p(np.exp(ov_k * pen)) / max(ov_k, 1e-6)).mean(axis=0)
                        pet_dist_th = float(getattr(config, 'RISK_PET_DIST_TH', 2.0))
                        pet_alpha = float(getattr(config, 'RISK_PET_ALPHA', 8.0))
                        pet_beta = float(getattr(config, 'RISK_PET_BETA', 0.7))
                        t_idx = np.arange(T_pred).reshape(T_pred, 1)
                        act = 1.0 / (1.0 + np.exp(-pet_alpha * (pet_dist_th - dist)))
                        r_pet_t = act * np.exp(-pet_beta * t_idx)
                        r_pet = r_pet_t.mean(axis=0)
                        score = w_min * r_min + w_ttc * r_ttc + w_ov * r_ov + w_pet * r_pet
                        order = np.argsort(-score)
                        highlight_indices = [int(j) for j in order[:k] if np.isfinite(score[j])]
            except Exception as e:
                print(f"[WARN] risk detection failed: {e}")
        
        # 清理后的配置
        mode = "Scenario_Pred"  # 生成完整场景
        Save = True             # 保存结果
        
        # 地图用于可视化：复用已加载的 laneletmap（若前面加载失败，这里不再重复加载）
        if laneletmap is None:
            if not use_lanelet2_lib:
                print("[WARN] lanelet2 不可用，跳过底图绘制。")
            else:
                # 仅在可视化阶段补一次尝试（防止上方意外失败）
                try:
                    dataset_root = os.path.dirname(os.path.dirname(csv_path))
                    map_file_path = os.path.join(dataset_root, "maps", "DR_USA_Intersection_EP1.osm")
                    if os.path.exists(map_file_path):
                        projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(settings.lat_origin, settings.lon_origin))
                        laneletmap = lanelet2.io.load(map_file_path, projector)
                    else:
                        print(f"[WARN] 地图文件不存在：{map_file_path}，跳过底图绘制。")
                except Exception as _e:
                    print(f"[WARN] 可视化阶段加载 lanelet 地图失败：{_e}")
        
        # 生成并保存动画/指标
        base_file = os.path.basename(csv_path)
        timestamp = str(datetime.now()).replace("-", "_").replace(" ", "_").replace(".", "_").replace(":", "_")
        def save_one_variant(name_base, y_pred_one, rank_i, sel_idx_val):
            name = name_base + ".gif"
            print(f"Generating animation: {name}")
            # 写出与 GIF 同名的 CSV 指标文件（本场景的完整重排名表，标注selected）
            try:
                csv_out_path = os.path.join(output_path, name[:-4] + ".csv")
                with open(csv_out_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["idx","rank","score","ade","out_cnt","out_frac","jerk","selected","lambda_lane","lambda_jerk"]) 
                    for rk, rec in enumerate(ranked_for_csv):
                        selected_flag = 1 if rec['idx'] == sel_idx_val else (1 if rec['idx'] in set(sel_idx_for_csv) else 0)
                        writer.writerow([
                            rec['idx'], rk, f"{rec['score']:.6f}", f"{rec['ade']:.6f}", rec['out_cnt'], f"{rec['out_frac']:.4f}", f"{rec['jerk']:.6f}", selected_flag, settings.lambda_lane, settings.lambda_jerk
                        ])
                print(f"[RERANK] metrics CSV saved: {csv_out_path}")
                # 追加到 master
                master_path = os.path.join(output_path, "rerank_master.csv")
                master_header = ["gif","idx","rank","score","ade","out_cnt","out_frac","jerk","selected","lambda_lane","lambda_jerk"]
                newfile = not os.path.exists(master_path)
                with open(master_path, 'a', newline='') as mf:
                    mw = csv.writer(mf)
                    if newfile:
                        mw.writerow(master_header)
                    for rk, rec in enumerate(ranked_for_csv):
                        selected_flag = 1 if rec['idx'] == sel_idx_val else (1 if rec['idx'] in set(sel_idx_for_csv) else 0)
                        mw.writerow([
                            name, rec['idx'], rk, f"{rec['score']:.6f}", f"{rec['ade']:.6f}", rec['out_cnt'], f"{rec['out_frac']:.4f}", f"{rec['jerk']:.6f}", selected_flag, settings.lambda_lane, settings.lambda_jerk
                        ])
                print(f"[RERANK] appended to master CSV: {master_path}")
            except Exception as e:
                print(f"[WARN] failed to write metrics CSV: {e}")
            try:
                print("[GIF] calling plot_trajectory_animation ...")
                plot_trajectory_animation(x, y, y_pred_one, neighbor, laneletmap, mode, output_path, name, meta=md_all, danger_flag=danger_flag, highlight_indices=highlight_indices, map_overlay=map_overlay)
                print("[GIF] done")
            except Exception as e:
                print(f"[ERROR] plot/save GIF failed: {e}")
            return name

        name_prefix = base_file.rpartition('_')[0] + "_" + timestamp
        saved_scenes = []  # list of tuples (name, y_pred_variant)
        if getattr(settings, 'per_traj_output', False):
            # 逐条输出：为 final_sel 中的每条候选生成独立文件
            for local_rank, (sel_idx_val) in enumerate(final_sel):
                name_base = f"{name_prefix}_sel{sel_idx_val:02d}_rank{local_rank}"
                y_one = y_pred[local_rank:local_rank+1, :, :, :]  # 形状保持 (1,T,B,2)，绘图函数仍能处理
                nm = save_one_variant(name_base, y_one, local_rank, sel_idx_val)
                saved_scenes.append((nm, y_one))
        else:
            # 原路径：单个 GIF 叠加多条（y_pred 已是筛选后的 K 条）
            name_base = name_prefix
            nm = save_one_variant(name_base, y_pred, 0, sel_idx_for_csv[0] if sel_idx_for_csv else -1)
            saved_scenes.append((nm, y_pred))
        
        # 保存轨迹数据
        if Save:
            print(f"Saving trajectory data files for case {num}...")
            long_all = os.path.join(output_path, 'trajs_long.csv')
            long_new_all = not os.path.exists(long_all)
            for name, y_var in saved_scenes:
                base_name = name[0:-4]
                try:
                    np.save(os.path.join(output_path, f"_Attack_His_{base_name}.npy"), x.cpu().detach().numpy())
                    np.save(os.path.join(output_path, f"_Attack_Tru_{base_name}.npy"), y.cpu().detach().numpy())
                    np.save(os.path.join(output_path, f"_Neighbor_{base_name}.npy"), neighbor.cpu().detach().numpy())
                    np.save(os.path.join(output_path, f"_Attack_Gen_{base_name}.npy"), y_var.cpu().detach().numpy())
                except Exception as e:
                    print(f"[ERROR] saving NPY failed: {e}")
                # 写出每个场景的轨迹长表（全样本，不仅危险）
                try:
                    per_scene_csv = os.path.join(output_path, f"{base_name}_trajs.csv")
                    per_new = not os.path.exists(per_scene_csv)
                    ob_len = x.shape[0]
                    T_pred = y_var.shape[1]
                    ego_hist = x.detach().cpu().numpy()[:, 0, 0:2]           # (ob_len,2)
                    ego_tru = y.detach().cpu().numpy()[:, 0, 0:2]            # (T_pred,2)
                    ego_gen = y_var[0, :, 0, :].detach().cpu().numpy()      # (T_pred,2)
                    nei_full = neighbor.detach().cpu().numpy()               # (ob+pred,1,Nn,6)
                    n_ids = md_all.get('neighbor_ids', [])
                    n_types = md_all.get('neighbor_types', [])
                    with open(long_all, 'a', newline='') as lfa, open(per_scene_csv, 'a', newline='') as pfa:
                        lw_all = csv.writer(lfa)
                        lw_per = csv.writer(pfa)
                        if long_new_all:
                            lw_all.writerow(['gif','src_case_id','window_start','danger','role','agent_id','agent_type','t','x','y'])
                            long_new_all = False
                        if per_new:
                            lw_per.writerow(['gif','src_case_id','window_start','danger','role','agent_id','agent_type','t','x','y'])
                        def wr(lw, role, aid, atp, t_idx, xx, yy):
                            lw.writerow([name, md_all.get('src_case_id', ''), md_all.get('window_start_frame', ''), int(danger_flag), role, aid, atp, t_idx, f"{xx:.6f}", f"{yy:.6f}"])
                        # ego hist
                        for t_i in range(ego_hist.shape[0]):
                            wr(lw_all, 'ego_hist', md_all.get('ego_id', ''), md_all.get('ego_type', ''), t_i, ego_hist[t_i,0], ego_hist[t_i,1])
                            wr(lw_per, 'ego_hist', md_all.get('ego_id', ''), md_all.get('ego_type', ''), t_i, ego_hist[t_i,0], ego_hist[t_i,1])
                        # ego true future
                        for t_i in range(ego_tru.shape[0]):
                            wr(lw_all, 'ego_tru', md_all.get('ego_id', ''), md_all.get('ego_type', ''), t_i, ego_tru[t_i,0], ego_tru[t_i,1])
                            wr(lw_per, 'ego_tru', md_all.get('ego_id', ''), md_all.get('ego_type', ''), t_i, ego_tru[t_i,0], ego_tru[t_i,1])
                        # ego generated future
                        for t_i in range(ego_gen.shape[0]):
                            wr(lw_all, 'ego_gen', md_all.get('ego_id', ''), md_all.get('ego_type', ''), t_i, ego_gen[t_i,0], ego_gen[t_i,1])
                            wr(lw_per, 'ego_gen', md_all.get('ego_id', ''), md_all.get('ego_type', ''), t_i, ego_gen[t_i,0], ego_gen[t_i,1])
                        # neighbors hist & future
                        if nei_full.ndim == 4:
                            Nn = nei_full.shape[2]
                            # hist 0:ob_len
                            for j in range(Nn):
                                aid = n_ids[j] if j < len(n_ids) else ''
                                atp = n_types[j] if j < len(n_types) else ''
                                # 历史
                                for t_i in range(ob_len):
                                    xx, yy = nei_full[t_i, 0, j, 0], nei_full[t_i, 0, j, 1]
                                    if np.isclose(xx, 0.0) and np.isclose(yy, 0.0):
                                        continue
                                    wr(lw_all, 'neighbor_hist', aid, atp, t_i, xx, yy)
                                    wr(lw_per, 'neighbor_hist', aid, atp, t_i, xx, yy)
                                # 未来
                                for t_i in range(T_pred):
                                    xx, yy = nei_full[ob_len + t_i, 0, j, 0], nei_full[ob_len + t_i, 0, j, 1]
                                    if np.isclose(xx, 0.0) and np.isclose(yy, 0.0):
                                        continue
                                    wr(lw_all, 'neighbor_future', aid, atp, t_i, xx, yy)
                                    wr(lw_per, 'neighbor_future', aid, atp, t_i, xx, yy)
                    print(f"[EXPORT] wrote per-scene trajs to {per_scene_csv} and appended to {long_all}")
                except Exception as e:
                    print(f"[WARN] failed to export per-scene trajectories: {e}")

            # 精简表：仅关键邻居，包含速度、加速度与航向角
            try:
                hi_only_all = os.path.join(output_path, 'trajs_highlight_only.csv')
                hi_only_new = not os.path.exists(hi_only_all)
                hi_scene_csv = os.path.join(output_path, f"{base_name}_trajs_highlight.csv")
                hi_scene_new = not os.path.exists(hi_scene_csv)
                ob_len = x.shape[0]
                T_pred = y_pred.shape[1]
                # Ego hist with derivatives from x (already has vx,vy,ax,ay)
                ego_hist6 = x.detach().cpu().numpy()[:, 0, :6]
                ego_hist_head = calculate_headings(ego_hist6[:,0], ego_hist6[:,1])
                # Ego true future (compute v,a from pos)
                ego_tru_pos = y.detach().cpu().numpy()[:, 0, :]
                v0_tru = ego_tru_pos[0] - x[-1, 0, 0:2].detach().cpu().numpy()
                ego_tru_vel = np.zeros_like(ego_tru_pos)
                ego_tru_vel[0] = v0_tru
                if T_pred > 1:
                    ego_tru_vel[1:] = ego_tru_pos[1:] - ego_tru_pos[:-1]
                ego_tru_acc = np.zeros_like(ego_tru_pos)
                ego_tru_acc[0] = ego_tru_vel[0] - x[-1, 0, 2:4].detach().cpu().numpy()
                if T_pred > 1:
                    ego_tru_acc[1:] = ego_tru_vel[1:] - ego_tru_vel[:-1]
                ego_tru_head = calculate_headings(ego_tru_pos[:,0], ego_tru_pos[:,1])
                # Ego gen future (top-1)
                ego_gen_pos = y_pred[0, :, 0, :].detach().cpu().numpy()
                ego_gen_vel = np.zeros_like(ego_gen_pos)
                ego_gen_vel[0] = ego_gen_pos[0] - x[-1, 0, 0:2].detach().cpu().numpy()
                if T_pred > 1:
                    ego_gen_vel[1:] = ego_gen_pos[1:] - ego_gen_pos[:-1]
                ego_gen_acc = np.zeros_like(ego_gen_pos)
                ego_gen_acc[0] = ego_gen_vel[0] - x[-1, 0, 2:4].detach().cpu().numpy()
                if T_pred > 1:
                    ego_gen_acc[1:] = ego_gen_vel[1:] - ego_gen_vel[:-1]
                ego_gen_head = calculate_headings(ego_gen_pos[:,0], ego_gen_pos[:,1])
                # neighbors
                nei_full = neighbor.detach().cpu().numpy()
                n_ids = md_all.get('neighbor_ids', [])
                n_types = md_all.get('neighbor_types', [])
                hi_idx = list(highlight_indices or [])
                with open(hi_only_all, 'a', newline='') as fa, open(hi_scene_csv, 'a', newline='') as fs:
                    wa = csv.writer(fa)
                    ws = csv.writer(fs)
                    header = ['gif','src_case_id','window_start','danger','role','agent_id','agent_type','t','x','y','vx','vy','ax','ay','heading_deg','speed']
                    if hi_only_new:
                        wa.writerow(header)
                    if hi_scene_new:
                        ws.writerow(header)
                    def wr(wrtr, role, aid, atp, t_idx, xx, yy, vx, vy, ax_, ay_, head):
                        spd = float(np.sqrt((vx or 0.0)**2 + (vy or 0.0)**2)) if not (np.isnan(vx) or np.isnan(vy)) else 0.0
                        wrtr.writerow([name, md_all.get('src_case_id', ''), md_all.get('window_start_frame', ''), int(danger_flag), role, aid, atp, t_idx, f"{xx:.6f}", f"{yy:.6f}", f"{(vx if not np.isnan(vx) else 0.0):.6f}", f"{(vy if not np.isnan(vy) else 0.0):.6f}", f"{(ax_ if not np.isnan(ax_) else 0.0):.6f}", f"{(ay_ if not np.isnan(ay_) else 0.0):.6f}", f"{float(head):.3f}", f"{spd:.6f}"])
                    # ego hist
                    for t_i in range(ob_len):
                        xx, yy, vx, vy, ax_, ay_ = ego_hist6[t_i, 0], ego_hist6[t_i, 1], ego_hist6[t_i, 2], ego_hist6[t_i, 3], ego_hist6[t_i, 4], ego_hist6[t_i, 5]
                        h = ego_hist_head[t_i]
                        wr(wa, 'ego_hist', md_all.get('ego_id', ''), md_all.get('ego_type', ''), t_i, xx, yy, vx, vy, ax_, ay_, h)
                        wr(ws, 'ego_hist', md_all.get('ego_id', ''), md_all.get('ego_type', ''), t_i, xx, yy, vx, vy, ax_, ay_, h)
                    # ego true
                    for t_i in range(T_pred):
                        xx, yy = ego_tru_pos[t_i]
                        vx, vy = ego_tru_vel[t_i]
                        ax_, ay_ = ego_tru_acc[t_i]
                        h = ego_tru_head[t_i]
                        wr(wa, 'ego_tru', md_all.get('ego_id', ''), md_all.get('ego_type', ''), t_i, xx, yy, vx, vy, ax_, ay_, h)
                        wr(ws, 'ego_tru', md_all.get('ego_id', ''), md_all.get('ego_type', ''), t_i, xx, yy, vx, vy, ax_, ay_, h)
                    # ego gen
                    for t_i in range(T_pred):
                        xx, yy = ego_gen_pos[t_i]
                        vx, vy = ego_gen_vel[t_i]
                        ax_, ay_ = ego_gen_acc[t_i]
                        h = ego_gen_head[t_i]
                        wr(wa, 'ego_gen', md_all.get('ego_id', ''), md_all.get('ego_type', ''), t_i, xx, yy, vx, vy, ax_, ay_, h)
                        wr(ws, 'ego_gen', md_all.get('ego_id', ''), md_all.get('ego_type', ''), t_i, xx, yy, vx, vy, ax_, ay_, h)
                    # neighbors (only highlighted)
                    if nei_full.ndim == 4 and len(hi_idx) > 0:
                        for j in hi_idx:
                            aid = n_ids[j] if j < len(n_ids) else ''
                            atp = n_types[j] if j < len(n_types) else ''
                            # hist
                            hist_xy = nei_full[:ob_len, 0, j, 0:2]
                            hist_v = nei_full[:ob_len, 0, j, 2:4]
                            hist_a = nei_full[:ob_len, 0, j, 4:6]
                            head_hist = calculate_headings(hist_xy[:,0], hist_xy[:,1]) if hist_xy.shape[0] > 1 else np.array([0.0])
                            for t_i in range(ob_len):
                                xx, yy = hist_xy[t_i]
                                vx, vy = hist_v[t_i]
                                ax_, ay_ = hist_a[t_i]
                                h = head_hist[t_i if t_i < len(head_hist) else -1]
                                if np.isclose(xx, 0.0) and np.isclose(yy, 0.0):
                                    continue
                                wr(wa, 'neighbor_hist', aid, atp, t_i, xx, yy, vx, vy, ax_, ay_, h)
                                wr(ws, 'neighbor_hist', aid, atp, t_i, xx, yy, vx, vy, ax_, ay_, h)
                            # future
                            fut_xy = nei_full[ob_len:ob_len+T_pred, 0, j, 0:2]
                            fut_v = nei_full[ob_len:ob_len+T_pred, 0, j, 2:4]
                            fut_a = nei_full[ob_len:ob_len+T_pred, 0, j, 4:6]
                            head_fut = calculate_headings(fut_xy[:,0], fut_xy[:,1]) if fut_xy.shape[0] > 1 else np.array([0.0])
                            for t_i in range(T_pred):
                                xx, yy = fut_xy[t_i]
                                vx, vy = fut_v[t_i]
                                ax_, ay_ = fut_a[t_i]
                                h = head_fut[t_i if t_i < len(head_fut) else -1]
                                if np.isclose(xx, 0.0) and np.isclose(yy, 0.0):
                                    continue
                                wr(wa, 'neighbor_future', aid, atp, t_i, xx, yy, vx, vy, ax_, ay_, h)
                                wr(ws, 'neighbor_future', aid, atp, t_i, xx, yy, vx, vy, ax_, ay_, h)
                print(f"[EXPORT] wrote highlight-only trajs to {hi_scene_csv} and appended to {hi_only_all}")
            except Exception as e:
                print(f"[WARN] failed to export highlight-only trajectories: {e}")

            # 若危险：同时把轨迹摘要追加到总 CSV
            if danger_flag:
                try:
                    danger_csv = os.path.join(output_path, 'dangerous_trajs.csv')
                    newfile = not os.path.exists(danger_csv)
                    with open(danger_csv, 'a', newline='') as f:
                        w = csv.writer(f)
                        if newfile:
                            w.writerow(['gif','src_case_id','window_start','ego_id','ego_type','neighbor_ids','neighbor_types','risk_score','risk_min_dist','risk_ttc','risk_pet','risk_overlap','highlight_neighbor_ids'])
                        w.writerow([
                            name,
                            (meta.get('src_case_id') if isinstance(meta, dict) else ''),
                            (meta.get('window_start_frame') if isinstance(meta, dict) else ''),
                            (meta.get('ego_id') if isinstance(meta, dict) else ''),
                            (meta.get('ego_type') if isinstance(meta, dict) else ''),
                            ';'.join(map(str, meta.get('neighbor_ids', []))) if isinstance(meta, dict) else '',
                            ';'.join(map(str, meta.get('neighbor_types', []))) if isinstance(meta, dict) else '',
                            f"{risk_score_val:.6f}" if risk_score_val is not None else '',
                            f"{risk_detail.get('risk_min_dist', float('nan')):.6f}" if risk_detail else '',
                            f"{risk_detail.get('risk_ttc', float('nan')):.6f}" if risk_detail else '',
                            f"{risk_detail.get('risk_pet', float('nan')):.6f}" if risk_detail else '',
                            f"{risk_detail.get('risk_overlap', float('nan')):.6f}" if risk_detail else '',
                            ';'.join(map(lambda z: str(meta.get('neighbor_ids', [])[z]) if (isinstance(meta, dict) and z < len(meta.get('neighbor_ids', []))) else str(z), (highlight_indices or [])))
                        ])
                    print(f"[RISK] appended dangerous traj to {danger_csv}")
                    # 详细逐时刻轨迹导出（长表）：仅导出未来段
                    long_csv = os.path.join(output_path, 'dangerous_trajs_long.csv')
                    long_new = not os.path.exists(long_csv)
                    # 未来时长
                    T_pred = y_pred.shape[1]
                    # ego 生成轨迹（top-1）
                    ego_gen = y_pred[0, :, 0, :].detach().cpu().numpy()
                    # 邻居未来位置
                    ob_len = x.shape[0]
                    nei_full = neighbor.detach().cpu().numpy()  # (ob+pred,1,Nn,6)
                    if nei_full.ndim == 4 and nei_full.shape[0] >= ob_len + T_pred:
                        nei_future = nei_full[ob_len:ob_len+T_pred, 0, :, 0:2]  # (T_pred,Nn,2)
                    else:
                        nei_future = None
                    with open(long_csv, 'a', newline='') as lf:
                        lw = csv.writer(lf)
                        if long_new:
                            lw.writerow(['gif','src_case_id','t','role','agent_id','agent_type','x','y'])
                        # ego rows
                        for t_i in range(T_pred):
                            lw.writerow([name, (meta.get('src_case_id') if isinstance(meta, dict) else ''), t_i, 'ego_gen', (meta.get('ego_id') if isinstance(meta, dict) else ''), (meta.get('ego_type') if isinstance(meta, dict) else ''), f"{ego_gen[t_i,0]:.6f}", f"{ego_gen[t_i,1]:.6f}"])
                        # neighbor rows
                        if nei_future is not None:
                            n_ids = meta.get('neighbor_ids', []) if isinstance(meta, dict) else []
                            n_types = meta.get('neighbor_types', []) if isinstance(meta, dict) else []
                            for j in range(nei_future.shape[1]):
                                aid = n_ids[j] if j < len(n_ids) else ''
                                atp = n_types[j] if j < len(n_types) else ''
                                for t_i in range(T_pred):
                                    xj, yj = nei_future[t_i, j, 0], nei_future[t_i, j, 1]
                                    lw.writerow([name, (meta.get('src_case_id') if isinstance(meta, dict) else ''), t_i, 'neighbor', aid, atp, f"{xj:.6f}", f"{yj:.6f}"])
                    print(f"[RISK] appended detailed trajs to {long_csv}")
                except Exception as e:
                    print(f"[WARN] failed to append dangerous_trajs.csv: {e}")
        
        plt.close('all')  # 关闭所有图形窗口，避免阻塞
        print(f"Case {num} completed successfully.")


#%%
"""
    neighbor_array= neighbor.cpu().detach().numpy().squeeze()
    
    x_min, x_max = neighbor_array[:,:,0].min(), neighbor_array[:,:,0].max()
    y_min, y_max = neighbor_array[:,:,1].min(), neighbor_array[:,:,1].max()

    x_min, x_max = x_min - 0.2 * (x_max - x_min), x_max + 0.2 * (x_max - x_min)
    y_min, y_max = y_min - 0.2 * (y_max - y_min), y_max + 0.2 * (y_max - y_min)
    
    x = np.linspace(x_min, x_max, 200)
    y = np.linspace(y_min, y_max, 200)
    X, Y = np.meshgrid(x, y)

    kappa = 1
    kappa1,kappa2 = 0.2, 0.4

    lambda_val,gamma_val,m_val=2,0.5,2
    temp_neighbor_array=neighbor_array[0,:,:]
    
    for index, row in enumerate(temp_neighbor_array):
        
        x0=row[0]
        y0=row[1]
        Vx=row[2]
        Vy=row[3]*3.6
        ax=row[4]*3.6
        ay=row[5]
        if index==0:  
            field_strength = DRF.DRF_strength(X, Y, x0, y0, Vx,Vy, ax,ay, kappa,kappa1,kappa2,lambda_val,gamma_val,m_val)
            print(field_strength)
        else:
            field_strength += DRF.DRF_strength(X, Y, x0, y0, Vx,Vy, ax,ay, kappa,kappa1,kappa2,lambda_val,gamma_val,m_val)
       
            print(field_strength)
            
    #field_strength = np.where(field_strength >=1 , field_strength, 0)   
    field_strength = np.clip(field_strength, None,5)
        
    plt.contourf(X, Y, field_strength, levels=100, cmap='viridis')
        
    plt.colorbar(label='Field Strength')
    plt.title('Elliptical Field Strength')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.gca().set_aspect('equal', adjustable='box')  # 保持横纵坐标比例一致
    plt.grid(True)
    
    #Lanelet_Map_Viz
    plt.show()
    
    DR_USA_Roundabout_FT.osm
    
"""