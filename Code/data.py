import os, sys
import torch
import numpy as np
from typing import List, Optional, Sequence

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor


class Dataloader(torch.utils.data.Dataset):
    """
    The purpose of this class is to generate a fixed number of mini-batches per epoch, 
    deviating from the standard batch sampling.    
    """
    class FixedNumberBatchSampler(torch.utils.data.sampler.BatchSampler):
        def __init__(self, n_batches, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.n_batches = n_batches
            self.sampler_iter = None #iter(self.sampler)
        def __iter__(self):
            # same with BatchSampler, but StopIteration every n batches
            counter = 0
            batch = []
            while True:
                if counter >= self.n_batches:
                    break
                if self.sampler_iter is None: 
                    self.sampler_iter = iter(self.sampler)
                try:
                    idx = next(self.sampler_iter)
                except StopIteration:
                    self.sampler_iter = None
                    if self.drop_last: batch = []
                    continue
                batch.append(idx)
                if len(batch) == self.batch_size:
                    counter += 1
                    yield batch
                    batch = []

    """
    files: A list of file paths containing trajectory data in text format.
    ob_horizon: The observation horizon (number of steps to consider for observation).
    pred_horizon: The prediction horizon (number of steps to consider for prediction).
    batch_size: The batch size for training.
    drop_last: A boolean indicating whether to drop the last batch if its size is less than batch_size.
    shuffle: A boolean indicating whether to shuffle the data during training.
    batches_per_epoch: Number of batches per epoch (if specified, overrides automatic calculation based on batch size and dataset size).
    frameskip: The number of frames to skip between consecutive observations.
    inclusive_groups: A sequence of inclusive groups for each file. Default is None.
    batch_first: A boolean indicating whether the batch dimension is the first dimension in the data.
    seed: An optional random seed for reproducibility.
    device: An optional parameter specifying the device (CPU or GPU) to use for data loading.
    flip, rotate, scale: Boolean parameters indicating whether to apply data augmentation techniques (flip, rotate, scale).
    """

    def __init__(self, 
        files: List[str], ob_horizon: int, pred_horizon: int,
        batch_size: int, drop_last: bool=False, shuffle: bool=False, batches_per_epoch=None, 
        frameskip: int=1, inclusive_groups: Optional[Sequence]=None,
        batch_first: bool=False, seed: Optional[int]=None,
        device: Optional[torch.device]=None,
        flip: bool=False, rotate: bool=False, scale: bool=False,
        include_map_meta: bool=False, map_raster_size: Optional[Sequence[int]]=None, map_channels: int=2,
        map_osm_xy_path: Optional[str]=None,
        # 新增：VRU 渗透与边界软化参数（用于控制“软约束”强度）
        map_vru_leak_to_vehicle: float = 0.0,
        map_raster_soften_k: int = 0,
        map_raster_soften_iters: int = 0
    ):
        super().__init__()
        self.ob_horizon = ob_horizon
        self.pred_horizon = pred_horizon
        self.horizon = self.ob_horizon+self.pred_horizon
        self.frameskip = int(frameskip) if frameskip and int(frameskip) > 1 else 1
        self.batch_first = batch_first
        self.flip = flip
        self.rotate = rotate
        self.scale = scale
        self.include_map_meta = bool(include_map_meta)
        if map_raster_size is None:
            self.map_raster_size = (256, 256)  # (H, W)
        else:
            assert len(map_raster_size) == 2, "map_raster_size must be (H,W)"
            self.map_raster_size = (int(map_raster_size[0]), int(map_raster_size[1]))
        self.map_channels = int(map_channels)
        self.map_osm_xy_path = map_osm_xy_path
        # 软约束参数
        self.map_vru_leak_to_vehicle = float(map_vru_leak_to_vehicle)
        self.map_raster_soften_k = int(map_raster_soften_k)
        self.map_raster_soften_iters = int(map_raster_soften_iters)
        # preloaded map polygons (world coords, meters)
        self._map_lanelet_polys = None  # list of np.ndarray (K,2)
        self._map_keepout_polys = None  # list of np.ndarray (K,2)
        self._map_vru_polys = None      # list of np.ndarray (K,2)

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available else "cpu") 
        else:
            self.device = device

        if inclusive_groups is None:
            inclusive_groups = [[] for _ in range(len(files))]
        assert(len(inclusive_groups) == len(files))

        print(" Scanning files...")
        files_ = []
        for path, incl_g in zip(files, inclusive_groups):
            if os.path.isdir(path):
                files_.extend([(os.path.join(root, f), incl_g) \
                    for root, _, fs in os.walk(path) \
                    for f in fs if f.endswith(".txt")])
            elif os.path.exists(path):
                files_.append((path, incl_g))
        data_files = sorted(files_, key=lambda _: _[0])

        data = []
        
        done = 0
        # too large of max_workers will cause the problem of memory usage
        max_workers = min(len(data_files), torch.get_num_threads(), 20)
        
        # 安全检查：确保max_workers至少为1，避免ProcessPoolExecutor错误
        if max_workers == 0:
            print("\n   Warning: No data files found! Check your data path.")
            max_workers = 1
            data_files = []  # 确保空列表
        
        disable_pool = os.environ.get("DATA_DISABLE_PROCESSPOOL", "0") == "1"
        if len(data_files) > 0:
            if disable_pool:
                print("   DATA_DISABLE_PROCESSPOOL=1 -> 使用串行方式加载数据 (便于调试/避免多进程问题)")
                for f, incl_g in data_files:
                    item = self.__class__.load(self, f, incl_g)
                    done += 1
                    if item is not None:
                        data.extend(item)
                    sys.stdout.write("\r\033[K Loading data files...{}/{}".format(done, len(data_files)))
            else:
                with ProcessPoolExecutor(mp_context=multiprocessing.get_context("spawn"), max_workers=max_workers) as p:
                    futures = [p.submit(self.__class__.load, self, f, incl_g) for f, incl_g in data_files]
                    for fut in as_completed(futures):
                        done += 1
                        sys.stdout.write("\r\033[K Loading data files...{}/{}".format(
                            done, len(data_files)
                        ))
                    for fut in futures:
                        item = fut.result()
                        if item is not None:
                            data.extend(item)
                        sys.stdout.write("\r\033[K Loading data files...{}/{} ".format(
                            done, len(data_files)
                        ))
        self.data = np.array(data, dtype=object)
        del data
        print("\n   {} trajectories loaded.".format(len(self.data)))
        
        # 重要：如果没有加载到数据，添加一个虚拟数据项防止后续错误
        if len(self.data) == 0:
            print("   Warning: No trajectories found. Adding dummy data to prevent crashes.")
            # 创建一个虚拟轨迹数据 (hist, future, neighbor)
            dummy_hist = np.zeros((self.ob_horizon, 6), dtype=np.float32)
            dummy_future = np.zeros((self.pred_horizon, 2), dtype=np.float32) 
            dummy_neighbor = np.ones((self.ob_horizon + self.pred_horizon, 1, 6), dtype=np.float32) * 1e9
            self.data = np.array([(dummy_hist, dummy_future, dummy_neighbor)], dtype=object)
        
        self.rng = np.random.RandomState()
        if seed: self.rng.seed(seed)

        # If semantic map requested, try to preload polygons from .osm_xy (preferred) or via lanelet2
        if self.include_map_meta and self.map_osm_xy_path and os.path.exists(self.map_osm_xy_path):
            try:
                self._load_osm_xy_polygons(self.map_osm_xy_path)
                print("   Loaded polygons from .osm_xy for MAP_BCE (lanelets={}, keepouts={}, vru={}).".format(
                    len(self._map_lanelet_polys or []), len(self._map_keepout_polys or []), len(self._map_vru_polys or [])
                ))
                # Build global raster and world2map once (use entire map bbox)
                self._map_world_bounds = self._compute_map_bounds()
                self._map_raster_global, self._world2map_global = self._build_global_map_raster(self._map_world_bounds)
                if self._map_raster_global is not None:
                    H,W = self._map_raster_global.shape[-2:]
                    print(f"   Built global map raster for MAP_BCE: size=({H},{W})")
            except Exception as e:
                print(f"   [WARN] Failed loading .osm_xy polygons: {e}. Fallback to dummy raster.")

        if shuffle:
            sampler = torch.utils.data.sampler.RandomSampler(self)
        else:
            sampler = torch.utils.data.sampler.SequentialSampler(self)
        if batches_per_epoch is None:
            self.batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last)
            self.batches_per_epoch = len(self.batch_sampler)
        else:
            self.batch_sampler = self.__class__.FixedNumberBatchSampler(batches_per_epoch, sampler, batch_size, drop_last)
            self.batches_per_epoch = batches_per_epoch

    def collate_fn(self, batch):
        X, Y, NEIGHBOR = [], [], []
        agent_groups = []  # keep original group tokens for ego
        for item in batch:
            hist, future, neighbor = item[0], item[1], item[2]
            group_tok = item[3] if len(item) > 3 else None

            hist_shape = hist.shape
            neighbor_shape = neighbor.shape
            hist = np.reshape(hist, (-1, 2))
            neighbor = np.reshape(neighbor, (-1, 2))
            if self.flip:
                if self.rng.randint(2):
                    hist[..., 1] *= -1
                    future[..., 1] *= -1
                    neighbor[..., 1] *= -1
                if self.rng.randint(2):
                    hist[..., 0] *= -1
                    future[..., 0] *= -1
                    neighbor[..., 0] *= -1
            if self.rotate:
                rot = self.rng.random() * (np.pi+np.pi) 
                s, c = np.sin(rot), np.cos(rot)
                r = np.asarray([
                    [c, -s],
                    [s,  c]
                ])
                hist = (r @ np.expand_dims(hist, -1)).squeeze(-1)
                future = (r @ np.expand_dims(future, -1)).squeeze(-1)
                neighbor = (r @ np.expand_dims(neighbor, -1)).squeeze(-1)
            if self.scale:
                s = self.rng.randn()*0.05 + 1 # N(1, 0.05)
                hist = s * hist
                future = s * future
                neighbor = s * neighbor
            hist = np.reshape(hist, hist_shape)
            neighbor = np.reshape(neighbor, neighbor_shape)

            X.append(hist)
            Y.append(future)
            NEIGHBOR.append(neighbor)
            agent_groups.append(group_tok)
        
        n_neighbors = [n.shape[1] for n in NEIGHBOR]
        max_neighbors = max(n_neighbors) 
        if max_neighbors != min(n_neighbors):
            NEIGHBOR = [
                np.pad(neighbor, ((0, 0), (0, max_neighbors-n), (0, 0)), 
                "constant", constant_values=1e9)
                for neighbor, n in zip(NEIGHBOR, n_neighbors)
            ]
        stack_dim = 0 if self.batch_first else 1
        x = np.stack(X, stack_dim)
        y = np.stack(Y, stack_dim)
        neighbor = np.stack(NEIGHBOR, stack_dim)

        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device)
        neighbor = torch.tensor(neighbor, dtype=torch.float32, device=self.device)

        if not self.include_map_meta:
            return x, y, neighbor

        # --- Build map_meta from global raster if available; else fallback to per-batch or ones ---
        try:
            # Prefer global map raster if available (stable world2map across batches)
            if getattr(self, '_map_raster_global', None) is not None and getattr(self, '_world2map_global', None) is not None:
                raster = self._map_raster_global.to(device=self.device, dtype=torch.float32)
                world2map = self._world2map_global.to(device=self.device, dtype=torch.float32)
            else:
                # If no global raster, fallback to previous per-batch computation (rare)
                # 1) Compute batch bbox and world->map affine
                hist_xy = x[...,:2]
                fut_xy = y
                xy_concat = torch.cat([hist_xy.reshape(-1, 2), fut_xy.reshape(-1, 2)], dim=0)
                mask_valid = (xy_concat.abs() < 1e8).all(dim=-1)
                xy_valid = xy_concat[mask_valid] if mask_valid.any() else xy_concat
                min_xy = xy_valid.min(dim=0).values
                max_xy = xy_valid.max(dim=0).values
                margin = 1.0
                if (max_xy - min_xy).max() < 1e-6:
                    max_xy = min_xy + torch.tensor([1.0, 1.0], device=xy_valid.device)
                min_xy = min_xy - margin
                max_xy = max_xy + margin
                H, W = self.map_raster_size
                sx = (W - 1) / (max_xy[0] - min_xy[0]).clamp_min(1e-6)
                sy = (H - 1) / (max_xy[1] - min_xy[1]).clamp_min(1e-6)
                tx = -min_xy[0] * sx
                ty = -min_xy[1] * sy
                world2map = torch.tensor([[sx, 0.0, tx], [0.0, sy, ty]], dtype=torch.float32, device=self.device)
                # 2) Assembled per-batch raster of ones (small fallback)
                raster = torch.ones((self.map_channels, H, W), dtype=torch.float32, device=self.device)
            # agent type per sample (batch column). Fallback to 'vehicle' if unknown.
            def group_to_agent_type(gtok):
                if gtok is None: return 'vehicle'
                # normalize to list of tokens
                tokens = gtok if isinstance(gtok, (list, tuple)) else [str(gtok)]
                norm = [str(t).lower() for t in tokens]
                # VRU keywords
                vru_keys = ['ped', 'pedestrian', 'person', 'walker', 'foot', 'footway', 'sidewalk', 'bike', 'bicy', 'cyclist', 'vru']
                if any(any(k in t for k in vru_keys) for t in norm):
                    return 'vru'
                return 'vehicle'
            agent_types = [group_to_agent_type(g) for g in agent_groups]
            # Build channel_per_agent tensor: 0 for vehicle else 1
            ch_per_agent = [0 if at != 'vru' else 1 for at in agent_types]
            channel_per_agent = torch.tensor(ch_per_agent, device=self.device, dtype=torch.long)
            map_meta = {
                'raster': raster,
                'world2map': world2map,
                'channel': None,
                'agent_type': 'vehicle',
                'agent_types': agent_types,
                'channel_per_agent': channel_per_agent,
            }
        except Exception as e:
            print(f"   [WARN] map_meta rasterization failed: {e}. Using fallback ones raster.")
            map_meta = {
                'raster': torch.ones((self.map_channels, *self.map_raster_size), dtype=torch.float32, device=self.device),
                'world2map': torch.tensor([[1.0,0.0,0.0],[0.0,1.0,0.0]], dtype=torch.float32, device=self.device),
                'channel': None,
                'agent_type': 'vehicle',
                'agent_types': ['vehicle'] * (x.shape[1] if x.dim()>=2 else 1),
                'channel_per_agent': torch.zeros((x.shape[1] if x.dim()>=2 else 1,), dtype=torch.long, device=self.device),
            }
        return x, y, neighbor, map_meta

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def load(self, filename, inclusive_groups):
        if os.path.isdir(filename): return None

        horizon = (self.horizon-1)*self.frameskip
        with open(filename, "r") as record:
            data = self.load_traj(record)
        data = self.extend(data, self.frameskip)
        
        time = np.sort(list(data.keys()))
        if len(time) < horizon+1: return None
        valid_horizon = self.ob_horizon + self.pred_horizon

        traj = []
        e = len(time)
        tid0 = 0
        while tid0 < e-horizon:
            tid1 = tid0+horizon
            t0 = time[tid0]
            
            #print(data[t0].items())
            #print("__________________________")
            
            idx = [aid for aid, d in data[t0].items() if not inclusive_groups or any(g in inclusive_groups for g in d[-1])]
            if idx:
                idx_all = list(data[t0].keys())
                for tid in range(tid0+self.frameskip, tid1+1, self.frameskip):
                    t = time[tid]
                    idx_cur = [aid for aid, d in data[t].items() if not inclusive_groups or any(g in inclusive_groups for g in d[-1])]
                    if not idx_cur: # ignore empty frames
                        tid0 = tid
                        idx = []
                        break
                    idx = np.intersect1d(idx, idx_cur)
                    if len(idx) == 0: break
                    idx_all.extend(data[t].keys())
            if len(idx):
                data_dim = 6
                neighbor_idx = np.setdiff1d(idx_all, idx) #to find values in idx_all that are not in idx
                if len(idx) == 1 and len(neighbor_idx) == 0:
                    agents = np.array([
                        [data[time[tid]][idx[0]][:data_dim]] + [[1e9]*data_dim]
                        for tid in range(tid0, tid1+1, self.frameskip)
                    ]) # L x 2 x 6
                else:
                    agents = np.array([
                        [data[time[tid]][i][:data_dim] for i in idx] +
                        [data[time[tid]][j][:data_dim] if j in data[time[tid]] else [1e9]*data_dim for j in neighbor_idx]
                        for tid in range(tid0, tid1+1, self.frameskip)
                    ])  # L X N x 6
                for i in range(len(idx)):
                    hist = agents[:self.ob_horizon,i]  # L_ob x 6
                    future = agents[self.ob_horizon:valid_horizon,i,:2]  # L_pred x 2
                    neighbor = agents[:valid_horizon, [d for d in range(agents.shape[1]) if d != i]] # L x (N-1) x 6
                    # ego group tokens at t0 (if present)
                    ego_group = None
                    try:
                        gtok = data[time[tid0]][idx[i]][-1]
                        ego_group = gtok
                    except Exception:
                        ego_group = None
                    traj.append((hist, future, neighbor, ego_group))
            tid0 += 1

        items = []
        for hist, future, neighbor, ego_group in traj:
            hist = np.float32(hist)
            future = np.float32(future)
            neighbor = np.float32(neighbor)
            items.append((hist, future, neighbor, ego_group))
        return items
                
    def extend(self, data, frameskip):
        """
        extending and interpolating data in a trajectory dataset. 
        """
        time = np.sort(list(data.keys()))
        dts = np.unique(time[1:] - time[:-1])
        dt = dts.min()
        if np.any(dts % dt != 0):
            raise ValueError("Inconsistent frame interval:", dts) # If there are inconsistent intervals, it raises a ValueError.
        i = 0
        while i < len(time)-1:
            if time[i+1] - time[i] != dt:
                time = np.insert(time, i+1, time[i]+dt) #numpy.insert(arr, obj, values, axis=None)
            i += 1
        """     
        The method then removes entries (identified by idx) that only appear at one frame 
        and are not present in the neighboring frames within a given frameskip.
        """
        for tid, t in enumerate(time):
            removed = []
            if t not in data: data[t] = {}
            for idx in data[t].keys():
                t0 = time[tid-frameskip] if tid >= frameskip else None
                t1 = time[tid+frameskip] if tid+frameskip < len(time) else None
                if (t0 is None or t0 not in data or idx not in data[t0]) and \
                (t1 is None or t1 not in data or idx not in data[t1]):
                    removed.append(idx)
            for idx in removed:
                data[t].pop(idx)
        
        # calculate v
        for tid in range(len(time)-frameskip):
            t0 = time[tid]
            t1 = time[tid+frameskip]
            if t1 not in data or t0 not in data: continue
            for i, item in data[t1].items():
                if i not in data[t0]: continue
                x0 = data[t0][i][0]
                y0 = data[t0][i][1]
                x1 = data[t1][i][0]
                y1 = data[t1][i][1]
                vx, vy = x1-x0, y1-y0
                data[t1][i].insert(2, vx)
                data[t1][i].insert(3, vy)
                if tid < frameskip or i not in data[time[tid-1]]:
                    data[t0][i].insert(2, vx)
                    data[t0][i].insert(3, vy)
        
        # calculate a
        for tid in range(len(time)-frameskip):
            t_1 = None if tid < frameskip else time[tid-frameskip]
            t0 = time[tid]
            t1 = time[tid+frameskip]
            if t1 not in data or t0 not in data: continue
            for i, item in data[t1].items():
                if i not in data[t0]: continue
                vx0 = data[t0][i][2]
                vy0 = data[t0][i][3]
                vx1 = data[t1][i][2]
                vy1 = data[t1][i][3]
                ax, ay = vx1-vx0, vy1-vy0
                data[t1][i].insert(4, ax)
                data[t1][i].insert(5, ay)
                if t_1 is None or i not in data[t_1]:
                    # first appearing frame, pick value from the next frame
                    data[t0][i].insert(4, ax)
                    data[t0][i].insert(5, ay)
        return data

    def load_traj(self, file):
        data = {}
        for row in file.readlines():
            item = row.split()
            if not item: continue
            t = int(float(item[0]))
            idx = int(float(item[1]))
            x = float(item[2])
            y = float(item[3])
            group = item[4].split("/") if len(item) > 4 else None
            """
            group = item[4].split("/") if len(item) > 4 else None: Checks if there is a fifth item.
            If present, it splits the fifth item using "/" and assigns the result to group; otherwise, group is set to None.
            """
            if t not in data:
                data[t] = {}
            data[t][idx] = [x, y, group]
        return data

    def _compute_map_bounds(self):
        # Compute world bounds from all available polygons
        xs, ys = [], []
        for coll in [self._map_lanelet_polys, self._map_vru_polys, self._map_keepout_polys]:
            if coll is None:
                continue
            for poly in coll:
                if poly is None or len(poly) == 0:
                    continue
                xs.append(poly[:,0]); ys.append(poly[:,1])
        if not xs:
            return None
        minx = float(np.min([x.min() for x in xs])); maxx = float(np.max([x.max() for x in xs]))
        miny = float(np.min([y.min() for y in ys])); maxy = float(np.max([y.max() for y in ys]))
        # add small margin
        m = 1.0
        return (minx - m, miny - m, maxx + m, maxy + m)

    def _build_global_map_raster(self, bounds):
        try:
            if bounds is None or self._map_lanelet_polys is None:
                return None, None
            minx, miny, maxx, maxy = bounds
            H, W = self.map_raster_size
            sx = (W - 1) / max(1e-6, (maxx - minx))
            sy = (H - 1) / max(1e-6, (maxy - miny))
            tx = -minx * sx
            ty = -miny * sy
            world2map = torch.tensor([[sx, 0.0, tx], [0.0, sy, ty]], dtype=torch.float32)
            # Pixel grid
            uu = np.arange(W, dtype=np.float32)
            vv = np.arange(H, dtype=np.float32)
            U, V = np.meshgrid(uu, vv)
            pts = np.stack([U.ravel(), V.ravel()], axis=1)
            # transform helper
            W2M = world2map.numpy()
            def tf_poly(poly):
                xy = np.asarray(poly, dtype=np.float32)
                u = xy[:,0]*W2M[0,0] + xy[:,1]*W2M[0,1] + W2M[0,2]
                v = xy[:,0]*W2M[1,0] + xy[:,1]*W2M[1,1] + W2M[1,2]
                return np.stack([u, v], axis=1)
            # prefer matplotlib.path if available
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
            # vehicle mask
            allowed = np.zeros((H*W,), dtype=bool)
            for poly in (self._map_lanelet_polys or []):
                p_uv = tf_poly(poly)
                if MplPath is not None:
                    allowed |= MplPath(p_uv, closed=True).contains_points(pts)
                else:
                    allowed |= pip_numpy(p_uv, pts)
            for poly in (self._map_keepout_polys or []):
                p_uv = tf_poly(poly)
                if MplPath is not None:
                    allowed &= ~MplPath(p_uv, closed=True).contains_points(pts)
                else:
                    allowed &= ~pip_numpy(p_uv, pts)
            veh_mask = allowed.reshape(H, W).astype(np.float32)
            # vru mask
            if self._map_vru_polys is not None and len(self._map_vru_polys) > 0:
                allowed_vru = np.zeros((H*W,), dtype=bool)
                for poly in (self._map_vru_polys or []):
                    p_uv = tf_poly(poly)
                    if MplPath is not None:
                        allowed_vru |= MplPath(p_uv, closed=True).contains_points(pts)
                    else:
                        allowed_vru |= pip_numpy(p_uv, pts)
                for poly in (self._map_keepout_polys or []):
                    p_uv = tf_poly(poly)
                    if MplPath is not None:
                        allowed_vru &= ~MplPath(p_uv, closed=True).contains_points(pts)
                    else:
                        allowed_vru &= ~pip_numpy(p_uv, pts)
                vru_mask = allowed_vru.reshape(H, W).astype(np.float32)
            else:
                vru_mask = veh_mask.copy()

            # ---- 可选：VRU 对车辆道的“少量渗透”与边界软化（创建更平滑的 BCE）----
            leak = max(0.0, float(getattr(self, 'map_vru_leak_to_vehicle', 0.0)))
            if leak > 0.0:
                # 允许 VRU 通道在车辆道内有少量概率（避免硬性禁止“过街”）；仍受 keepout 限制
                vru_mask = np.clip(vru_mask + leak * veh_mask, 0.0, 1.0)

            def box_blur01(img: np.ndarray, k: int, iters: int) -> np.ndarray:
                if k is None or k <= 1 or iters is None or iters <= 0:
                    return img
                # 归一化盒滤波核（k x k），简单循环实现，224x224 上开销很小
                pad = k // 2
                out = img.copy()
                for _ in range(iters):
                    padded = np.pad(out, ((pad,pad),(pad,pad)), mode='edge')
                    acc = np.zeros_like(out, dtype=np.float32)
                    # 朴素卷积
                    for dy in range(k):
                        for dx in range(k):
                            acc += padded[dy:dy+H, dx:dx+W]
                    out = acc / float(k*k)
                # clamp 到 [0,1]
                return np.clip(out, 0.0, 1.0)

            ksize = int(getattr(self, 'map_raster_soften_k', 0))
            iters = int(getattr(self, 'map_raster_soften_iters', 0))
            if ksize >= 2 and iters > 0:
                veh_mask = box_blur01(veh_mask, ksize, iters)
                vru_mask = box_blur01(vru_mask, ksize, iters)
            # assemble raster
            if self.map_channels >= 2:
                raster_np = np.stack([veh_mask, vru_mask], axis=0)
            else:
                raster_np = np.expand_dims(veh_mask, axis=0)
            raster = torch.from_numpy(raster_np)
            return raster, world2map
        except Exception:
            return None, None

    # ---------------- OSM_XY loader: parse polygons without lanelet2 ----------------
    def _load_osm_xy_polygons(self, path: str):
        import xml.etree.ElementTree as ET
        if not os.path.exists(path):
            raise FileNotFoundError(path)
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
        # relations
        lanelet_polys = []
        keepout_polys = []
        vru_polys = []
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
                        # build polygon by concatenating left and reversed right
                        poly = np.array(left_pts + right_pts[::-1], dtype=np.float32)
                        # classify by subtype tag (if present)
                        subtype = tags.get('subtype', '').lower()
                        if subtype in ('crosswalk','footway','sidewalk','pedestrian','pedestrian_area'):
                            vru_polys.append(poly)
                        else:
                            lanelet_polys.append(poly)
            elif rtype == 'multipolygon':
                subtype = tags.get('subtype', '')
                if subtype == 'keepout':
                    # collect outers only
                    for m in r.findall('member'):
                        if m.attrib.get('role') == 'outer' and m.attrib.get('type') == 'way':
                            wid = int(m.attrib['ref'])
                            if wid in ways:
                                pts = [nodes[nid] for nid in ways[wid] if nid in nodes]
                                if len(pts) >= 3:
                                    keepout_polys.append(np.array(pts, dtype=np.float32))
                elif subtype.lower() in ('crosswalk','footway','sidewalk','pedestrian','pedestrian_area'):
                    for m in r.findall('member'):
                        if m.attrib.get('role') == 'outer' and m.attrib.get('type') == 'way':
                            wid = int(m.attrib['ref'])
                            if wid in ways:
                                pts = [nodes[nid] for nid in ways[wid] if nid in nodes]
                                if len(pts) >= 3:
                                    vru_polys.append(np.array(pts, dtype=np.float32))
        self._map_lanelet_polys = lanelet_polys if lanelet_polys else None
        self._map_keepout_polys = keepout_polys if keepout_polys else None
        self._map_vru_polys = vru_polys if vru_polys else None
