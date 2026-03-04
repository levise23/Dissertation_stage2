import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import math

class DataBroker:
    def __init__(self, cfg, max_top_k=50):
        self.cfg = cfg
        print("[*] 正在加载数据并预计算相似度...")
        
        # 加载卫星地库
        g_df = pd.read_csv(cfg['gallery_csv'])
        self.g_names = g_df[cfg.get('g_name_col', 'path')].values  # 绝对路径列
        self.g_coords_array = g_df[[cfg['g_x_col'], cfg['g_y_col']]].values
        g_data = torch.load(cfg['gallery_pt'], map_location='cpu')
        self.g_feats = (g_data['features'] if isinstance(g_data, dict) else g_data).cuda().float()
        
        # 加载无人机查询
        q_df = pd.read_csv(cfg['query_csv'])
        self.q_names = q_df[cfg.get('q_name_col', 'drone_path')].values # 绝对路径列
        self.q_coords_array = q_df[[cfg['q_x_col'], cfg['q_y_col']]].values
        q_data = torch.load(cfg['query_pt'], map_location='cpu')
        self.q_feats = (q_data['features'] if isinstance(q_data, dict) else q_data).cuda().float()

        with torch.no_grad():
            sim_matrix = torch.matmul(self.q_feats, self.g_feats.T)
            scores, indices = torch.topk(sim_matrix, max_top_k, dim=-1)
            self.q_topk_scores = scores.cpu().numpy()
            self.q_topk_indices = indices.cpu().numpy()

        diff = self.g_coords_array[:, np.newaxis, :] - self.g_coords_array[np.newaxis, :, :]
        self.g_dist_matrix = np.linalg.norm(diff, axis=-1)

class KViterbiReranker:
    def __init__(self, broker, cfg):
        self.broker = broker
        self.top_k, self.epsilon_m, self.sigma = cfg['top_k'], cfg['epsilon_m'], cfg['sigma']
        self.w_vis, self.w_geo = cfg['w_vis'], cfg['w_geo']
        self.w_angle = cfg.get('w_angle', 0.8)

    def solve(self, q_indices, gt_coords):
        K = len(q_indices)
        states = [{'idx': self.broker.q_topk_indices[q, :self.top_k], 'scores': self.broker.q_topk_scores[q, :self.top_k]} for q in q_indices]
        dp = np.full((K, self.top_k), -float('inf'))
        backtrack = np.zeros((K, self.top_k), dtype=int)
        
        dp[0] = self.w_vis * states[0]['scores']
        for k in range(1, K):
            vec_rtk = gt_coords[k] - gt_coords[k-1]
            dist_rtk = np.linalg.norm(vec_rtk)
            for curr in range(self.top_k):
                id_c = states[k]['idx'][curr]
                for prev in range(self.top_k):
                    if dp[k-1][prev] == -float('inf'): continue
                    map_dist = self.broker.g_dist_matrix[id_c, states[k-1]['idx'][prev]]
                    err = abs(map_dist - dist_rtk)
                    if err > self.epsilon_m: continue
                    geo_reward = math.exp(-(err**2) / (2 * self.sigma**2))
                    if map_dist > 1e-6:
                        vec_map = self.broker.g_coords_array[id_c] - self.broker.g_coords_array[states[k-1]['idx'][prev]]
                        cos_sim = np.dot(vec_rtk, vec_map) / ((dist_rtk+1e-6) * map_dist)
                        geo_reward += self.w_angle * (max(0, cos_sim) ** 2)
                    score = dp[k-1][prev] + self.w_vis * states[k]['scores'][curr] + self.w_geo * geo_reward
                    if score > dp[k][curr]: dp[k][curr], backtrack[k][curr] = score, prev

        best_path = []
        curr = np.argmax(dp[K-1])
        if dp[K-1][curr] == -float('inf'):
            best_path = [states[k]['idx'][0] for k in range(K)]
        else:
            for k in range(K-1, -1, -1):
                best_path.append(states[k]['idx'][curr])
                curr = backtrack[k][curr]
            best_path.reverse()
        return best_path

def run_evaluation(cfg, broker=None, verbose=True):
    if broker is None: broker = DataBroker(cfg)
    solver = KViterbiReranker(broker, cfg)
    K, offset = cfg['window_size'], cfg['formation_offset']
    total, max_step = len(broker.q_coords_array), (K - 1) * offset
    if total <= max_step: return None

    metrics = {f'r1_uav{k}': 0 for k in range(K)}
    metrics.update({f'c1_uav{k}': 0 for k in range(K)})
    metrics.update({'r1':0, 'r10':0, 'c1':0})
    match_records = []
    count = 0
    
    for i in (tqdm(range(total - max_step)) if verbose else range(total - max_step)):
        q_indices = [i + k*offset for k in range(K)]
        gt_coords = [broker.q_coords_array[q] for q in q_indices]
        best_path = solver.solve(q_indices, gt_coords)
        
        # 记录路径对（绝对路径）
        match_records.append({
            'uav_path': broker.q_names[q_indices[0]],
            'sate_path': broker.g_names[best_path[0]],
            'retrieved_x': broker.g_coords_array[best_path[0]][0],
            'retrieved_y': broker.g_coords_array[best_path[0]][1],
            'gt_x': gt_coords[0][0], 'gt_y': gt_coords[0][1]
        })

        # 统计精度
        for k in range(K):
            s_top1 = broker.q_topk_indices[q_indices[k], 0]
            if np.linalg.norm(broker.g_coords_array[s_top1] - gt_coords[k]) < cfg['success_radius_m']:
                metrics[f'r1_uav{k}'] += 1
            if np.linalg.norm(broker.g_coords_array[best_path[k]] - gt_coords[k]) < cfg['success_radius_m']:
                metrics[f'c1_uav{k}'] += 1
        
        # 修复 NameError: 统计首机 R@10
        s_top10 = broker.q_topk_indices[q_indices[0], :10]
        if any(np.linalg.norm(broker.g_coords_array[idx] - gt_coords[0]) < cfg['success_radius_m'] for idx in s_top10):
            metrics['r10'] += 1
        count += 1

    res = {k: (v/count)*100 for k, v in metrics.items()}
    res['r1'], res['c1'] = res['r1_uav0'], res['c1_uav0']
    res['improvement'] = res['c1'] - res['r1']
    pd.DataFrame(match_records).to_csv('match_pairs_for_fine.csv', index=False)
    if verbose: print(f"[*] 结果已保存，首机提升: +{res['improvement']:.2f}%")
    return res