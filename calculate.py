import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import math

# ==========================================
# 1. 极速数据中心 (Data Broker)
# ==========================================
class DataBroker:
    def __init__(self, cfg, max_top_k=50):
        self.cfg = cfg
        print("[*] 初始化 DataBroker，执行预计算...")
        
        # 加载 Gallery
        g_data = torch.load(cfg['gallery_pt'], map_location='cpu')
        g_feats = g_data['features'] if isinstance(g_data, dict) and 'features' in g_data else g_data
        self.g_feats = g_feats.cuda().float() if torch.cuda.is_available() else g_feats.float()
        
        self.g_coords_array = pd.read_csv(cfg['gallery_csv'])[[cfg['g_x_col'], cfg['g_y_col']]].values
        
        # 加载 Query
        q_data = torch.load(cfg['query_pt'], map_location='cpu')
        q_feats = q_data['features'] if isinstance(q_data, dict) and 'features' in q_data else q_data
        self.q_feats = q_feats.cuda().float() if torch.cuda.is_available() else q_feats.float()
        
        self.q_coords_array = pd.read_csv(cfg['query_csv'])[[cfg['q_x_col'], cfg['q_y_col']]].values

        # 预计算相似度 Top-K
        with torch.no_grad():
            sim_matrix = torch.matmul(self.q_feats, self.g_feats.T)
            scores, indices = torch.topk(sim_matrix, max_top_k, dim=-1)
            self.q_topk_scores = scores.cpu().numpy()
            self.q_topk_indices = indices.cpu().numpy()

        # 预计算底库两两距离方阵
        diff = self.g_coords_array[:, np.newaxis, :] - self.g_coords_array[np.newaxis, :, :]
        self.g_dist_matrix = np.linalg.norm(diff, axis=-1)

# ==========================================
# 2. 协同重排序求解器 (KViterbiReranker)
# ==========================================
class KViterbiReranker:
    def __init__(self, broker, cfg):
        self.broker = broker
        self.top_k = cfg['top_k']
        self.epsilon_m = cfg['epsilon_m']
        self.sigma = cfg['sigma']
        self.w_vis = cfg['w_vis']
        self.w_geo = cfg['w_geo']
        
        # 权重读取自 cfg，支持外部调参
        self.w_angle = cfg.get('w_angle', 1.5)
        self.w_skip = cfg.get('w_skip', 0.8)

    def solve_and_rerank(self, q_indices, gt_coords):
        K = len(q_indices)
        states = [{'idx': self.broker.q_topk_indices[q, :self.top_k], 
                   'scores': self.broker.q_topk_scores[q, :self.top_k]} for q in q_indices]

        dp = np.full((K, self.top_k), -float('inf'))
        backtrack = np.zeros((K, self.top_k), dtype=int)
        roots = np.full((K, self.top_k), -1, dtype=int)

        dp[0] = self.w_vis * states[0]['scores']
        for i in range(self.top_k): roots[0][i] = i 

        for k in range(1, K):
            # --- 预计算本帧的 RTK 物理指标 ---
            vec_rtk = gt_coords[k] - gt_coords[k-1]
            dist_rtk = np.linalg.norm(vec_rtk)
            v1_norm = dist_rtk + 1e-6
            
            # 跨帧 RTK 预计算
            if k >= 2:
                vec_skip_rtk = gt_coords[k] - gt_coords[k-2]
                dist_skip_rtk = np.linalg.norm(vec_skip_rtk)

            for curr in range(self.top_k):
                id_curr = states[k]['idx'][curr]
                coord_curr = self.broker.g_coords_array[id_curr]
                
                for prev in range(self.top_k):
                    if dp[k-1][prev] == -float('inf'): continue
                    
                    id_prev = states[k-1]['idx'][prev]
                    coord_prev = self.broker.g_coords_array[id_prev]
                    
                    # 1. 查表获取地图位移
                    map_dist = self.broker.g_dist_matrix[id_curr, id_prev]
                    err = abs(map_dist - dist_rtk)
                    if err > self.epsilon_m: continue
                    
                    # 2. 几何位移奖励
                    geo_reward = math.exp(-(err**2) / (2 * self.sigma**2))
                    
                    # 3. 航向角一致性 (利用查表得到的 map_dist 提速)
                    if map_dist > 1e-6:
                        vec_map = coord_curr - coord_prev
                        cos_sim = np.dot(vec_rtk, vec_map) / (v1_norm * map_dist)
                        # 【核心优化】：使用更陡峭的奖励函数，过滤微小偏航
                        if cos_sim > 0.7: # 约 45 度以内
                            geo_reward += self.w_angle * (cos_sim ** 3)
                        else:
                            geo_reward -= self.w_angle * 0.5 # 航向不对直接倒扣分

                    # 4. 跨帧二阶约束
                    if k >= 2:
                        id_pp = states[k-2]['idx'][backtrack[k-1][prev]]
                        dist_skip_map = self.broker.g_dist_matrix[id_curr, id_pp]
                        err_skip = abs(dist_skip_map - dist_skip_rtk)
                        geo_reward += self.w_skip * math.exp(-(err_skip**2) / (2 * self.sigma**2))

                    score = dp[k-1][prev] + self.w_vis * states[k]['scores'][curr] + self.w_geo * geo_reward
                    
                    if score > dp[k][curr]:
                        dp[k][curr] = score
                        backtrack[k][curr] = prev
                        roots[k][curr] = roots[k-1][prev]

        # 计算首机重排得分
        final_scores = np.full(self.top_k, -float('inf'))
        for i in range(self.top_k):
            if dp[K-1][i] > -float('inf'):
                r = roots[K-1][i]
                if dp[K-1][i] > final_scores[r]: final_scores[r] = dp[K-1][i]

        return states[0]['idx'][np.argsort(-final_scores)]

# ==========================================
# 3. 评测引擎 (Evaluator)
# ==========================================
def run_evaluation(cfg, broker=None, verbose=True):
    if broker is None: broker = DataBroker(cfg)
    solver = KViterbiReranker(broker, cfg)
    
    total = len(broker.q_coords_array)
    K, offset = cfg['window_size'], cfg['formation_offset']
    max_step = (K - 1) * offset
    if total <= max_step: return None

    metrics = {'r1':0, 'r5':0, 'r10':0, 'c1':0, 'c5':0, 'c10':0}
    count = 0
    iterator = tqdm(range(total - max_step)) if verbose else range(total - max_step)

    for i in iterator:
        q_idx = [i + k*offset for k in range(K)]
        gt_coords = [broker.q_coords_array[q] for q in q_idx]
        collab_topk = solver.solve_and_rerank(q_idx, gt_coords)
        
        gt_0 = gt_coords[0]
        single_topk = broker.q_topk_indices[q_idx[0]]

        def check(indices, top_n):
            for idx in indices[:top_n]:
                if np.linalg.norm(broker.g_coords_array[idx] - gt_0) < cfg['success_radius_m']:
                    return True
            return False

        if check(single_topk, 1): metrics['r1'] += 1
        if check(single_topk, 5): metrics['r5'] += 1
        if check(single_topk, 10): metrics['r10'] += 1
        if check(collab_topk, 1): metrics['c1'] += 1
        if check(collab_topk, 5): metrics['c5'] += 1
        if check(collab_topk, 10): metrics['c10'] += 1
        count += 1

    res = {k: (v/count)*100 for k, v in metrics.items()}
    res['improvement'] = res['c1'] - res['r1']
    
    if verbose:
        print("\n" + "="*50)
        print(f"🎯 协同定位报告 (K={K}, Off={offset})")
        print(f"单机: R@1:{res['r1']:.2f}% | R@10:{res['r10']:.2f}%")
        print(f"协同: C@1:{res['c1']:.2f}% | C@10:{res['c10']:.2f}%")
        print(f"📈 提升: +{res['improvement']:.2f}%")
        print("="*50)
    return res

if __name__ == "__main__":
    CONFIG = {
        'gallery_pt': 'sate02res.pt', 'query_pt': 'uav02res.pt',
        'gallery_csv': 'gallery/gallery.csv', 'query_csv': 'ground_truth.csv',
        'g_x_col': 'center_x_m', 'g_y_col': 'center_y_m',
        'q_x_col': 'drone_x_m', 'q_y_col': 'drone_y_m',
        'window_size': 3, 'formation_offset': 5, 'success_radius_m': 150.0,
        'top_k': 15, 'epsilon_m': 200.0, 'sigma': 150.0, 'w_vis': 1.0, 'w_geo': 0.8
    }
    run_evaluation(CONFIG)