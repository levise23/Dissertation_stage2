import itertools
import pandas as pd
import time
from tqdm import tqdm
from calculate import run_evaluation, DataBroker

def main():
    # 1. 基础配置
    base_cfg = {
        'gallery_pt': 'sate02res.pt', 
        'query_pt': 'uav02res.pt',
        'gallery_csv': 'gallery/gallery.csv', 
        'query_csv': 'ground_truth.csv',
        'g_x_col': 'center_x_m', 'g_y_col': 'center_y_m',
        'q_x_col': 'drone_x_m', 'q_y_col': 'drone_y_m',
        'success_radius_m': 150.0, 
        'w_vis': 1.0
    }

    # 2. 参数搜索网格 (补全了 sigma)
    param_grid = {
        'window_size': [3, 5],             # 帧数
        'formation_offset': [5],           # 步长
        'top_k': [20],                     # 搜索深度，保持在20以利用60%+的潜力
        'epsilon_m': [200.0, 300.0],       # 几何门限
        'sigma': [100.0, 150.0],           # 软奖励平滑度 (补上这个键)
        'w_geo': [1.0, 1.5],               # 几何基础权重
        'w_angle': [1.0, 2.0, 3.0],        # 方向一致性权重
        'w_skip': [0.5, 1.0]               # 跨帧长程约束权重
    }

    keys = param_grid.keys()
    combinations = list(itertools.product(*param_grid.values()))
    
    # 3. 预加载数据
    print("[*] 正在一次性预加载数据到内存...")
    shared_broker = DataBroker(base_cfg)

    results = []
    start_time = time.time()
    
    # 4. 执行穷举
    print(f"[*] 启动网格搜索，共 {len(combinations)} 种组合...")
    for combo in tqdm(combinations, desc="正在压榨 60% 潜力"):
        cfg = base_cfg.copy()
        for i, k in enumerate(keys): 
            cfg[k] = combo[i]
            
        res = run_evaluation(cfg, broker=shared_broker, verbose=False)
        if res: 
            results.append({**cfg, **res})

    # 5. 保存与结果展示
    if not results:
        print("❌ 搜索完成但无有效结果，请检查数据路径或参数。")
        return

    df = pd.DataFrame(results).sort_values(by='c1', ascending=False)
    df.to_csv('grid_search_results.csv', index=False)
    
    print(f"\n✅ 搜索完成，总耗时: {time.time()-start_time:.1f}s")
    print("\n🏆 全局前 3 名最优参数组合 (重点看 C@1):")
    for _, row in df.head(3).iterrows():
        print("-" * 60)
        print(f"参数: K={int(row['window_size'])}, Eps={row['epsilon_m']}, w_angle={row['w_angle']}, w_geo={row['w_geo']}, w_skip={row['w_skip']}")
        print(f"单机: R@1:{row['r1']:.2f}% | R@10:{row['r10']:.2f}%")
        print(f"协同: C@1:{row['c1']:.2f}% | C@10:{row['c10']:.2f}%")
        print(f"🚀 绝对提升 (C@1 vs R@1): {row['improvement']:+.2f}%")

if __name__ == "__main__":
    main()