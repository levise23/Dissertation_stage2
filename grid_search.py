import itertools
import pandas as pd
from tqdm import tqdm
from calculate import run_evaluation, DataBroker

def main():
    base_cfg = {
        'gallery_pt': 'sate02res.pt', 'query_pt': 'uav02res.pt',
        'gallery_csv': './gallery/gallery.csv',  'query_csv': 'ground_truth.csv',
        'g_x_col': 'center_x_m', 'g_y_col': 'center_y_m',
        'q_x_col': 'drone_x_m', 'q_y_col': 'drone_y_m',
        'g_name_col': 'path',       # gallery.csv 中的绝对路径列
        'q_name_col': 'drone_path',  # ground_truth.csv 中的绝对路径列
        'success_radius_m': 150.0, 'w_vis': 1.0
    }

    param_grid = {
        'window_size': [3], 'formation_offset': [5], 'top_k': [20],
        'epsilon_m': [350.0, 400.0], 'sigma': [100.0],
        'w_geo': [1.0, 1.5], 'w_angle': [0.8, 1.2]
    }

    keys, combinations = param_grid.keys(), list(itertools.product(*param_grid.values()))
    shared_broker = DataBroker(base_cfg)
    results = []
    
    for combo in tqdm(combinations, desc="搜索中"):
        cfg = base_cfg.copy()
        for i, k in enumerate(keys): cfg[k] = combo[i]
        res = run_evaluation(cfg, broker=shared_broker, verbose=False)
        if res: results.append({**cfg, **res})

    df = pd.DataFrame(results).sort_values(by='c1', ascending=False)
    df.to_csv('grid_search_results.csv', index=False)
    print("\n🏆 Top 1 组合:")
    row = df.iloc[0]
    print(f"UAV0: {row['r1_uav0']:.2f}% -> {row['c1_uav0']:.2f}% | UAV1: {row['c1_uav1']:.2f}%")

if __name__ == "__main__":
    main()