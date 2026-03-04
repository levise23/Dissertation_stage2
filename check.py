import pandas as pd

gallery_csv = '/usr1/home/s125mdg43_07/remote/stage2dataset/gallery/gallery.csv'
uav_csv = '/usr1/home/s125mdg43_07/remote/rebuild_UAV/test_pairs.csv'

gallery_df = pd.read_csv(gallery_csv)
uav_df = pd.read_csv(uav_csv)

# 检查是否有重复
assert gallery_df['path'].is_unique, "gallery.csv path 列有重复"
assert uav_df['path'].is_unique, "test_pairs.csv path 列有重复"

# 检查顺序是否一致（举例：与真值 pairs 文件对比）
# 假设 pairs 文件有两列 path1, path2
pairs_df = pd.read_csv('/path/to/pairs.csv')
assert all(pairs_df['path1'] == gallery_df['path']), "gallery 顺序与 pairs 不一致"
assert all(pairs_df['path2'] == uav_df['path']), "UAV 顺序与 pairs 不一致"