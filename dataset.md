这个部分流程：
1.从卫星图中切片
/usr1/home/s125mdg43_07/remote/stage2dataset/picture_precess_offline.py
数据来源：tif 02.csv 
记得修改其中的卫星参数,包括切片大小，得到的是切片集合gallery


2.从图中提取特征向量，目前用的是02.tif 也就是一开始的test 13部分
/usr1/home/s125mdg43_07/remote/stage2dataset/extract_feature.py
他来读取刚才的gallery_output/csv，将所有gallery_patch提取为特征，保存在stae_{idx}_res.pt
cd ~/remote/stage2dataset
python extract_feature.py \
    --csv_path '/usr1/home/s125mdg43_07/remote/stage2dataset/gallery/gallery.csv' \
    --output_pt 'sate02res.pt' \
    --weight_path '/usr1/home/s125mdg43_07/remote/rebuild/Dissertation/checkpoints/curriculum_s2_o1/net_020.pth' 

还要从传过来的切片中读取图片，将切片转化为特征 保存在uav{idx}res.pt
python extract_feature.py \
    --csv_path '/usr1/home/s125mdg43_07/remote/rebuild_UAV/test_pairs.csv' \
    --output_pt '/usr1/home/s125mdg43_07/remote/stage2dataset/uav02res.pt' \
    --weight_path '/usr1/home/s125mdg43_07/remote/rebuild/Dissertation/checkpoints/curriculum_s2_o1/net_020.pth' 

3.真值建构
/usr1/home/s125mdg43_07/remote/stage2dataset/build_ground_truth.py
现在是参数写死，后面要改成可调

4.使用calculate.py计算集群的定位，现在的R@1能有0.35 集群优化后的R@1有0.58
将其匹配结果作为样本对，放进特征提取（ai调用库）fine_localize.py,输出match_pairs_for_fine.csv
给出精准定位结果