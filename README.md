基于光流计算的人群异常分析

数据集：https://github.com/hosseinm/med

标签：详见上述仓库内matlab脚本，标签为逐帧标记，原始脚本与视频有1-2帧的差，修订后的标签生成脚本见本仓库```labeling.m```

计算方法：
1. 对原始视频进行缩放，缩放倍数为0.25（可采用其他缩放比例）
2. 计算视频的光流，采用cv2.calcOpticalFlowFarneback
3. 对每帧进行分类计算
4. 可考虑上下文依赖关系（多帧进行序列分类）
5. 光流不进行归一化处理

预处理：
1. dataloader中（见main函数内示例）：
   ```flows, flow_index = flow_load(flow_root)```进行光流预处理
2. 训练代码：```mmodel_train```

## TODO：
1. 对光流计算结果进行可视化
2. 训练metric可视化

