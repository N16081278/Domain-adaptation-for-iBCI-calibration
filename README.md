# Domain Adaptation of Transfer Learning for intracortical Brain-Computer Interface Calibration
# 基於遷移學習之域適應實現侵入式腦機介面之模型校正方案
## 作者: 王錦虔 Ogk；技術支援: 黃俊叡、邱柏雄
***
# 數據集
## data: https://zenodo.org/record/583331
---
### ./data/: 前處理過的數據集(以64ms為 bin width)
### ./experiments_code_for_server: 一些code
### ./MMD/MMD.ipynb: 計算兩兩session間的最大均值差異(MMD)
### ./OGK_domainAdaptation/: 一些用pytorch的code (裡面有fixed model)
### ./results/: 一些儲存結果
### ./Trajectory/posTrajectory.ipynb: 不同記錄時長的軌跡圖
### ./Trajectory/polarArrow.ipynb: 不同記錄時長的移動向量
___
### average_spikeWaveform : 各channel平均spike波形
### averge_firingRate: 各channel平均firing rate(依陣列方式排列)
### changeMeanFiringRate: 不同session各channel的平均firing rate
### channelArray: 猴子Indy在M1的電極通道編號
### cross_test: 串接過去資料作訓練對當前測試結果的影響
### firingRate: Firing rate heatmap
### MMD_fixed: Maximum mean discrepancy 與 解碼性能之關係
### overall_result: 各種校正方法訓練資料及性能總覽
### pannel_trajectory: 平板樣式與運動軌跡
### raw_filter: 原始訊號raw data與濾波後訊號
### source_target_tSNE: 相鄰兩個session的分布，包含target session的training data(虔320秒)與testing data
### spikeTrain_kinematic: spike train與對應時間的運動狀態
### spikeTrain: Spike train
### train_continuousFinetune: continuous Fine-tune (在第一個session訓練，於之後的模型微調)
### train_domainAdverserial: Domain Adversarial calibration 域對抗校正
### train_domainComfuse: Domain confusion calibration 域混淆校正
### train_finetune: model fine-tune (pretrain-finetune)
### train_mixCalibration: cross session之訓練與預測(串聯2個相鄰session, source&target)
### train_onTarget: Daily retrain之訓練與測試
### trainDatalength: 訓練資料量對解碼性能之影響