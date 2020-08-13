## ML100Days
### 1. 機器學習概論 Machine Learning Introduction
### 從概念上理解機器學習的目的與限制，並瞭解機器學習流程
- Day_001: 如何評估資料
- Day_002: 瞭解機器學習的應用
- Day_003: 閱讀機器學習文章，瞭解其專案目標、技術與資料來源
- Day_004: 熟悉Python擷取資料的操作

### 2. 資料清理與數據前處理Processing
- Day_005: 建立DataFrame, 使用Request抓取資料
- Day_006: 資料編碼one hot encoding
- Day_007: 觀察資料類型，以Kaggle鐵達尼生存預測資料為例
- Day_008: 統計值與直方圖
- Day_009: 檢視Outlier
- Day_010: 刪除Outlier，以Kaggle房價預測資料為例
- Day_011: 以中位數和眾數取代Outlier、數據標準化
- Day_012: 補缺失值與標準化
- Day_013: DataFrame操作、資料表串接

### 3. 探索式數據分析EDA
- Day_014: 相關係數Correlation
- Day_015: EDA from Correlation
- Day_016: EDA 不同數值範圍間的特徵如何檢視/繪圖與樣式Kernel Density Estimation (KDE)
- Day_017: EDA 簡化連續型變數
- Day_018: EDA 把連續型變數離散化
- Day_019: 資料視覺化_多圖檢視 Subplots
- Day_020: 資料視覺化_熱像圖/格狀圖 Heatmap & Grid-plot
- Day_021: 模型初體驗 Logistic Regression

### 4.特徵工程 Feature Engineering
- Day_022: 特徵工程流程架構
- Day_023: 數值型特徵_去除偏態
- Day_024: 類別型特徵_基礎處理
- Day_025: 類別型特徵_均值編碼
- Day_026: 類別型特徵_其他進階處理(計數編碼、雜湊編碼)
- Day_027: 時間型特徵
- Day_028: 特徵組合_數值與數值組合
- Day_029: 特徵組合_類別與數值組合
- Day_030: 特徵選擇
- Day_031: 特徵評估
- Day_032: 分類型特徵優化_葉編碼

### 5.機器學習基礎模型建立 Model Selection
- Day_033: 機器學習的定義、Overfit是什麼及如何解決
- Day_034: 訓練/測試集切分
- Day_035: regression vs. classification
- Day_036: 評估指標選定 evaluation metrics
- Day_037: regression model: 線性回歸/羅吉斯回歸
- Day_038: 使用Scikit-learn撰寫線性回歸/羅吉斯回歸的程式碼
- Day_039: regression model: LASSO回歸/Ridge回歸
- Day_040: 使用Scikit-learn撰寫LASSO回歸/Ridge回歸的程式碼
- Day_041: tree based model 理論基礎
- Day_042: tree based 程式碼撰寫
- Day_043: tree based model - 隨機森林(Random Forest)模型理論基礎
- Day_044: 使用Scikit-learn撰寫隨機森林(Random Forest)模型的程式碼
- Day_045: tree based model - 梯度提升機(Gradient Boosting Machine)模型理論基礎
- Day_046: 使用Scikit-learn撰寫梯度提升機(Gradient Boosting Machine)模型的程式碼

### 6. 參數調整與集成 Fine-tuning and Ensemble
- Day_047: 超參數調整與優化 Hyper-paramter
- Day_048: 如何參加Kaggle競賽
- Day_049: 集成方法: 混和泛化 Blending
- Day_050: 集成方法: 堆疊泛化 Stacking
- Day_051-053: 期中考: 透過Kaggle完成資料科學專案(ML與調參相關)

### 7. 非監督式機器學習 Unsupervised Learning
### 利用分群和降維方法探索資料模式 Clustering and Dimension Reduction
- Day_054: clustering 1 非監督式機器學習簡介與應用場景
- Day_055: clustering 2 聚類算法 K-means
- Day_056: K-means觀察: 使用輪廓分析
- Day_057: clustering 3 階層分群算法
- Day_058: 階層分群法觀察: 使用2D樣版資料集(非監督評估方法)
- Day_059: dimension reduction 1 降維方法: 主成分分析PCA
- Day_060: PCA觀察: 使用手寫辨識資料及
- Day_061: dimension reduction 2 降維方法: T-SNE
- Day_062: T-SNE觀察: 分群與流形還原

### 8. 深度學習 Deep Learning
### 深度神經網路 Deep Neural Networks (DNN)
- Day_063: 類神經歷史與深度學習概念
- Day_064: 模型調整與學習曲線 TensorFlow PlayGround
- Day_065: 啟動函數與正規化
- Day_066: Keras安裝與介紹
- Day_067: Keras embedded dataset 介紹與應用
- Day_068: 序列模型搭建網路 Keras Sequential API
- Day_069: Keras Module API 介紹與應用
- Day_070: 深度神經網路的發展、架構與優缺點
- Day_071: 損失函數 Loss Function
- Day_072: 啟動函數 Activation Function
- Day_073-074: 梯度下降 Gradient Descent
- Day_075: 反向式傳播 BackPropagation
- Day_076: 優化器 Optimizers

### 訓練神經網路的細節與技巧
- Day_077: Validation and overfit
- Day_078: 訓練神經網路前的注意事項(資料處理、運算資源、超參數設置)
- Day_079: Learning rate effect
- Day_080: 搭配不同的優化器與學習率進行神經網路訓練
- Day_081: 正規化 Regularization
- Day_082: 機移除 Dropout
- Day_083: 批次標準化 Batch normalization
- Day_084: 正規化/機移除/批次標準化 的組合與比較
- Day_085: 使用 callbacks 函數做 earlystop
- Day_086: 使用 callbacks 函數儲存 model
- Day_087: 使用 callbacks 函數做 reduce learning rate
- Day_088: 撰寫自己的 callbacks 函數
- Day_089: 撰寫自己的 Loss function

- Day_090: 使用傳統電腦視覺與機器學習進行影像辨識
- Day_091: 應用傳統電腦視覺方法 + 機器學習進行CIFAR-10分類

### 卷積神經網路 Convolutional Neural Networks (CNN)
- Day_092: CNN的重要性與組成結構
- Day_093: CNN為什麼比DNN更適合處理影像問題
- Day_094: 卷積層原理與參數說明
- Day_095: 池化層原理與參數說明
- Day_096: Keras中常用的CNN layers
- Day_097: 透過CNN 訓練CIFAR-10並比較其與DNN的差異

### 訓練卷積神經網路的細節與處理技巧
- Day_098: 處理大量數據 (使用Python的生成器generator)
- Day_099: 處理小量數據 (使用資料增強提升準確率)
- Day_100: 遷移學習 Transfer Learning

- Day_101-103: 期末考:透過Kaggle影像辨識測驗，綜合應用深度學習的課程內容，並體驗遷移學習的威力

### 進階補充
- Day_104: 史丹佛線上 ConvNetJS
- Day_105: CNN 卷積網路的有趣應用
- Day_106: 電腦視覺常用公開資料集
- Day_107: CNN 卷積網路在實際生活中的應用案例


課程內容來源: [CUPOY機器學習百日馬拉松](https://www.accupass.com/event/2004230939451973735324)
