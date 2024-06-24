# 玉山人工智慧挑戰賽 2021 夏季賽 TBrain AI Summer Competition 2021
## 中文手寫影像辨識
隊員名單：益鴻、宗朋
* [比賽官網](https://tbrain.trendmicro.com.tw/Competitions/Details/14)
## 競賽成果
模型準確率近 80%，並且獲得 57/468 名之成績。

## 模型
* VGG16(pretrained,no-pretrained)
* VGG19(pretrained,no-pretrained)
## 技巧
* Data Augmentation
  * Basic image manipulations(flip, rotation...)
* 開源手寫中文圖片資料集
  * [CASIA](https://nlpr.ia.ac.cn/databases/handwriting/Home.html)
* brightness and contrast adjustment
  * OpenCV 
## 檔案說明
### 1. API Server 部署檔案(`final_project.py`)
參考玉山官方提供的[API範例](https://github.com/Esun-DF/ai_competition_api_sharedoc)進行修改，最後部署至雲端(GCP)
### 2. 模型訓練程式碼(`main_code_colab.ipynb`)
此為模型訓練程式碼，主要以 VGG16 模型為主架構進行模型訓練

## Reference：
### 參數設定
https://www.gushiciku.cn/pl/2B5Z/zh-tw
### Data Augumentation
https://zhuanlan.zhihu.com/p/53367135
https://blog.csdn.net/qq_32768091/article/details/78735140
### brightness and contrast adjustment
https://blog.csdn.net/wl_Honest/article/details/107569135

