import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

#data = cv2.imread(r"C:\Users\Red\Desktop\2718.jpg")

#data_mask = np.zeros((data.shape[0], data.shape[1], 1))
#NR = np.copy(data[:,:,2])
#NR[NR > 160] = 0

# -*- encoding: utf-8 -*-
import cv2
import numpy as np
import os


def remove_red_seal(image):
    """
    去除红色印章
    """

    # 获得红色通道
    blue_c, green_c, red_c = cv2.split(image)

    # 多传入一个参数cv2.THRESH_OTSU，并且把阈值thresh设为0，算法会找到最优阈值
    #thresh, ret = cv2.threshold(red_c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 实测调整为95%效果好一些
    #filter_condition = int(thresh * 1.1)

    red_c_res = np.reshape(red_c, (red_c.shape[0]*red_c.shape[1], 1))

    res_mean = np.mean(red_c_res)
    res_std  = np.std(red_c_res)



    filter_condition = int(np.ceil((res_mean - res_std)*0.86)) # 去除紅線門檻
    _, red_thresh = cv2.threshold(red_c, filter_condition, 255, cv2.THRESH_BINARY)

    # 把图片转回 3 通道
    result_img = np.expand_dims(red_thresh, axis=2)
    result_img = np.concatenate((result_img, result_img, result_img), axis=-1)

    return result_img



path = "C:\\Users\\Red\\Desktop\\T_brain\\Dataset\\train"
word_dir = os.listdir(path)

save_path = "C:\\Users\\Red\\Desktop\\T_brain\\Dataset\\train_preprocess_filter_automatic"
for word in range(0, len(word_dir)):
    if not os.path.exists(save_path + "\\" + word_dir[word]):
        os.makedirs(save_path + "\\" + word_dir[word])


for dir in range(0, len(word_dir)):

    inner_dir = os.listdir(path + "\\" + word_dir[dir])

    for inner in range(0, len(inner_dir)):
        image = cv2.imdecode(np.fromfile(path + "\\" + word_dir[dir] + "\\" + inner_dir[inner], dtype=np.uint8), -1)
        rm_img = remove_red_seal(image)
        mask = cv2.cvtColor(rm_img, cv2.COLOR_BGR2GRAY)
        #cv2.imwrite("C:\\Users\\Red\\Desktop\\T_brain\\Dataset\\PP\\" + str(word_dir[dir]), mask)

        cv2.imencode('.jpg', mask)[1].tofile(save_path + "\\" + word_dir[dir] + "\\" + str(inner_dir[inner]))



#
# import shutil

# from os import walk
# from os.path import join
#
# MyPath  = r"C:\Users\Red\Desktop\ding"# 當下目錄
# # KeyWord = input('請輸入檔案關鍵字:')
# file_name = []
# for root, dirs, files in walk(MyPath):
#     count = 0
#     for i in files:
#         file_name.append(i)
#         FullPath = join(root, i)
# # FullPath = r"C:\Users\Red\Desktop\ding"
#         KeyWord = files[count][:-6]
#         shutil.copy(FullPath, 'C:\\Users\\Red\\Desktop\\ding_process\\' + KeyWord + ".jpg")
#         count +=1
