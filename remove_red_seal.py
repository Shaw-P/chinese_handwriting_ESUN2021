import cv2
import numpy as np
import shutil
import os
from os import walk
from os.path import join

def remove_red_seal(image):
    """
    去除红色印章
    """
    red_c_res = np.reshape(red_c, (red_c.shape[0] * red_c.shape[1], 1))

    res_mean = np.mean(red_c_res)
    res_std = np.std(red_c_res)

    filter_condition = int(np.ceil((res_mean - res_std) * 0.86))  # 去除紅線門檻
    # 红色通道
    blue_c, green_c, red_c = cv2.split(image)

    #thresh, ret = cv2.threshold(red_c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 實測調整為95%效果好一些
    # filter_condition = int(thresh * 1.1)
    filter_condition = 140 # 去除紅線門檻
    _, red_thresh = cv2.threshold(red_c, filter_condition, 255, cv2.THRESH_BINARY)
    # 把圖片轉回 3 通道
    result_img = np.expand_dims(red_thresh, axis=2)
    result_img = np.concatenate((result_img, result_img, result_img), axis=-1)
    return result_img

###  rename&copy file
path = r"D:\ESUN_2021\train_preprocess"
# word_dir = os.listdir(path)
train_path = []
for dirs in walk(path):
    train_path.append(dirs[0])
train_path.pop(0)
# print(train_path[0])
## D:\ESUN_2021\train\丁
FullPath = r"D:\ESUN_2021\train_preprocess"
# Path = r"D:\ESUN_2021\train"

for i in range(len(train_path)):
    path = train_path[i]
    file_list = []
    for files in walk(path):
        file_list.append(files[2])
        file_list = [i for item in file_list for i in item]
        for j in range(len(file_list)):
            # print(path + "\\" + file_list[j])
            img_file = file_list[j]
            img = cv2.imread(path + "\\" + img_file)
            rm_img = remove_red_seal(img)
            cv2.imwrite(path + "\\" + img_file, rm_img)
            # shutil.copyfile(path + "\\" + img, FullPath + "\\" +str(i)+ "\\" + img[:-6] +".jpg")





#
#
# img = cv2.imread(r"D:\ESUN_2021\10214.jpg")
# # cv2.imwrite('output.jpg', img)
# cv2.imshow("img", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




