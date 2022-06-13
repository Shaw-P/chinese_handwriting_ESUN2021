import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

#load img

img_path = r"D:\ESUN_2021\Dataset_og\train\丁\511_丁.jpg"

#load model
from keras.models import load_model
res_model = load_model('50_120_model.h5')

word_list = ['丁','三','上','不','世','丞','中','主','久','之','事','于','五','亞','交','亨','京',
             '亮','人','仁','介','仕','仙','代','仲','任','份','企','伊','伍','伯','伶','位','佑',
             '何','余','作','佩','佳','來','侑','侯','俊','保','信','修','倉','倫','偉','健','傅',
             '傑','備','傳','僑','儀','億','儒','優','允','元','兆','先','光','克','兒','內','全',
             '公','其','具','典','冠','冷','凌','凍','凱','分','利','刷','券','創','劉','力','加',
             '助','勁','勇','動','務','勝','勤','勳','包','化','北','匠','匯','區','千','升','卉',
             '卓','協','南','博','印','卿','厚','原','友','古','可','台','司','合','吉','同','名',
             '君','吟','吳','呂','呈','告','周','和','品','員','哲','唐','商','問','啟','善','喜',
             '喬','嘉','器','國','園','圓','圖','團','土','地','坊','坤','城','培','基','堂','堅',
             '堡','堯','報','場','塑','境','士','壽','多','大','天','太','央','奇','奕','女','好',
             '如','妤','妮','妹','委','姚','姜','姿','威','娟','娥','婉','婕','婷','媒','媛','嬌',
             '子','孟','孫','學','宇','守','安','宋','宏','宗','定','宜','客','宣','室','宥','家',
             '宸','容','密','富','寓','實','寧','寬','寶','專','小','尚','局','居','屋','展','屬',
             '山','岳','峰','峻','崇','川','工','巧','巨','巫','市','希','帝','師','常','平','年',
             '幸','幼','店','庭','康','廈','廖','廠','廣','廷','建','弘','張','強','彥','彩','彬',
             '彭','彰','影','律','徐','得','御','復','德','心','志','忠','念','思','怡','恆','恩',
             '悅','惠','意','愛','愷','慈','慧','慶','憲','憶','應','成','戴','戶','房','所','扶',
             '承','技','投','拉','拓','振','捷','揚','政','敏','教','敦','敬','整','數','文','斌',
             '料','斯','新','方','施','旅','日','旭','旺','旻','昆','昇','昌','明','易','昕','星',
             '映','春','昭','昱','昶','時','晉','晟','晨','普','景','晴','晶','智','暉','暘','曉',
             '曜','書','曹','曾','會','月','有','服','朝','期','木','本','朱','李','材','村','杜',
             '杰','東','松','林','果','枝','柏','柔','柯','格','桂','桃','梁','梅','械','棋','森',
             '楊','業','榕','榮','樂','樓','樹','樺','橋','機','權','欣','欽','款','歐','正','武',
             '毅','毓','民','氣','水','永','江','池','汽','沈','沙','沛','油','治','泉','泓','法',
             '泰','洋','津','洪','洲','活','流','浩','海','消','涵','淑','淨','添','清','游','湯',
             '源','漁','漢','潔','潘','潤','澄','澤','瀚','灣','炳','煌','煒','煜','照','燕','營',
             '燦','燿','爾','牙','物','特','玉','王','玫','玲','玻','珊','珍','珠','班','珮','球',
             '理','琇','琦','琪','琳','琴','瑄','瑋','瑛','瑜','瑞','瑩','璃','璟','環','瓊','生',
             '產','用','田','申','男','療','登','發','白','百','皇','皓','盈','益','盛','盟','盧',
             '真','眾','睿','石','研','碧','碩','磊','社','祐','神','祥','祺','禎','福','禮','禾',
             '秀','私','秉','秋','科','租','秦','程','穎','空','立','竑','章','竣','竹','策','筱',
             '管','築','簡','籌','米','精','系','紀','紅','紋','純','紘','紙','素','紹','統','絲',
             '經','綠','維','網','綸','綺','緯','縣','總','織','羅','美','群','義','羽','翁','翊',
             '翔','翠','翰','耀','聖','聚','聯','聰','聲','職','股','育','胡','能','腦','膠','自',
             '致','臺','臻','興','舒','舜','航','良','色','艾','芝','芬','花','芳','芷','芸','苑',
             '苗','若','英','茂','范','茶','茹','莉','莊','菁','菊','華','菱','萍','萬','萱','葉',
             '董','葳','蓁','蓉','蓮','蔡','蔣','蕙','蕭','薇','薛','藍','藝','藥','蘇','蘭','處',
             '虹','融','行','術','衛','衣','裕','裝','製','西','視','覽','觀','言','計','訊','託',
             '記','設','許','診','詠','詩','詮','詹','語','誠','誼','調','謙','謝','證','護','谷',
             '豐','豪','貝','貞','財','貨','貴','買','貿','賀','賃','資','賓','賜','賢','賴','超',
             '趙','車','軒','輝','輪','辰','農','迅','迪','通','造','連','進','逸','運','道','達',
             '遠','邦','邱','郁','郎','部','郭','鄧','鄭','酒','醫','采','里','重','金','鈞','鈦',
             '鈴','鈺','鉅','銀','銓','銘','銷','鋁','鋐','鋒','鋼','錦','鍾','鎧','鎮','鏡','鐘',
             '鐵','鑫','鑽','長','門','開','閎','閔','關','防','阿','限','陞','院','陳','陽','隆',
             '際','險','雄','雅','集','雨','雪','雯','雲','電','震','霖','霞','青','靖','靜','韋',
             '音','韻','頂','順','顏','顧','顯','風','飛','食','飲','飾','餐','館','首','香','馨',
             '馬','馮','駿','騰','體','高','魏','鳳','鴻','鵬','麒','麗','麟','黃','黎','鼎','齊',
             '龍','isnull']
             
def remove_red_seal(image):
    """
    去除红色印章
    """
    # 獲得红色通道
    blue_c, green_c, red_c = cv2.split(image)

    # 多傳入一個参數cv2.THRESH_OTSU，並且把閥值thresh設為0，算法會找到最優閥值
    #thresh, ret = cv2.threshold(red_c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 实测调整为95%效果好一些
    #filter_condition = int(thresh * 1.1)

    red_c_res = np.reshape(red_c, (red_c.shape[0]*red_c.shape[1], 1))

    res_mean = np.mean(red_c_res)
    res_std  = np.std(red_c_res)

    filter_condition = int(np.ceil((res_mean - res_std)*0.86)) # 去除紅線門檻
    _, red_thresh = cv2.threshold(red_c, filter_condition, 255, cv2.THRESH_BINARY)

    # 把圖片轉回 3 通道
    result_img = np.expand_dims(red_thresh, axis=2)
    result_img = np.concatenate((result_img, result_img, result_img), axis=-1)

    return result_img



#cv2.imwrite("C:\\Users\\Red\\Desktop\\T_brain\\Dataset\\PP\\" + str(word_dir[dir]), mask)

# cv2.imencode('.jpg', mask)[1].tofile(save_path + "\\" + word_dir[dir] + "\\" + str(inner_dir[inner]))

 
H_Big = 90  # 67
W_Big = 175  # 155
count = 0

def fill_space:
    image = cv2.imdecode(np.fromfile(image_loads[image_num], dtype=np.uint8), -1)
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.shape[0] < H_Big:
        padding = np.ones(( (H_Big - image.shape[0]), image.shape[1]))*255
        image = np.vstack((image, padding))
    if image.shape[1] < W_Big:
        fill_left = math.ceil((W_Big - image.shape[1]) / 2) 
        fill_right = (W_Big - image.shape[1]) - fill_left

        fill_left_ones = np.ones(( image.shape[0], fill_left))*255
        fill_right_ones = np.ones(( image.shape[0], fill_right))*255

        image = np.hstack((fill_left_ones, image))
        image = np.hstack((image, fill_right_ones))
    if image.shape[0] > H_Big and image.shape[1] > W_Big:
        image = cv2.resize(image, (H_Big, W_Big), interpolation=cv2.INTER_AREA)
    image = img_to_array(image)
    
    predict = res_model.predict(image)
    if np.max(predict) > 0.85:
        result_idx = np.argmax(res_model.predict(image), axis=1)
        return word_list[result_idx]
    else:
        return word_list[800]
        
 
 
image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
rm_img = remove_red_seal(image)
mask_img = cv2.cvtColor(rm_img, cv2.COLOR_BGR2GRAY)
image = fill_space(mask_img)
predict = res_model.predict(image)
if np.max(predict) > 0.85:
    result_idx = np.argmax(res_model.predict(image), axis=1)
    res_model_pred_result = word_list[result_idx]
else:
    res_model_pred_result = word_list[800]
    
print(res_model_pred_result)