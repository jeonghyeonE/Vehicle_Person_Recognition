import os
import random
import shutil

datas_dir = 'D:/dataset/차량 및 사람 인지 영상/Training/바운딩박스/서울특별시/20201125_서울특별시_front_5_0006'

train_images_dir = 'C:/practice_coding/Vehicle_Person_Recognition/Vehicle_Person_Recognition/data/images/train'
train_labels_dir = ''
val_images_dir = ''
val_labels_dir = ''

os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

image_files = [f for f in os.listdir(datas_dir) if f.endswith('.png')]
label_files = [f for f in os.listdir(datas_dir) if f.endswith('.json')]