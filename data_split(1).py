import os
import shutil

def move_train_files(train_image_files, train_label_files, train_data_dir, train_images_dir, train_labels_dir):
    # 이미지 및 라벨 파일 이름 기준으로 정렬
    train_image_files.sort()
    train_label_files.sort()

    # 이미지 및 라벨 파일 매칭
    matched_files = [(img, lbl) for img, lbl in zip(train_image_files, train_label_files) if img.replace('.png', '') == lbl.replace('.json', '')]

    # 파일 이동 함수
    def move_files(file_list, src_dir, dst_img_dir, dst_lbl_dir):
        for img_file, lbl_file in file_list:
            # 이미지 파일 이동
            shutil.move(os.path.join(src_dir, img_file), os.path.join(dst_img_dir, img_file))
            # 라벨 파일 이동
            shutil.move(os.path.join(src_dir, lbl_file), os.path.join(dst_lbl_dir, lbl_file))

    # 훈련 파일 이동
    move_files(matched_files, train_data_dir, train_images_dir, train_labels_dir)

    print(f'Moved {len(matched_files)} files to the train directory.')

def move_val_files(val_image_files, val_label_files, val_image_data_dir, val_label_data_dir, val_images_dir, val_labels_dir):
    # 이미지 및 라벨 파일 이름 기준으로 정렬
    val_image_files.sort()
    val_label_files.sort()

    # 이미지 및 라벨 파일 매칭
    matched_files = [(img, lbl) for img, lbl in zip(val_image_files, val_label_files) if img.replace('.png', '') == lbl.replace('.json', '')]

    # 파일 이동 함수
    def move_files(file_list, src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir):
        for img_file, lbl_file in file_list:
            # 이미지 파일 이동
            shutil.move(os.path.join(src_img_dir, img_file), os.path.join(dst_img_dir, img_file))
            # 라벨 파일 이동
            shutil.move(os.path.join(src_lbl_dir, lbl_file), os.path.join(dst_lbl_dir, lbl_file))

    # 검증 파일 이동
    move_files(matched_files, val_image_data_dir, val_label_data_dir, val_images_dir, val_labels_dir)

    print(f'Moved {len(matched_files)} files to the validation directory.')


# train 파일 이동
# train_data_dir = 'D:/dataset/차량 및 사람 인지 영상/Training/바운딩박스/서울특별시/새 폴더'

# train_images_dir = 'C:/practice_coding/Vehicle_Person_Recognition/data/images/train'
# train_labels_dir = 'C:/practice_coding/Vehicle_Person_Recognition/data/labels_json/train'

# os.makedirs(train_images_dir, exist_ok=True)
# os.makedirs(train_labels_dir, exist_ok=True)

# train_image_files = [f for f in os.listdir(train_data_dir) if f.endswith('.png')]
# train_label_files = [f for f in os.listdir(train_data_dir) if f.endswith('.json')]


# val 파일 이동
val_image_data_dir = 'D:/dataset/차량 및 사람 인지 영상/Validation/바운딩박스/대구광역시/[원천]대구광역시_FLRR_2_01/20201209_대구광역시_left_4_0178'
val_label_data_dir = 'D:/dataset/차량 및 사람 인지 영상/Validation/바운딩박스/대구광역시/[라벨]대구광역시_FLRR_2_01/20201209_대구광역시_left_4_0178'

val_images_dir = 'C:/practice_coding/Vehicle_Person_Recognition/data/images/val'
val_labels_dir = 'C:/practice_coding/Vehicle_Person_Recognition/data/labels_json/val'

os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# val 디렉토리 내 이미지 및 라벨 파일 리스트
val_image_files = [f for f in os.listdir(val_image_data_dir) if f.endswith('.png')]
val_label_files = [f for f in os.listdir(val_label_data_dir) if f.endswith('.json')]

# train 사용 예시
# move_train_files(train_image_files, train_label_files, train_data_dir, train_images_dir, train_labels_dir)

# val 사용 예시
move_val_files(val_image_files, val_label_files, val_image_data_dir, val_label_data_dir, val_images_dir, val_labels_dir)