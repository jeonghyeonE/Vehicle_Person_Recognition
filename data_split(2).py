import os
import shutil

def move_files_recursive(base_dir, image_ext, label_ext, dst_images_dir, dst_labels_dir):
    # 디렉토리 생성
    os.makedirs(dst_images_dir, exist_ok=True)
    os.makedirs(dst_labels_dir, exist_ok=True)

    # 재귀적으로 모든 하위 폴더 탐색
    for root, dirs, files in os.walk(base_dir):
        # 이미지 및 라벨 파일 찾기
        image_files = sorted([f for f in files if f.endswith(image_ext)])
        label_files = sorted([f for f in files if f.endswith(label_ext)])

        # 이미지만 있는 경우도 있고, 라벨만 있는 경우도 있으므로 각각 따로 처리
        moved_files_count = 0

        # 이미지 파일 이동
        for img_file in image_files:
            img_src = os.path.join(root, img_file)
            img_dst = os.path.join(dst_images_dir, img_file)

            shutil.move(img_src, img_dst)
            moved_files_count += 1

        # 라벨 파일 이동
        for lbl_file in label_files:
            lbl_src = os.path.join(root, lbl_file)
            lbl_dst = os.path.join(dst_labels_dir, lbl_file)

            shutil.move(lbl_src, lbl_dst)
            moved_files_count += 1

    print(f'Moved {moved_files_count} files from {base_dir} to the target directories.')

# 이미지와 라벨을 저장할 경로 설정
dst_images_dir = 'D:/dataset/차량 및 사람 인지 영상/Training/바운딩박스/서울특별시/data/images'
dst_labels_dir = 'D:/dataset/차량 및 사람 인지 영상/Training/바운딩박스/서울특별시/data/labels_json'

# 기본 폴더 경로 설정 (예시)
base_dir = 'D:/dataset/차량 및 사람 인지 영상/Training/바운딩박스/서울특별시/새 폴더'

# 파일 이동 실행
move_files_recursive(base_dir, '.png', '.json', dst_images_dir, dst_labels_dir)
