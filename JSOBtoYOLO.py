import json
import os

# JSON 파일과 이미지가 있는 디렉토리 설정
json_dir = 'Vehicle_Person_Recognition/data/labels/val'  # JSON 파일들이 저장된 디렉토리
image_dir = 'Vehicle_Person_Recognition/data/images/val'  # 이미지 파일들이 저장된 디렉토리
output_label_dir = 'Vehicle_Person_Recognition/data/labels_yolo/val'  # YOLO 형식으로 저장할 레이블 디렉토리

# 출력 디렉토리가 없는 경우 생성
os.makedirs(output_label_dir, exist_ok=True)

# 클래스 목록 정의 (필요에 따라 수정)
class_dict = {
    "목적차량(특장차)": 0,
    "보행자": 1,
    "이륜차": 2,
    "일반차량": 3
}

# JSON 파일 순회
for json_file in os.listdir(json_dir):
    if not json_file.endswith('.json'):
        continue

    # JSON 파일 로드
    json_path = os.path.join(json_dir, json_file)
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 이미지 파일 이름 추출 (확장자를 제거하여 .txt 파일 이름으로 사용)
    image_file = data['filename']
    txt_file_name = os.path.splitext(image_file)[0] + '.txt'

    # 이미지 파일 경로 및 크기 확인
    image_path = os.path.join(image_dir, image_file)
    if not os.path.exists(image_path):
        print(f"Image file {image_file} not found, skipping.")
        continue

    # 이미지 크기 추출 (이미지가 실제로 있는 경우 OpenCV 등으로 크기를 확인할 수 있음)
    image_width = int(data['camera']['resolution_width'])
    image_height = int(data['camera']['resolution_height'])

    # YOLO 형식 레이블 저장 파일 경로
    txt_output_path = os.path.join(output_label_dir, txt_file_name)

    # YOLO 형식으로 변환된 데이터를 저장할 리스트
    yolo_labels = []

    # 각 주석(annotation)에 대해 YOLO 형식으로 변환
    for annotation in data['annotations']:
        label = annotation['label']
        if label not in class_dict:
            print(f"Label {label} not in class_dict, skipping.")
            continue
        
        # 클래스 번호
        class_id = class_dict[label]

        # 좌표 (x_min, y_min, x_max, y_max)를 YOLO 형식으로 변환
        points = annotation['points']
        x_min, y_min = points[0]
        x_max, y_max = points[2]

        # YOLO 형식으로 변환 (중심 좌표와 너비/높이로 변환)
        x_center = (x_min + x_max) / 2.0 / image_width
        y_center = (y_min + y_max) / 2.0 / image_height
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height

        # YOLO 형식: [클래스 번호] [x_center] [y_center] [width] [height]
        yolo_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")

    # YOLO 형식 텍스트 파일 저장
    with open(txt_output_path, 'w') as f:
        f.write("\n".join(yolo_labels))

    print(f"YOLO formatted labels saved to {txt_output_path}")
