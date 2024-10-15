import json
import os
import cv2
import numpy as np
from ultralytics import YOLO

# IoU (Intersection over Union) 계산 함수
def calculate_iou(box1, box2):
    # box1, box2는 [x_min, y_min, x_max, y_max] 형식의 바운딩 박스
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # 교차 영역 계산
    x_min_inter = max(x1_min, x2_min)
    y_min_inter = max(y1_min, y2_min)
    x_max_inter = min(x1_max, x2_max)
    y_max_inter = min(y1_max, y2_max)

    inter_area = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter)

    # 두 박스의 전체 영역 계산
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # IoU 계산
    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou

# JSON에서 실제 바운딩 박스 정보 추출
def extract_bounding_boxes_from_json(json_path, image_width, image_height):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    gt_boxes = []
    for annotation in data['annotations']:
        points = annotation['points']
        x_min, y_min = points[0]
        x_max, y_max = points[2]

        # 이미지 크기에 맞게 좌표 변환
        x_min = int(x_min)
        y_min = int(y_min)
        x_max = int(x_max)
        y_max = int(y_max)

        gt_boxes.append([x_min, y_min, x_max, y_max])

    return gt_boxes

# 바운딩 박스 그리기 함수 (좌표를 정수로 변환해야 함)
def draw_bounding_boxes(image, boxes, color, label):
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)  # 좌표를 정수로 변환
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# 테스트 이미지 폴더 및 JSON 파일 경로
test_image_dir = 'Vehicle_Person_Recognition/data/test_data/images'
test_json_dir = 'Vehicle_Person_Recognition/data/test_data/json'

# 학습된 YOLOv8 모델 불러오기
model = YOLO('runs/detect/yolo_vehicle_person/weights/best.pt')

# 테스트 이미지 파일 순회
for image_file in os.listdir(test_image_dir):
    if not (image_file.endswith('.jpg') or image_file.endswith('.png')):
        continue

    image_path = os.path.join(test_image_dir, image_file)
    json_file = image_file.replace('.jpg', '.json').replace('.png', '.json')
    json_path = os.path.join(test_json_dir, json_file)

    if not os.path.exists(json_path):
        print(f"JSON file for {image_file} not found, skipping.")
        continue

    # 이미지 로드 및 크기 정보 얻기
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # 원본 이미지를 복제하여 각각 모델 예측, JSON 정보, 합친 이미지를 만들기 위한 이미지
    model_image = image.copy()
    json_image = image.copy()
    combined_image = image.copy()

    # JSON에서 실제 바운딩 박스 정보 추출
    gt_boxes = extract_bounding_boxes_from_json(json_path, width, height)

    # 모델 예측
    results = model.predict(image_path)

    # 모델 예측 바운딩 박스 얻기
    predicted_boxes = []
    for result in results[0].boxes.xyxy:
        predicted_boxes.append(result.tolist())  # [x_min, y_min, x_max, y_max]

    # JSON 바운딩 박스 그리기 (초록색)
    draw_bounding_boxes(json_image, gt_boxes, (0, 255, 0), "Ground Truth")
    draw_bounding_boxes(combined_image, gt_boxes, (0, 255, 0), "Ground Truth")

    # 모델 예측 바운딩 박스 그리기 (파란색)
    draw_bounding_boxes(model_image, predicted_boxes, (255, 0, 0), "Prediction")
    draw_bounding_boxes(combined_image, predicted_boxes, (255, 0, 0), "Prediction")

    # 이미지 출력 (3개 이미지)
    cv2.imshow(f'Model Prediction: {image_file}', model_image)
    cv2.imshow(f'Ground Truth (JSON): {image_file}', json_image)
    cv2.imshow(f'Combined: {image_file}', combined_image)

    # # 이미지 저장 경로 설정
    # output_dir = 'test_results'
    # os.makedirs(output_dir, exist_ok=True)

    # # 결과 이미지 저장
    # cv2.imwrite(os.path.join(output_dir, f'model_{image_file}'), model_image)
    # cv2.imwrite(os.path.join(output_dir, f'json_{image_file}'), json_image)
    # cv2.imwrite(os.path.join(output_dir, f'combined_{image_file}'), combined_image)

    # 이미지 창 닫기
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("Test images processed and results saved.")
