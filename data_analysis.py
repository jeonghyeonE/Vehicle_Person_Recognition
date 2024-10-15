import json
import cv2
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def put_korean_text(image, text, position, font_path='data/NanumGothic.ttf', font_size=20, color=(0, 0, 255)):
    # OpenCV 이미지를 PIL 이미지로 변환
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)

    # 텍스트 추가
    draw.text(position, text, font=font, fill=color)

    # PIL 이미지를 다시 OpenCV 이미지로 변환
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img

# JSON 파일과 이미지가 있는 디렉토리 설정
json_dir = 'Vehicle_Person_Recognition/data/labels/train'  # JSON 파일들이 저장된 디렉토리
image_dir = 'Vehicle_Person_Recognition/data/images/train'  # 이미지 파일들이 저장된 디렉토리
output_dir = 'Vehicle_Person_Recognition/data/output'  # 바운딩박스가 그려진 이미지를 저장할 디렉토리

# 출력 디렉토리가 없는 경우 생성
os.makedirs(output_dir, exist_ok=True)

# 이미지 파일과 JSON 파일의 매칭을 위해 동일한 이름이 있다고 가정
json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')]

# JSON 파일과 이미지 파일 순회
for json_file in json_files:
    # JSON 파일 로드
    json_path = os.path.join(json_dir, json_file)
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # JSON 파일명에서 이미지 파일명 추출
    image_file = data['filename']
    image_path = os.path.join(image_dir, image_file)

    # 이미지 파일이 있는지 확인
    if not os.path.exists(image_path):
        print(f"Image file {image_file} not found, skipping.")
        continue

    # PIL을 사용하여 이미지 로드
    image = Image.open(image_path)

    # 이미지 데이터를 OpenCV 형식으로 변환
    image = np.array(image)

    # BGR로 변환 (PIL은 RGB 형식이므로 OpenCV의 BGR 형식으로 변환 필요)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 바운딩 박스 그리기
    for annotation in data['annotations']:
        points = annotation['points']
        label = annotation['label']

        # 바운딩 박스 좌표 추출 (왼쪽 상단, 오른쪽 하단)
        top_left = (int(points[0][0]), int(points[0][1]))
        bottom_right = (int(points[2][0]), int(points[2][1]))

        # 바운딩 박스 그리기 (색상: 파랑, 두께: 2)
        cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)

        # 레이블 텍스트 추가
        # cv2.putText(image, label, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        image = put_korean_text(image, label, (top_left[0], top_left[1] - 30))

    # 결과 이미지 출력
    cv2.imshow(f'Image: {image_file}', image)

    # 이미지 저장 경로
    output_image_path = os.path.join(output_dir, f'output_{image_file}')
    
    # 이미지 저장
    cv2.imwrite(output_image_path, image)

    # 이미지 창 닫기
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print(f"Processed and saved bounding box images to {output_dir}")
