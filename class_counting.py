import json
import os
from collections import defaultdict

# JSON 파일이 저장된 디렉토리 설정
json_dir = 'Vehicle_Person_Recognition/data/labels_json/train'  # JSON 파일들이 저장된 디렉토리

# 각 label의 등장 횟수를 저장할 딕셔너리
labels_count = defaultdict(int)

# JSON 파일 순회
for json_file in os.listdir(json_dir):
    if not json_file.endswith('.json'):
        continue

    # JSON 파일 로드
    json_path = os.path.join(json_dir, json_file)
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 각 JSON 파일의 annotations에서 label을 추출
    for annotation in data['annotations']:
        label = annotation['label']
        labels_count[label] += 1  # 해당 label의 카운트를 증가

# 결과 출력
print("Class count per label:")
for label, count in labels_count.items():
    print(f"{label}: {count} occurrences")

# 딕셔너리를 파일로 저장 (원하는 경우)
# output_file = 'class_count.json'
# with open(output_file, 'w', encoding='utf-8') as f:
#     json.dump(labels_count, f, ensure_ascii=False, indent=4)

# print(f"Class count saved to {output_file}")
