import json
import os

# JSON 파일이 저장된 디렉토리 설정
json_dir = 'Vehicle_Person_Recognition/data/labels_json/train'  # JSON 파일들이 저장된 디렉토리

# 모든 label들을 저장할 집합 (중복 제거를 위해 집합 사용)
labels_set = set()

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
        labels_set.add(label)  # 고유한 label을 집합에 추가

# 집합을 리스트로 변환하고 인덱스에 따라 클래스 번호 매핑
labels_list = sorted(list(labels_set))  # 정렬된 리스트로 변환
class_dict = {label: idx for idx, label in enumerate(labels_list)}

# 결과 출력
print("Generated class dictionary:")
print(class_dict)

# 딕셔너리를 파일로 저장 (원하는 경우)
output_file = 'class_dict.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(class_dict, f, ensure_ascii=False, indent=4)

print(f"Class dictionary saved to {output_file}")
