import os

def check_file_pairs(image_dir, json_dir, image_ext='.png', json_ext='.txt'):
    # 이미지와 JSON 파일 리스트 가져오기
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(image_ext)])
    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith(json_ext)])

    # 확장자 제거하고 이름만 추출
    image_names = {os.path.splitext(f)[0] for f in image_files}
    json_names = {os.path.splitext(f)[0] for f in json_files}

    # 쌍이 맞는 파일들 확인
    matched_files = image_names.intersection(json_names)
    
    # 쌍이 없는 이미지 및 라벨 파일 확인
    unmatched_images = image_names - json_names
    unmatched_jsons = json_names - image_names

    # 결과 출력
    print(f"Total images: {len(image_files)}")
    print(f"Total JSON files: {len(json_files)}")
    print(f"Matched pairs: {len(matched_files)}")
    print(f"Unmatched images: {len(unmatched_images)}")
    print(f"Unmatched JSON files: {len(unmatched_jsons)}")

    # 쌍이 맞지 않는 이미지 파일을 txt로 저장
    if unmatched_images:
        with open('C:/practice_coding/Vehicle_Person_Recognition/data/unmatched_images.txt', 'w') as f:
            for img in unmatched_images:
                f.write(f"{img}{image_ext}\n")

    # 쌍이 맞지 않는 JSON 파일을 txt로 저장
    if unmatched_jsons:
        with open('C:/practice_coding/Vehicle_Person_Recognition/data/unmatched_txt.txt', 'w') as f:
            for lbl in unmatched_jsons:
                f.write(f"{lbl}{json_ext}\n")
    
    return len(unmatched_images), len(unmatched_jsons)

# 이미지 및 JSON 파일 경로 설정
image_dir = 'C:/practice_coding/Vehicle_Person_Recognition/data/images/val'
json_dir = 'C:/practice_coding/Vehicle_Person_Recognition/data/labels/val'

# 파일 쌍 확인 실행
unmatched_images_count, unmatched_jsons_count = check_file_pairs(image_dir, json_dir)
