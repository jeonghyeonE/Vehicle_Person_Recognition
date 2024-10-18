import os

def delete_files_from_txt(txt_file, folder_path):
    # 1. txt 파일에서 삭제할 파일 이름 목록을 읽습니다.
    with open(txt_file, 'r', encoding='ANSI') as f:
        file_list = f.read().splitlines()
    
    # 2. 폴더 안에서 파일들을 확인하고, 목록에 있는 파일들을 삭제합니다.
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")

# 사용 예시
txt_file = 'C:/practice_coding/Vehicle_Person_Recognition/data/unmatched_txt.txt'  # 파일 이름이 적힌 txt 파일 경로
folder_path = 'D:/dataset/차량 및 사람 인지 영상/Training/바운딩박스/대전광역시/data/labels_json'  # 파일이 저장된 폴더 경로

delete_files_from_txt(txt_file, folder_path)
