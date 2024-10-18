from ultralytics import YOLO

def train_yolo():
    model = YOLO("yolov8s.pt")
    model.train(data='Vehicle_Person_Recognition/data.yaml',  # YAML 파일 경로
            epochs=50,           # 학습 에포크 수
            imgsz=640,           # 입력 이미지 크기
            batch=32,            # 배치 크기
            name='yolo_vehicle_person',  # 학습 결과 저장 폴더 이름
            workers=4)

if __name__ == "__main__":
    train_yolo()