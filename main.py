import cv2                             # OpenCV 라이브러리 임포트 (컴퓨터 비전 및 실시간 영상 처리용)
from gpiozero import Buzzer            # gpiozero 라이브러리에서 Buzzer 모듈 임포트 (하드웨어 부저 제어용)
import time                            # 시간 지연 및 제어를 위한 time 모듈 임포트

buzzerPin = Buzzer(16)                 # 라즈베리파이 GPIO 16번 핀에 연결된 부저 객체 생성

def main():                            # 프로그램의 메인 로직을 담은 함수 정의
    camera = cv2.VideoCapture(-1)      # 기본 카메라(-1 또는 0) 객체 생성 및 장치 연결
    camera.set(3,640)                  # 카메라 영상 프레임의 너비(Width)를 640 픽셀로 설정
    camera.set(4,480)                  # 카메라 영상 프레임의 높이(Height)를 480 픽셀로 설정
    
    # Haar Cascade 얼굴 탐지용 사전 학습된 XML 모델 파일의 절대 경로 설정
    face_xml = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml' 
    # Haar Cascade 눈 탐지용 사전 학습된 XML 모델 파일의 절대 경로 설정
    eye_xml = cv2.data.haarcascades + 'haarcascade_eye.xml' 
    
    face_cascade = cv2.CascadeClassifier(face_xml)             # 얼굴 탐지 분류기(CascadeClassifier) 객체 생성
    eye_cascade = cv2.CascadeClassifier(eye_xml)               # 눈 탐지 분류기(CascadeClassifier) 객체 생성
    
    while( camera.isOpened() ):                                # 카메라 장치가 정상적으로 열려있는 동안 무한 반복
        _, image = camera.read()                               # 카메라로부터 프레임 한 장을 캡처하여 image 변수에 저장
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)         # 연산 속도 향상을 위해 캡처한 컬러 이미지를 흑백(Grayscale) 이미지로 변환

        # 흑백 이미지에서 설정된 파라미터 값들을 바탕으로 얼굴 탐지 수행
        faces = face_cascade.detectMultiScale(gray, 
                                              scaleFactor=1.1,                 # 탐지 윈도우 크기를 10%씩 확대하며 탐색
                                              minNeighbors=5,                  # 이웃한 탐지 결과가 최소 5번 이상 중첩될 때 유효한 얼굴로 확정
                                              minSize=(100,100),               # 탐지할 얼굴의 최소 크기를 100x100 픽셀로 제한
                                              flags=cv2.CASCADE_SCALE_IMAGE)   # 이미지 스케일링 방식을 사용하여 탐지
                                              
        print("faces detected Number: " + str(len(faces)))                     # 현재 프레임에서 탐지된 얼굴의 총 개수를 터미널에 출력

        if len(faces):                                                         # 탐지된 얼굴이 1개 이상 존재한다면 (True)
            for (x,y,w,h) in faces:                                            # 탐지된 모든 얼굴의 시작 좌표(x, y)와 크기(너비 w, 높이 h)를 순회
                cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)               # 원본 컬러 이미지의 얼굴 영역에 파란색(255,0,0) 사각형(두께 2)을 그림
                
                face_gray = gray[y:y+h, x:x+w]                                 # 눈 탐지를 위해 흑백 이미지에서 얼굴 영역만 잘라내어(ROI) 별도 추출
                face_color = image[y:y+h, x:x+w]                               # 사각형 표시를 위해 컬러 이미지에서 얼굴 영역만 잘라내어(ROI) 별도 추출
                
                # 추출된 얼굴 영역(흑백) 안에서 눈 탐지 수행
                eyes = eye_cascade.detectMultiScale(face_gray, 
                                                    scaleFactor=1.1,           # 탐지 윈도우 10%씩 확대
                                                    minNeighbors=5)            # 최소 5회 중복 탐지 시 유효한 눈으로 확정
                
                if len(eyes) <= 1:              # 탐지된 눈의 개수가 1개 이하라면 (즉, 눈을 감았거나 졸아서 눈이 안 보이는 상태로 판단)
                    buzzerPin.on()              # 부저 핀에 신호를 주어 경고음 발생
                else:                           # 탐지된 눈의 개수가 2개 이상이라면 (정상적으로 두 눈을 뜨고 있는 상태)
                    buzzerPin.off()             # 부저 핀의 신호를 차단하여 경고음 중지
                
                for (ex,ey,ew,eh) in eyes:      # 탐지된 모든 눈의 시작 좌표와 크기를 순회
                    # 컬러 이미지의 얼굴 영역(face_color) 위에 눈 위치를 초록색(0,255,0) 사각형(두께 2)으로 그림
                    cv2.rectangle(face_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2) 
        
        cv2.imshow('result', image)             # 사각형 바운딩 박스가 그려진 최종 결과 이미지 프레임을 'result'라는 이름의 창에 실시간 출력
        
        if cv2.waitKey(1) == ord('q'):          # 1ms 동안 키보드 입력을 대기하고, 입력된 키가 소문자 'q'와 일치한다면
            break                               # 무한 반복 루프를 즉시 빠져나와 프로그램 종료 절차로 이동
    
    cv2.destroyAllWindows()                     # 루프 종료 후 열려있는 모든 OpenCV 윈도우 창을 메모리에서 해제하여 닫음
    buzzerPin.off()                             # 프로그램이 강제로 종료될 때 부저가 계속 울리는 것을 방지하기 위해 안전하게 끔

if __name__ == '__main__':                      # 해당 파이썬 스크립트 파일이 다른 모듈에 의해 임포트되지 않고 직접 실행될 때만
    main()                                      # main() 함수를 호출하여 프로그램 구동
