import numpy as np
import cv2
import mediapipe as mp
from cvzone.FaceDetectionModule import FaceDetector
#from ultralytics import YOLO  # OFF


import warnings
warnings.filterwarnings("ignore")


detector: FaceDetector = FaceDetector()
cam: cv2.VideoCapture = cv2.VideoCapture(0)

#yolo_model = YOLO('yolov8n.pt') #OFF

MTHand = mp.solutions.hands
HendMT = MTHand.Hands()

def main():
    while True:
        ret, img = cam.read()

        if not ret or img is None:
            print("ERROR: Unable to scan face.")
            break
        if not isinstance(img, np.ndarray) or img.shape[0] == 0 or img.shape[1] == 0:
            print("ERROR: Invalid video!")
            break

        imgMT = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = HendMT.process(imgMT)
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(img, hand_landmarks, MTHand.HAND_CONNECTIONS)
                cont = 0
                for landmark in hand_landmarks.landmark:
                    if landmark.y > hand_landmarks.landmark[0].y:
                        cont += 1
                print(f"MT.G: {cont}")

        img, bboxes = detector.findFaces(img, draw=True)

       # Please, Kill-me.
       # results = yolo_model(img, stream=True)
       # for result in results:
        #    x, y, w, h = result['bbox']
         #   cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
          #  cv2.putText(img, f"{result['class']} {result['confidence']:.2f}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('MT_FACE.SCANNER ', img)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == "__main__":
    main()
