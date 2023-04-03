import torch
import numpy as np
import cv2
from time import time
import keyboard
import yaml
import cv2
import mediapipe as mp
from djitellopy import tello

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

class ObjectDetection:




    def __init__(self,capture_index,model_name):
        self.count =0
        self.capture_index = capture_index
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n\nDevice Used:", self.device)



    def hand_detection(self,cap,frame):
        with mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
            while cap.isOpened():
                image = frame
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Initially set finger count to 0 for each cap
                fingerCount = 0

                if results.multi_hand_landmarks:

                    for hand_landmarks in results.multi_hand_landmarks:
                        # Get hand index to check label (left or right)
                        handIndex = results.multi_hand_landmarks.index(hand_landmarks)
                        handLabel = results.multi_handedness[handIndex].classification[0].label

                        # Set variable to keep landmarks positions (x and y)
                        handLandmarks = []

                        # Fill list with x and y positions of each landmark
                        for landmarks in hand_landmarks.landmark:
                            handLandmarks.append([landmarks.x, landmarks.y])

                        # Test conditions for each finger: Count is increased if finger is
                        #   considered raised.
                        # Thumb: TIP x position must be greater or lower than IP x position,
                        #   deppeding on hand label.
                        if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:
                            fingerCount = fingerCount + 1
                        elif handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]:
                            fingerCount = fingerCount + 1

                        # Other fingers: TIP y position must be lower than PIP y position,
                        #   as image origin is in the upper left corner.
                        if handLandmarks[8][1] < handLandmarks[6][1]:  # Index finger
                            fingerCount = fingerCount + 1
                        if handLandmarks[12][1] < handLandmarks[10][1]:  # Middle finger
                            fingerCount = fingerCount + 1
                        if handLandmarks[16][1] < handLandmarks[14][1]:  # Ring finger
                            fingerCount = fingerCount + 1
                        if handLandmarks[20][1] < handLandmarks[18][1]:  # Pinky
                            fingerCount = fingerCount + 1

                        # Draw hand landmarks
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())

                # Display finger count
                return fingerCount

    def get_video_capture(self):
        return cv2.VideoCapture(self.capture_index)

    def load_model(self,model_name):

        if model_name:

            model = torch.hub.load('/Users/northman/.cache/torch/hub/ultralytics_yolov5_master','custom', path='best.pt',force_reload=True,source='local')
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):

        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)

        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):

        return self.classes[int(x)]

    def plot_boxes(self, results, frame):

        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.32:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                    row[3] * y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                print("{} what {}".format(labels[i],self.class_to_label(labels[i])))
                if labels[i] == 1 :
                    self.count+=1




        return frame

    def __call__(self):

        me = tello.Tello()
        me.connect()
        print(me.get_battery())
        me.streamon()
        cap = self.get_video_capture()
        assert cap.isOpened()
        begin_smoke =0


        while True:

            ret, frame = cap.read()

            assert ret

            frame = cv2.resize(frame,(416,416))
            start_time = time()
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            fingerCount = self.hand_detection(cap,frame)
            end_time = time()
            fps = 1 / np.round(end_time - start_time, 3)
            cv2.putText(frame,str(fingerCount), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.imshow("img", frame)
            if self.count >= 10 :
                print("Definity Smoking")
                if fingerCount == 10 :
                     print("Stop smoking")
                     self.count =0



            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()


# Create a new object and execute.
detection = ObjectDetection(capture_index=0,model_name='best.pt')
detection()

