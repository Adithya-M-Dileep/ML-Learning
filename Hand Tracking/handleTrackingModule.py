import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self):
        self.mpHand = mp.solutions.hands
        self.hands = self.mpHand.Hands()
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHand.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return lmList


hd = handDetector()

ptime = 0
ctime = 0

cap = cv2.VideoCapture(1)

tip = [8, 12, 16, 20]

while True:
    success, img = cap.read()
    # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = hd.findHands(img)
    h, w, c = img.shape

    ctime = time.time()
    fps = int(1/(ctime-ptime))
    ptime = ctime

    cv2.putText(img, str(fps), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    lmList = hd.findPosition(img, draw=False)

    if lmList:
        final = []
        if lmList[4][1] > lmList[3][1]:
            final.append(1)
        else:
            final.append(0)

        for id in tip:
            if lmList[id][2] < lmList[id-2][2]:
                final.append(1)
            else:
                final.append(0)
        print("Counts:", sum(final))

    cv2.imshow("Image", img)
    cv2.waitKey(1)
