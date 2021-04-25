import cv2
import numpy as np
import sys
import time
import dlib
class Controller():
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.frame = None
        self.faces = None
        self.cap = cv2.VideoCapture(0)
        #open phone camera API
        #self.address = "https://192.168.2.3:8080/video"
        #self.cap.open(self.address)
        self.gray = None
        self.cutEyesLoc = None
    def main(self):

        while True:

            self.readSelf()

            self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            self.faces = self.detector(self.gray)
            self.lanmarkingLeftEyes()
            
            cv2.imshow("Frame", self.frame)
            if self.cutEyesLoc is not None:
                cv2.imshow("cutting frame", self.cutEyesLoc)
            
            key = cv2.waitKey(1)
            if key == 27:
                break
        self.cap.release()
        cv2.destroyAllWindows()
    def readSelf(self):
        _, self.frame=self.cap.read()
    def lanmarkingLeftEyes(self):
        for face in self.faces:
            #x = face.left()
            #y = face.top()
            #x1 = face.right()
            #y1 = face.bottom()
            landmarks = self.predictor(self.gray, face)
            horizontalLineLeft = (landmarks.part(36).x, landmarks.part(36).y)
            horizontalLineRight = (landmarks.part(39).x, landmarks.part(39).y)
            
            verticalLineTop = (landmarks.part(38).x, landmarks.part(38).y)
            verticalLineBottom = (landmarks.part(40).x, landmarks.part(40).y)
            self.focusEye(landmarks.part(36).x,landmarks.part(38).y,landmarks.part(39).x,landmarks.part(40).y)
            cv2.line(self.frame, horizontalLineLeft,horizontalLineRight,(0,255,0),1)
            cv2.line(self.frame, verticalLineTop,verticalLineBottom,(0,255,0),1)
            
    def focusEye(self,x,y,x1,y1):
        self.cutEyesLoc = self.frame[y:y1, x:x1] 
        self.cutEyesLoc = cv2.cvtColor(self.cutEyesLoc, cv2.COLOR_BGR2GRAY)
        #_, self.cutEyesLoc = cv2.threshold(self.cutEyesLoc, 5,255,cv2.THRESH_BINARY)
        self.cutEyesLoc = cv2.resize(self.cutEyesLoc,None,fx=5,fy=5)

        #self.cutEyesLoc = cv2.GaussianBlur(self.cutEyesLoc, (7,7), 0)

        #video kaydı
        #self._fourcc = VideoWriter_fourcc(*'MP4V')
        #self._out = VideoWriter("has.mp4", self._fourcc, 20.0, (240,120))
    def getCameraShape(self):     
        for i in range(3):
            print(self.frame.shape[i])

if __name__ == "__main__":
    ct = Controller()
    ct.main()