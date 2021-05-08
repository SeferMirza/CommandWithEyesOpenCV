import cv2
import numpy as np
import sys, getopt
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
        self.address = None#"https://192.168.43.1:8080/video"
        self.threshold = 25
        self.grayFrame = None
        self.cutEyes = None
        self.cutEyesGray = None
        self.contours = None
        self.capThreshold = None
        self.left_eye = None
        self.maskEyes = None
        self.landmarks = None
        self.min_x = None
        self.max_x = None
        self.min_y = None
        self.max_y = None
        self.otps = None
        self.args = None
        self.cameraIs = False
        self.thresholdIs = False
        self.rgbIs = False
        self.eyeLinesIs = False
        self.fx = 1
        self.fy = 1
        self.check = True
        self.calibrationIsOkey = False
        self.testImage = None
        self.background = None
        self.eyeCenter = None
        self.centerArray = np.array([[0,0]])
        self.centerMean = None
        self.leftArray = np.array([[0,0]])
        self.leftMean = None
        self.rightArray = np.array([[0,0]])
        self.rightMean = None
        self.topArray = np.array([[0,0]])
        self.topMean = None
        self.bottomArray = np.array([[0,0]])
        self.bottomMean = None
    def getOtps(self):
        try:
            self.otps, self.args = getopt.getopt(sys.argv[1:],"h:c:t:r:a:e:",["help","cameradress","threshold","rgb","eyeline","halfcut","quartercut"])
        except getopt.GetoptError as err:
            print(err)
            sys.exit()
            #self.otps = []
    def nothing(self,x):
        pass
    def main(self):
        self.getOtps()
        for otp, arg in self.otps:

            if otp == '-a':
                self.address = str(arg)
                self.cap.open(self.address)
            elif otp == '--threshold':
                self.thresholdIs = True
                for ot , ar in self.otps:
                    if ot == '-t':
                        self.threshold = int(ar)
            elif (otp == '-r' and arg == 'True'):
                self.rgbIs = True
            elif otp == '-e' and arg == 'True':
                self.eyeLinesIs = True 
            elif otp == '-c' and arg == 'True':
                self.cameraIs = True
            elif otp == '--halfcut':
                self.fx = 0.5
                self.fy = 0.5
            elif otp == '--quartercut':
                self.fx = 0.25
                self.fy = 0.25
                
        #TODO
        #print(self.otps, self.args)
        #self.optimizationEyesLooking()
            
        while True:
            self.readSelf()
            self.frame = cv2.resize(self.frame,None,fx=self.fx,fy=self.fy)
            #TODO
            self.frame = cv2.rotate(self.frame,cv2.ROTATE_90_COUNTERCLOCKWISE)
            self.grayFrame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            self.faces = self.detector(self.grayFrame)
            self.optimizationEyesLooking()
            self.lanmarkingLeftEyes()

            if self.calibrationIsOkey == True:
                self.centerArray = self.centerArray[1:]
                self.bottomArray = self.bottomArray[1:]
                self.topArray = self.topArray[1:]
                self.leftArray = self.leftArray[1:]
                self.rightArray = self.rightArray[1:]
                self.centerMean = self.centerArray.mean(0)
                self.bottomMean = self.bottomArray.mean(0)
                self.topMean =  self.topArray.mean(0)
                self.leftMean = self.leftArray.mean(0)
                self.rightMean = self.rightArray.mean(0)
                self.lookingPointDrawCircle()
            if self.eyeLinesIs == True:
                self.eyeLines()
            if self.cameraIs == True:
                cv2.imshow("Frame", self.frame)
            #if self.cutEyesLoc is not None:
            #    cv2.imshow("cutting frame", self.capThreshold)
            
            
            key = cv2.waitKey(1)
            if key == 27:
                #print(self.eyeCenter)
                break
        self.cap.release()
        cv2.destroyAllWindows()
    def showImage(self):
        self.testImage = cv2.imread('anime.jpg')
        imageH, imageW, imageChannels= capImg.shape
        cv2.circle(self.testImage, ( (self.eyeCenter[0] * imageW) / self.rightMean[0], (self.eyeCenter[1] * imageH) / self.bottomMean[1]))
    def lookingPointDrawCircle(self):
        pass#cv2.circle()
    def readSelf(self):
        _, self.frame=self.cap.read()
    def lanmarkingLeftEyes(self):
        for face in self.faces:
            #x = face.left()
            #y = face.top()
            #x1 = face.right()
            #y1 = face.bottom()
            self.landmarks = self.predictor(self.grayFrame, face)
            self.left_eye = np.array([(self.landmarks.part(36).x, self.landmarks.part(36).y),
                                (self.landmarks.part(37).x, self.landmarks.part(37).y),
                                (self.landmarks.part(38).x, self.landmarks.part(38).y),
                                (self.landmarks.part(39).x, self.landmarks.part(39).y),
                                (self.landmarks.part(40).x, self.landmarks.part(40).y),
                                (self.landmarks.part(41).x, self.landmarks.part(41).y)], np.int32)
            
            h, w, _ = self.frame.shape
            mask = np.zeros((h, w), np.uint8)
            cv2.polylines(mask, [self.left_eye], True, 255, 2)
            cv2.fillPoly(mask, [self.left_eye], 255)
            self.maskEyes = cv2.bitwise_and(self.grayFrame, self.grayFrame, mask=mask)
            self.maskEyes = np.where(self.maskEyes==0, 255,self.maskEyes)

            self.min_x = np.min(self.left_eye[:,0])
            self.max_x = np.max(self.left_eye[:,0])
            self.min_y = np.min(self.left_eye[:,1])
            self.max_y = np.max(self.left_eye[:,1])

            self.cutEyes = self.maskEyes[self.min_y : self.max_y, self.min_x : self.max_x]
            self.cutEyes = cv2.resize(self.cutEyes, None, fx=5, fy=5)

            self.capThreshold = cv2.GaussianBlur(self.cutEyes, (5,5), 0)
            _, self.capThreshold = cv2.threshold(self.capThreshold, self.threshold, 255, cv2.THRESH_BINARY_INV)
            self.contours, _ = cv2.findContours(self.capThreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in self.contours:
                (x, y, w, h) = cv2.boundingRect(cnt)
                cv2.rectangle(self.capThreshold, (x, y), (x + w, y + h), (255, 0, 0), 1)
                #middle point x = x + int(w/2) y = y + int(h/2) 
                cv2.circle(self.cutEyes, (x+int(w/2),y+int(h/2)) ,5, (255,0,0),-1)
                self.eyeCenter = [x+int(w/2),y+int(h/2)]
                
                break
            if self.thresholdIs == True:
                cv2.imshow("2",self.capThreshold)
            if self.rgbIs == True:
                cv2.imshow("c", self.cutEyes)

    def eyeLines(self):
        horizontalLineLeft = (self.landmarks.part(36).x, self.landmarks.part(36).y)
        horizontalLineRight = (self.landmarks.part(39).x, self.landmarks.part(39).y)

        verticalLineTop = (self.landmarks.part(38).x, self.landmarks.part(38).y)
        verticalLineBottom = (self.landmarks.part(40).x, self.landmarks.part(40).y)

        cv2.line(self.frame, horizontalLineLeft, horizontalLineRight,(0,255,0),1)
        cv2.line(self.frame, verticalLineTop, verticalLineBottom,(0,255,0),1)
    
    def getCutEyeShape(self,x,y,x1,y1):
        return self.frame[y:y1, x:x1]
    
    def optimizationEyesLooking(self):

        t = time.localtime()
        start_time = time.strftime("%S", t)        
        background = np.zeros((screen.height,screen.width),np.uint8)
        cv2.namedWindow("aa", cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow("aa", screen.x - 1, screen.y - 1)
        cv2.setWindowProperty("aa", cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        if time.perf_counter() < 3:
            if self.eyeCenter != None:
                self.centerArray = np.append(self.centerArray, [self.eyeCenter],axis=0)
            cv2.circle(background, (screen.width//2,screen.height//2), 5, (255,255,255), -1)
            (text_width, text_height) = cv2.getTextSize("Follow point", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)[0]
            cv2.putText(background, "Follow point", ((screen.width//2)-(text_width//2),(screen.height//2)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1 , cv2.LINE_AA)
            cv2.imshow("aa",background)

        elif time.perf_counter() < 6:
            if self.eyeCenter != None:
                self.bottomArray = np.append(self.bottomArray, [self.eyeCenter],axis=0)

            cv2.circle(background, (screen.width//2,screen.height), 5, (255,255,255), -1)
            (text_width, text_height) = cv2.getTextSize("Follow point", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)[0]
            cv2.putText(background, "Follow point", ((screen.width//2)-(text_width//2),(screen.height//2)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1 , cv2.LINE_AA)
            cv2.imshow("aa",background)
            
        elif time.perf_counter() < 9:
            if self.eyeCenter != None:
                self.rightArray = np.append(self.rightArray, [self.eyeCenter],axis=0)
            cv2.circle(background, (screen.width,screen.height//2), 5, (255,255,255), -1)
            (text_width, text_height) = cv2.getTextSize("Follow point", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)[0]
            cv2.putText(background, "Follow point", ((screen.width//2)-(text_width//2),(screen.height//2)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1 , cv2.LINE_AA)
            cv2.imshow("aa",background)
            
        elif time.perf_counter() < 12:
            if self.eyeCenter != None:
                self.leftArray = np.append(self.leftArray, [self.eyeCenter],axis=0)
            cv2.circle(background, (0,screen.height//2), 5, (255,255,255), -1)
            (text_width, text_height) = cv2.getTextSize("Follow point", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)[0]
            cv2.putText(background, "Follow point", ((screen.width//2)-(text_width//2),(screen.height//2)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1 , cv2.LINE_AA)
            cv2.imshow("aa",background)
            
        elif time.perf_counter() < 25:
            if self.eyeCenter != None:
                self.topArray = np.append(self.topArray, [self.eyeCenter],axis=0)
            cv2.circle(background, (screen.width//2,0), 5, (255,255,255), -1)
            (text_width, text_height) = cv2.getTextSize("Follow point", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)[0]
            cv2.putText(background, "Follow point", ((screen.width//2)-(text_width//2),(screen.height//2)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1 , cv2.LINE_AA)
            cv2.imshow("aa",background)
        else:
            cv2.destroyAllWindows()
            self.calibrationIsOkey = True
            self.check = True
            #break
        #cv2.imshow("aa",background)

        
    def getCameraShape(self):     
        for i in range(3):
            print(self.frame.shape[i])
        return self.frame[1], self.frame[0]

if __name__ == "__main__":
    import screeninfo
    screen = screeninfo.get_monitors()[0]
    ct = Controller()
    ct.main()