import cv2

class VideoIO():
    def __init__(self, win_name):
        self.cap = None
        self.out = None
        self.pause = False
        self.frame_count = 0
        self.win_name = win_name
        # cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)

    def isCap(self):
        if self.cap == None:
                raise Exception('Please Initialize video capture first')
        return True

    def isOut(self):
        if self.out == None:
                raise Exception('Please Initialize video writer first')
        return True

    def initVideoCapture(self, video_path):
        self.cap = cv2.VideoCapture(video_path)

    def initVideoWriter(self, width, height, target_video_path='test.mp4', fps=29.97):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(target_video_path, fourcc, fps, (width,height))

    def saveVideo(self, frame):
        if self.isOut():
            self.out.write(frame)

    def getMergedFrame(self, frame1, frame2):
        return cv2.resize(cv2.hconcat(frame1,frame2), self.getFrameSize, interpolation=cv2.INTER_CUBIC)

    def pauseFrame(self):
        self.pause = not self.pause

    def mergeFrame(self, bicubic, preds):
        if bicubic.ndim != preds.ndim:
            raise Exception('frame1 & frame2 is not same size')
        w, _ = self.getFrameSize()
        h,w = bicubic.shape[:2]
        return cv2.hconcat([bicubic[:, w // 4:3 * w // 4, :], preds[:, w // 4:3 * w // 4, :]])

    def showFrame(self, frame):
        cv2.imshow(self.win_name, frame)
        cv2.waitKey(1)

    def getFrame(self):
        if self.isCap():
            if self.pause:
                self.setFrame(self.frame_count)
            else:
                self.frame_count += 1
            ret, frame = self.cap.read()
        return ret, frame

    def setFrame(self, idx):
        if self.isCap():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES,idx)

    def resetFrame(self):
        if self.isCap():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def getFPS(self):
        if self.isCap():
            fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps

    def getFrameSize(self): 
        if self.isCap():
            w, h = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
        return w, h

    def getTotalFrame(self):
        if self.isCap():
            total_frame = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        return int(total_frame)
    
    def closeVideoCapture(self):
        if self.isCap():
            self.cap.release()

    def closeVideoWriter(self):
        if self.isOut():
            self.out.release()

    #TODO cv2.destroyAllWindows()