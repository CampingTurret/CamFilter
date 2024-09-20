import time
import cv2 as cv
from multiprocessing import Process, Queue, Value
import numpy as np
import pandas as pd
import ultralytics as ul
import ultralytics.engine.results as res
import keyboard
import torchvision 
import torch


class Frame():

    draw_eyes = False
    draw_center = True
    set_Leyes = False
    set_Reyes = False
    set_center = False


    def __init__(self,a):
        self.image = a
        self.G_image = cv.cvtColor(a,cv.COLOR_BGR2GRAY)
        self.create_time = time.time()

    def Set_Leye(self,Lx :int,Ly:int,Lh:int,Lw:int):
        self.Leye = (Lx,Ly,Lh,Lw)
        self.set_Leyes = True

    def Set_Reye(self,Rx :int,Ry:int,Rh:int,Rw:int):
        self.Reye = (Rx,Ry,Rh,Rw)
        self.set_Reyes = True
    

    def Set_center(self,x,y,h,w):
        self.center = (x,y,h,w)
        self.set_center = True

    def Set_Rotation(self,a1,a2,a3):
        self.rotation = (a1,a2,a3)


        
def Display_Frame(name,Frame):
    cv.imshow(name,Frame)

def Pass(a:Frame) -> Frame:
    return a

def Draw_contours(a :Frame,threshold : int = 150):
    """
    threshold should be between 0-255
    """
    I = a.image
    I_gray = a.G_image
    ret,thresh = cv.threshold(I_gray,threshold,255,cv.THRESH_BINARY)
    contours, hiercachy = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    I_copy = I.copy()
    cv.drawContours(I_copy,contours,-1,(0,255,0),2,cv.LINE_AA)
    a.image = I_copy
    return a



class Face_tracking():
    Active = False

    def __init__(self,Q1,Q2):
        self.face_cascade = cv.CascadeClassifier()
        self.eyes_cascade = cv.CascadeClassifier()
        self.face_cascade.load(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eyes_cascade.load(cv.data.haarcascades + "haarcascade_eye.xml")
        self.Q1 = Q1
        self.Q2 = Q2


    def Track_Face(self,a : Frame):
        I = a.image
        I_G = a.G_image


        face = self.face_cascade.detectMultiScale(I_G, minNeighbors = 10, minSize = (40,40))
        a = Drawstep(a, face)
        

        return a

    def Start_Tracking(self):
        self.Active = True
        while(True):

            while(self.Active):
                if keyboard.is_pressed('l') :
                    self.Active = False
                print("Cascade")
                frame = self.Q1.get()
                if frame.create_time > time.time()-0.1:
                    self.Q2.put(self.Track_Face(frame))
                if keyboard.is_pressed('l') :
                    self.Active = False
            if keyboard.is_pressed('k'):
                self.Active = True


class YOLO():
    Active = False

    def __init__(self,Q1,Q2):
        self.model = ul.YOLO('yolov8n.pt')
        self.Q1 = Q1
        self.Q2 = Q2

    def Track_Face(self,a : Frame):
        I = a.image

        results: list[res.Results] = self.model.predict(I, verbose= False)
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for i, box in enumerate(boxes):
                class_id = int(box.data[0][-1])
                r = box.xyxy[0].astype(int)
                cv.rectangle(I, r[:2], r[2:], (255, 255, 255), 2)
                cv.putText(I,result.names[class_id], r[:2] , cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),thickness=1)
                
        a.image = I
        return a

    def Start_Tracking(self):
        self.Active = False
        while(True):
            while(self.Active):
                if keyboard.is_pressed('p'):
                    self.Active = False
                    continue
                frame = self.Q1.get()
                print("YOLO")
                if frame.create_time > time.time()-0.1:
                    self.Q2.put(self.Track_Face(frame))
            if keyboard.is_pressed('o'):
                self.Active =  True

class FRCNN():
    Active = False

    def __init__(self,Q1,Q2):
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT') 
        self.model.eval()
        self.Q1 = Q1
        self.Q2 = Q2

    def Track_Face(self,a : Frame):
        I = a.image
        img = I
        	
        COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        img = transform(img)
        pred = self.model([img])
        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
        pred_score = list(pred[0]['scores'].detach().numpy())
        pred_l = [pred_score.index(x) for x in pred_score if x>0.5]
        if len(pred_l) == 0:
            return a
        pred_t = pred_l[-1]
        results = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]
        

        for i in range(len(results)): 
            cv.rectangle(I, (int(results[i][0][0]),int(results[i][0][1])), (int(results[i][1][0]),int(results[i][1][1])), (0, 255, 0), thickness=2) 
            cv.putText(I,pred_class[i], (int(results[i][0][0]),int(results[i][0][1])), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),thickness=1) 

        a.image = I
        return a

    def Start_Tracking(self):
        self.Active = False
        while(True):
            while(self.Active):
            
                frame = self.Q1.get()
                print('CRNN')
                if frame.create_time > time.time()-0.1:
                    self.Q2.put(self.Track_Face(frame))
                if keyboard.is_pressed('m'):
                    self.Active = False
            if keyboard.is_pressed('n'):
                self.Active =  True



def Drawstep(a:Frame, face):

    I = a.image
    for f in face:
        cv.rectangle(I, (f[0], f[1]), (f[0] + f[2], f[1] + f[3]), (0, 255, 0), 4)
    if(a.draw_eyes):
       print('test')
    
       
        
    a.image = I
    return a


class object():

    center : list
    rotation : list
    eyes : list
    number : int

    def __init__(self):
        self.number = int(10*np.random())

    def Load_model(self,a):
        self.model =a 


class display:

    Active : bool
    framequeue : Queue

    def __init__(self,name,framequeue : Queue):
        self.framequeue = framequeue
        self.name = name
       

    def Start_show(self):

        self.Active = True
        while(self.Active):
            frame = self.framequeue.get()
            if frame is None:
                continue
            cv.imshow(self.name,frame.image)
            if cv.waitKey(1) & 0xFF  == ord('q'):
                self.Active = False

    def Stop_show(self):
        self.Active = False




class webcam:

    Active : bool
    framequeue : Queue
    def __init__(self,CameraID,framequeue : Queue):
        self.framequeue = framequeue
        self.Cam = cv.VideoCapture(CameraID)
       

    def Start_capture(self):

        self.Active = True
        while(self.Active):
            ret,image = self.Cam.read()
            frame = Frame(image)
            self.framequeue.put(frame)

    def Stop_capture(self):
        self.Active = False


