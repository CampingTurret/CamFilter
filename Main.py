from multiprocessing import Process, Queue, Value
from Functions import webcam, display, Pass, Draw_contours, Frame, Face_tracking, YOLO, FRCNN
from time import sleep

def Iinitalize_Queues(I:int):
    Queues = []
    for i in range(I):
        Queues.append(Queue())

    return Queues

def capture(Q):
    w = webcam(0,Q)
    w.Start_capture()

def Passer(Q1,Q2):
    while(True):
        Q2.put(Pass(Q1.get()))

def Contours(Q1,Q2,threshold :int = 150):
    while(True):
        Q2.put(Draw_contours(Q1.get(),threshold))

def cascade(Q1,Q2):
    f = Face_tracking(Q1,Q2)
    f.Start_Tracking()

def yolo(Q1, Q2):
    f = YOLO(Q1,Q2)
    f.Start_Tracking()

def rcnn(Q1, Q2):
    f = FRCNN(Q1,Q2)
    f.Start_Tracking()

    
def Output(Q):
    s = display("output",Q)
    s.Start_show()

if __name__ == '__main__':

     Q = Iinitalize_Queues(2)
     processes=[]
     p = Process(target=capture, args=(Q[0], ))
     p.start()
     processes.append(p)
     p = Process(target=Output, args=(Q[-1], ))
     p.start()
     processes.append(p)
     p = Process(target=yolo, args=(Q[0],Q[-1],)) # p-o
     p.start()
     processes.append(p)
     p = Process(target=cascade, args=(Q[0],Q[-1],)) # l-k
     p.start()
     processes.append(p)
     p = Process(target=rcnn, args=(Q[0],Q[-1],))  # m-n
     p.start()
     processes.append(p)

     


     for p in processes:
         p.join()


