import cv2
from mtcnn.detector import MtcnnDetector
import time
import datetime
detector = MtcnnDetector()
count = 0
def sim_detect(image, name):
    global count
    img = image.copy()
    (h,w) = img.shape[:2]
    boxes, facial5points = detector.detect_faces(img)
    if(len(boxes)!=0):
        for box in boxes:
            (startX,startY,endX,endY)=box[:4].astype('int')
            #ensure the bounding boxes fall within the dimensions of the frame
            (startX,startY)=(max(0,startX),max(0,startY))
            (endX,endY)=(min(w-1,endX), min(h-1,endY))

            #extract the face ROI, convert it from BGR to RGB channel, resize it to 224,224 and preprocess it

            color = (0,255,0)

            label=""
            #display the label and bounding boxes
            cv2.putText(img,label,(startX,startY-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)
            cv2.rectangle(img,(startX,startY),(endX,endY),color,1)
            count+=1
    return img
def PlayCamera(id, name = "test"):    
    video_capture = cv2.VideoCapture(id)

    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))
    video_cod = cv2.VideoWriter_fourcc(*'XVID')
    video_output= cv2.VideoWriter(name+'.avi',
                        video_cod,
                        30,
                        (frame_width,frame_height))
    
    while True:
    
        ret, frame = video_capture.read()
        img = sim_detect(frame,name)
        cv2.imshow('{}'.format(id), img) 
        video_output.write(img)   
        time.sleep(0.01)    
        if(count == 250):
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
PlayCamera(0,"truyen")