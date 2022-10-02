from re import X
import threading
import cv2
from facenet_pytorch import InceptionResnetV1
# import tensorflow as tf
import time 
import torch
# import tensorflow as tf
# from tensorflow.keras.models import load_model
import pickle
import cv2
import numpy as np
import cv2
from torchvision import transforms
from align_faces import warp_and_crop_face, get_reference_facial_points
from mtcnn.detector import MtcnnDetector
detector = MtcnnDetector()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def trans(img):
    transform = transforms.Compose([
            transforms.ToTensor(),
            fixed_image_standardization
        ])
    return transform(img).unsqueeze(0)
def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor

model_resnet = InceptionResnetV1(
    classify=False,
    pretrained="casia-webface"
).to(device)
model_resnet.eval()

def embed_face(img):
    img= cv2.resize(img,(160,160))
    with torch.no_grad():
        embed = model_resnet(trans(img).to(device))
    return embed

model = pickle.load(open("Model/2022_10_01_13-57-05_SVM.pkl","rb"))
names = pickle.load(open("List_user/2022_10_01_11-32-55_ListUser.pkl","rb"))

probability = 0.87

def indentify(lst_face):
    x = time.time()
    out = 1-model.predict_proba(np.asarray(lst_face))
    lst_out = []
    lst_prob = []
    for x in out:
        if(np.max(x)>=probability):
            lst_out.append(np.argmax(out))
            lst_prob.append(np.max(x))
        else:
            lst_out.append(-1)
            lst_prob.append(np.max(x))
    print(lst_prob)
    print(lst_out)
    print(out)
    
    return lst_out,lst_prob


def predict_face(embed_img):
    out = 1 - model.predict_proba(embed_img.reshape(1,-1))
    proba = np.max(out)
    print(out)
    id = np.argmax(out)
    if(proba<probability):
        return "unknow", proba
    else:
        return names[id], proba




def mask_detect(image):
    img = image.copy()
    (h,w) = img.shape[:2]
    boxes, facial5points = detector.detect_faces(img)
    lst_embeddings = []
    lst_max = []
    lst_out = []
    if(len(boxes)!=0):

        for box in boxes:
            (startX,startY,endX,endY)=box[:4].astype('int')
            #ensure the bounding boxes fall within the dimensions of the frame
            (startX,startY)=(max(0,startX),max(0,startY))
            (endX,endY)=(min(w-1,endX), min(h-1,endY))

            #extract the face ROI, convert it from BGR to RGB channel, resize it to 224,224 and preprocess it
            face=img[startY:endY, startX:endX]
            embed = embed_face(face).numpy().reshape(512)
            name, proba = predict_face(embed)

            if(name == "unknow"):
                color = (255,0,0)

                label="{}: {:.2f}%".format(name,proba*100)
                #display the label and bounding boxes
                cv2.putText(img,label,(startX,startY-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)
                cv2.rectangle(img,(startX,startY),(endX,endY),color,2)
            else:
                color = (0,255,0)

                label="{}: {:.2f}%".format(name,proba*100)
                #display the label and bounding boxes
                cv2.putText(img,label,(startX,startY-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)
                cv2.rectangle(img,(startX,startY),(endX,endY),color,2)
    return img

global a
a = 10

def PlayCamera(id):    
    video_capture = cv2.VideoCapture(id)
    while True:
        x = time.time()
        ret, frame = video_capture.read()
        # img = frame[0:128,0:128]
        # print(model.predict(np.array([img])))
        img = mask_detect(frame)
        print(time.time() - x)
        cv2.imshow('{}'.format(id), img)        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()

cameraIDs = [0]
threads = []
for id in cameraIDs:
    threads += [threading.Thread(target=PlayCamera, args=(id,))]
for t in threads:    
    t.start()
for t in threads: 
    t.join()