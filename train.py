from facenet_pytorch import InceptionResnetV1
from mtcnn.detector import MtcnnDetector
import cv2
from align_faces import warp_and_crop_face, get_reference_facial_points
import numpy as np
from numpy.linalg import norm
import time
import datetime
from sklearn import preprocessing

from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
le = preprocessing.LabelEncoder()
detector = MtcnnDetector()
def face_detection(img, output_size, align = True):
    """Nếu align == True thì sẽ có thêm bước Face Alignment sau khi nhận diện gương mặt,
        ngược lại thì không có thêm bước này"""
    
    # Nhận diện gương mặt
    boxes, facial5points = detector.detect_faces(img)
    
    # Tiến hành Alignment
    if align == True:
        if len(facial5points) != 0:
            facial5points = np.reshape(facial5points[0], (2, 5))

            default_square = True
            inner_padding_factor = 0.25
            outer_padding = (0, 0)

            ## Yêu cầu 10:
            ## VIẾT CODE Ở ĐÂY:

            # sử dụng hàm get_reference_facial_points
            reference_5pts = get_reference_facial_points(
                output_size, inner_padding_factor, outer_padding, default_square)
            
            # sử dụng hàm warp_and_crop_face
            face = warp_and_crop_face(img, facial5points, reference_pts = reference_5pts, crop_size = output_size)

    # Không Alignment
    else:
        (h,w) = img.shape[:2]
        if len(boxes) != 0:
            for box in boxes:
                (startX, startY, endX, endY) = box[:4].astype('int')
                (startX, startY) = (max(0, startX),max(0, startY))
                (endX, endY) = (min(w-1, endX), min(h-1, endY))
                face = img[startY:endY, startX:endX]
                face = cv2.resize(face, output_size)
    return face

import glob
import torch 
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
import os
from PIL import Image
import numpy as np

IMG_PATH = "../Data/Image"
DATA_PATH = './data'
embeddings = []
names = []

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def trans(img):
    transform = transforms.Compose([
            transforms.ToTensor(),
            # fixed_image_standardization
        ])
    return transform(img).unsqueeze(0)
def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor

model = InceptionResnetV1(
    classify=False,
    pretrained="vggface2"
).to(device)
model.eval()
def face_detect(image):
    
    img = image.copy()
    (h,w) = img.shape[:2]
    boxes, facial5points = detector.detect_faces(img)
    for box in boxes:
        (startX,startY,endX,endY)=box[:4].astype('int')

        #ensure the bounding boxes fall within the dimensions of the frame
        (startX,startY)=(max(0,startX),max(0,startY))
        (endX,endY)=(min(w-1,endX), min(h-1,endY))

        #extract the face ROI, convert it from BGR to RGB channel, resize it to 224,224 and preprocess it
        face=img[startY:endY, startX:endX]
        face=cv2.resize(face,(160,160))

        # (mask,withoutMask) = model.predict(face.reshape(1,224,224,3))[0]

        # #determine the class label and color we will use to draw the bounding box and text
        # label='Mask' if mask>withoutMask else 'No Mask'
        # color=(0,255,0) if label=='Mask' else (0,0,255)
        color=(0,255,0)
        #include the probability in the label
        # label="{}: {:.2f}%".format(label,max(mask,withoutMask)*100)
        label = "Truyen"
        #display the label and bounding boxes
        cv2.putText(img,label,(startX,startY-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)
        cv2.rectangle(img,(startX,startY),(endX,endY),color,2)
        
    return face
for usr in os.listdir(IMG_PATH):
    for file in glob.glob(os.path.join(IMG_PATH, usr)+'/*.jpg'):
        try:
            # img = Image.open(file)
            img = cv2.imread(file)
            print(img.shape)
            face = face_detect(img)
            cv2.imshow("face", face)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            continue
        with torch.no_grad():
            embed = model(trans(face).to(device))
            embeddings.append(embed.numpy()) #1 anh, kich thuoc [1,512]
            names.append(usr)
    # if len(embeds) == 0:
    #     continue
    # embedding = torch.cat(embeds).mean(0, keepdim=True) #dua ra trung binh cua 50 anh, kich thuoc [1,512]
    # embeddings.append(embedding.numpy()) # 1 cai list n cai [1,512]
    
print(names)
le.fit_transform(names)
print(len(embeddings))

label = le.fit_transform(names)
data = np.asarray(embeddings).reshape(len(embeddings),512)
print("Shape data")
print(type(data))
print(data.shape)
print(label.shape)
print(label)
clf = make_pipeline(StandardScaler(), SVC(gamma='auto',probability = True))
clf.fit(data,label)
x = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
import pickle
file_model = "Model/"+x+"_SVM.pkl"
file_user =  "List_user/"+x+"_ListUser.pkl"
file_embed = "Embeded/EmbedFace/" + x + "_listFace_embeded.pkl"
file_list_name = "Embeded/EmbededList/" +x  +"_list_Name_embeded.pkl"
print(names)
print(le.classes_)
pickle.dump(clf,open(file_model,"wb"))
pickle.dump(le.classes_,open(file_user,"wb"))
pickle.dump(data,open(file_embed,"wb"))
pickle.dump(names,open(file_list_name,"wb"))