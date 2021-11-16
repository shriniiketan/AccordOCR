import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2


df = pd.read_csv('labels.csv')
df.head()

import xml.etree.ElementTree as xet

filename = df['filepath'][0]


def getFilename(filename):
    filename_image = xet.parse(filename).getroot().find('filename').text
    filepath_image = os.path.join('./images',filename_image)
    return filepath_image

image_path = list(df['filepath'].apply(getFilename))

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array

labels = df.iloc[:,1:].values

data = []
output = []
for ind in range(len(image_path)):
    image = image_path[ind]
    img_arr = cv2.imread(image)
    h,w,d = img_arr.shape
   
    
    load_image = load_img(image,target_size=(224,224))
    load_image_arr = img_to_array(load_image)
    norm_load_image_arr = load_image_arr/255.0 
    
    
    xmin,xmax,ymin,ymax = labels[ind]
    nxmin,nxmax = xmin/w,xmax/w
    nymin,nymax = ymin/h,ymax/h
    label_norm = (nxmin,nxmax,nymin,nymax) 
    
    
    data.append(norm_load_image_arr)
    output.append(label_norm)
    
    
X = np.array(data,dtype=np.float32)
y = np.array(output,dtype=np.float32)

x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=0)
x_train.shape,x_test.shape,y_train.shape,y_test.shape


from tensorflow.keras.applications import MobileNetV2, InceptionV3, InceptionResNetV2
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model
import tensorflow as tf


inception_resnet = InceptionResNetV2(weights="imagenet",include_top=False,
                                     input_tensor=Input(shape=(224,224,3)))
inception_resnet.trainable=False

headmodel = inception_resnet.output
headmodel = Flatten()(headmodel)
headmodel = Dense(500,activation="relu")(headmodel)
headmodel = Dense(250,activation="relu")(headmodel)
headmodel = Dense(4,activation='sigmoid')(headmodel)

model = Model(inputs=inception_resnet.input,outputs=headmodel)

model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
model.summary()

history = model.fit(x=x_train,y=y_train,batch_size=10,epochs=100,
                    validation_data=(x_test,y_test))

model.save('./models/object_detection.h5')

model = tf.keras.models.load_model('./models/object_detection.h5')
print('model loaded sucessfully')

path = 'C:/SelfDrivingMaterials/OCR Accord/IndianCars/licensed_car0.jpeg'
image = load_img(path) 
image = np.array(image,dtype=np.uint8) 
image1 = load_img(path,target_size=(224,224))
image_arr_224 = img_to_array(image1)/255.0  


h,w,d = image.shape
print('Height of the image =',h)
print('Width of the image =',w)

plt.figure(figsize=(10,8))
plt.imshow(image)
plt.show()

image_arr_224.shape

test_arr = image_arr_224.reshape(1,224,224,3)
test_arr.shape

#predic
coords = model.predict(test_arr)
print(coords)

#denormalize 
denorm = np.array([w,w,h,h])
coords = coords * denorm
print(coords)

coords = coords.astype(np.int32)
print(coords)


xmin, xmax,ymin,ymax = coords[0]
pt1 =(xmin-50,ymin-50)
pt2 =(xmax+50,ymax+50)
print(pt1, pt2)
cv2.rectangle(image,pt1,pt2,(0,255,0),3)

plt.figure(figsize=(10,8))
plt.imshow(image)
plt.show()




def object_detection(path):
    #read image
    image = load_img(path)
    image = np.array(image,dtype=np.uint8)#8bit array(0,255)
    image1 = load_img(path,target_size=(224,224))
    #data preprocessing
    image_arr_224 = img_to_array(image1)/255.0#convert into array and get the normalized output
    h,w,d = image.shape
    test_arr = image_arr_224.reshape(1,224,224,3)
    #make predictions
    coords = model.predict(test_arr)
    #denormalize
    denorm = np.array([w,w,h,h])
    coords = coords * denorm
    coords = coords.astype(np.int32)
    #draw bounding 
    xmin, xmax,ymin,ymax = coords[0]
    pt1 =(xmin,ymin)
    pt2 =(xmax,ymax)
    print(pt1, pt2)
    cv2.rectangle(image,pt1,pt2,(0,255,0),3)
    return image, coords



import pytesseract as pt

path = 'C:/SelfDrivingMaterials/OCR Accord/IndianCars/licensed_car78.jpeg'
image, cods = object_detection(path)

plt.imshow(image)
plt.show()

img = np.array(load_img(path))
xmin,xmax,ymin,ymax = cods[0]
print(cods)
if(ymin<50):
    ymin = 0
    
else:
    ymin = ymin - 50
    
if(xmin<50):
    ymin = 0
    
else:
    xmin = xmin - 50
roi = img[ymin:ymax+50,xmin:xmax+50]

plt.imshow(roi)
plt.show()


'''
text = pt.image_to_string(roi)
print(len(text))
'''


import easyocr

gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
reader = easyocr.Reader(['en'])
result = reader.readtext(gray)
print(result)

'''
n=0
i=0
m=3
newResult = []
maxbox = 0
maxDia = 0

for i in range(len(result)):
    (X_top_left,Y_top_left) = result[i][0][0] 
    (X_bottom_right,Y_bottom_right) = result[i][0][2]
    #newResult = newResult.append((X_top_left,Y_top_left),(X_bottom_right,Y_bottom_right),maxbox)
   
    sub1 = X_bottom_right - X_top_left
    sub2 = Y_bottom_right - Y_top_left
    pow1 = pow(sub1,2)
    pow2 = pow(sub2,2)
    tot = pow1+pow2
    sqrt = tot**0.5
   
    if (sqrt>maxDia):
        maxDia = sqrt 
        maxbox = i
       
#print("Plate no is - " + result[maxbox][1])   



num_plate = ''

for i in range(len(result)):

    num_plate = num_plate + ' ' + result[i][1]
    
print("Plate no is - " + num_plate) 



'''

correct = 0
non_correct = 0
total = 0

correct = correct +1

print(correct)

non_correct = non_correct+1

print(non_correct)

total = correct + non_correct

acc = correct / total * 100


print("Accuracy - " ,float(acc))

