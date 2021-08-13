''' IMPORTING THE REQUIRED LIBRARIES '''

import numpy as np 
import cv2 
import matplotlib.image as mpimg
from skimage.feature import hog
import glob 

''' GETTING THE TRAINING DATASET '''

car = glob.glob('C:\SelfDrivingMaterials\Section6\data\car\**\*.png')
no_car = glob.glob('no car\**\*.png')


car_len = len(car) 
no_car_len = len(no_car)

print(car_len)
print(no_car_len)

''' HOG FEATURES FOR CAR IMAGES '''

car_hog_accum = []
blurr_kernel = np.ones((3,3)) *1/9

for i in car :
    image_color = mpimg.imread(i)
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    
    blurred_image = cv2.filter2D(image_gray,-1,blurr_kernel)
    
    car_hog_feature, car_hog_img = hog(blurred_image,
                                       orientations = 11,
                                       pixels_per_cell =(16,16),
                                       cells_per_block=(2,2),
                                       transform_sqrt=False,
                                       visualize=True,
                                       feature_vector=True)
   
    car_hog_accum.append(car_hog_feature)
    

X_car = np.vstack(car_hog_accum).astype(np.float64)
Y_car = np.ones(len(X_car))

print(X_car.shape) 
print(Y_car.shape)


''' HOG FEATURE FOR NON CAR IMAGES '''


nocar_hog_accum = []
blurr_kernel = np.ones((3,3)) *1/9


for i in no_car :
    image_color = mpimg.imread(i)
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    
    blurred_image = cv2.filter2D(image_gray,-1,blurr_kernel)
    
    nocar_hog_feature, nocar_hog_img = hog(blurred_image,
                                       orientations = 11,
                                       pixels_per_cell =(16,16),
                                       cells_per_block=(2,2),
                                       transform_sqrt=False,
                                       visualize=True,
                                       feature_vector=True)
   
    nocar_hog_accum.append(nocar_hog_feature)
    

X_nocar = np.vstack(nocar_hog_accum).astype(np.float64)
Y_nocar = np.zeros(len(X_nocar))

print(X_nocar.shape) 
print(Y_nocar.shape)

X = np.vstack((X_car,X_nocar))
Y = np.hstack((Y_car,Y_nocar))

print(X.shape)
print(Y.shape)


''' TRAINING THE SVM CLASSIFIER '''


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state = 101)
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report 

svc_model = LinearSVC()
svc_model.fit(X_train,Y_train)

Y_predict = svc_model.predict(X_test)
print(classification_report(Y_test, Y_predict))


''' OPTMISATION OF THE C AND GAMMA PARAMETERS FOR RBF KERNEL'''

param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)
grid.fit(X_train,Y_train)


grid_predictions = grid.predict(X_test)
print(classification_report(Y_test,grid_predictions))


