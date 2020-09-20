import numpy as np
import cv2
import tensorflow as tf

# load the model
m_new = tf.keras.models.load_model('digit_Recognition.h5')

img = np.ones([400,400],dtype ='uint8')*255
img[50:350,50:350]=0

wname = 'Canvas'
cv2.namedWindow(wname)
state = False

def shape(event,x,y,flags,param):

    global state
    if event == cv2.EVENT_LBUTTONDOWN:
        state = True
        cv2.circle(img,(x,y),10,(255,255,255),-1)


    elif event == cv2.EVENT_MOUSEMOVE:
        if state == True:
            cv2.circle(img,(x,y),10,(255,255,255),-1)
    else:
        state = False

cv2.setMouseCallback(wname,shape)  # Shape is a sub method called in the setMouseCallback method

while True:
    cv2.imshow(wname,img)
    key = cv2.waitKey(1)
    if key == ord('q'):# q for Quit
        break
    elif key == ord('c'):# c for clearing the screen
        img[50:350,50:350]=0
    elif key == ord('w'):# w for saving the image drawn
        out = img[50:350,50:350]
        cv2.imwrite('Output.jpg',out)
    elif key == ord('p'):# p to predict the digit
        img_test = img[50:350,50:350]
        img_resize = cv2.resize(img_test,(28,28)).reshape(1,28,28)
        digit = m_new.predict_classes(img_resize)
        print("The digit recognised is:",digit) 
cv2.destroyAllWindows()
