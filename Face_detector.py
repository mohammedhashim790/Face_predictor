#!/usr/bin/env python
# coding: utf-8

# In[10]:


import cv2
import matplotlib.pyplot as plt
import dlib
import imutils as im
import time


# In[11]:


def rect_to(det,frame):
    rects = det(frame,1)
    for rect in rects:
        (x,y,w,h) = rect.left()-2,rect.top()-5,(rect.right()-rect.left())+5,rect.bottom()-rect.top()+10
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
    return frame


# In[13]:


#loading the dataset
print("{INFO}...Loading the face detector model and VideoStream")
time.sleep(5)
det = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture(0)
print("Done")


# In[14]:


#loading the camera stream
is_open,read = cap.read()
if not is_open:
    print("Error...! Unable to load Camera Device")
    exit()


# In[15]:


while True:
    try:
        is_open,read = cap.read()
        read = im.resize(read,width = 400)
        read = rect_to(det,read)
        cv2.imshow('Frames',read)
    except Exception as e:
        print("Error : {}".format(e))
        break
    k = cv2.waitKey(1)
    if k == 27:
        cap.release()
        print("Exiting...")
        break
        
cv2.destroyAllWindows()


# In[ ]:




