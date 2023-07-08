import cv2 as cv 
import numpy as np
from tts import *
from playsound import playsound
# Distance constants 
KNOWN_DISTANCE = 45
PERSON_WIDTH = 16
MOBILE_WIDTH = 3.0
CHAIR_WIDTH = 20.0

text1 = ""
text2 = ""

# Object detector constant 
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# colors for object detected
COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN =(0,255,0)
BLACK =(0,0,0)
# defining fonts 
FONTS = cv.FONT_HERSHEY_COMPLEX

# getting class names from classes.txt file 
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
#  setttng up opencv net
yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# object detector funciton /method
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # creating empty list to add objects data
    data_list =[]
    for (classid, score, box) in zip(classes, scores, boxes):
        x1,y1,x2,y2 = box
        center_x, center_y =  ( x1 + x2 ) / 2, ( y1 + y2 ) / 2
        height, width, channels = image.shape
        # print(x1,y1,x2,y2)
        # define color of each, object based on its class id 

        #detecting the centroid 
        #centroid = (center_x,center_y)
        #Measuring distance between two objects
       # for(classid2,score2,box2) in zip(classes,scores,boxes):
          #  if classid==classid2:
              #  continue
            #x12,y12,x22,y22 = box2
            #center_x2,center_y2 = (x12+x22)/2, (y12+y22)/2
            #distance = distance_finder(focal_person,PERSON_WIDTH,abs(center_x-center_x2))

         #define color of each object based on its class id   
        if center_x <= width/3:
            W_pos = "left"
        elif center_x <= (width/3 * 2):
            W_pos = "center"
        else:
            W_pos = "right"
        
        if center_y <= height/3:
            H_pos = "top"
        elif center_y <= (height/3 * 2):
            H_pos = "mid"
        else:
            H_pos = "bottom"

        text1 = W_pos
        text2 = H_pos
        color= COLORS[int(classid) % len(COLORS)]

    
        label = "%s : %f" % (class_names[classid], score)

        # draw rectangle on and label on object
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1]-14), FONTS, 0.5, color, 2)

         # getting the data 
        # 1: class name  
        # 2: object width in pixels, 
        # 3: position where have to draw text(distance)

        print("objects identified status")
        print("person identified : ",classid == 0)
        print("mobile identified : ",classid == 67)
        print("chair identified : ",classid == 56)

        # getting the data 
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        if classid == 0: # person class id 
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2),text1,text2])
        elif classid == 67:
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2),text1,text2])
        elif classid == 56:
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2),text1,text2])
        
        # if you want inclulde more classes then you have to simply add more [elif] statements here
        # returning list containing the object data. 
    return data_list

def focal_length_finder (measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length

# distance finder function 
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance_inches = (real_object_width * focal_length) / width_in_frmae
    distance_feet = distance_inches/12
    return distance_feet

# reading the reference image from dir 
ref_person = cv.imread('ReferenceImages/image14.png')
ref_mobile = cv.imread('ReferenceImages/image4.png')
ref_chair = cv.imread('ReferenceImages/image22.png')

mobile_data = object_detector(ref_mobile)
mobile_width_in_rf = mobile_data[1][1]

person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]

chair_data = object_detector(ref_person)
chair_width_in_rf = chair_data[0][1]

# print(f"Person width in pixels : {person_width_in_rf} mobile width in pixel: {mobile_width_in_rf}")

# finding focal length 
focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)
focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)
focal_chair = focal_length_finder(KNOWN_DISTANCE, CHAIR_WIDTH, chair_width_in_rf)

#d[]

def get_frame_output(frame, frame_cnt):
    output_text_file = open('output_text.txt','w') 
    data = object_detector(frame)
    for d in data:
        if d[0] =='person':
            distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
            x, y = d[2]
        elif d[0] =='cell phone':
            distance = distance_finder (focal_mobile, MOBILE_WIDTH, d[1])
            x, y = d[2]
        elif d[0] == 'chair':
            distance = distance_finder (focal_chair, CHAIR_WIDTH, d[1])
            x, y = d[2]
        
        text1,text2=d[3],d[4]

        cv.rectangle(frame, (x, y-3), (x+150, y+23),BLACK,-1 )
        cv.putText(frame, f'Dis: {round(distance,2)} feetcd ', (x+5,y+13), FONTS, 0.48, GREEN, 2)
        
        OUTPUTtEXT=""

        if distance > 100:
            OUTPUTtEXT = "Get closer"
            playsound('go closer.mp3')

        elif (round(distance) > 50) and (text2 == "mid"):
            OUTPUTtEXT="Go straight"
            playsound('go straight.mp3')
        else:
            OUTPUTtEXT = (str(d[0]) + " " + str(int(round(distance,1))) +" feet"+" take left or right")
            playsound('turn right.mp3')
                
        output_text_file.write(OUTPUTtEXT)

        output_text_file.write("\n")

        distance_feet = distance/12
        if distance_feet <=5:
            playsound('alert_voice.mp3')
    
    output_text_file.close()
    
    return frame

    