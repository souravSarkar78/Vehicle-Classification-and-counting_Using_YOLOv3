import cv2
import csv
import numpy as np
from tracker import *
tracker = EuclideanDistTracker()

# Initialize the videocapture object
cap = cv2.VideoCapture('video4.mp4')
whT = 320

confThreshold =0.1
nmsThreshold= 0.2

font_color = (0, 255, 200)
font_size = 0.5
font_thickness = 2

# Middle cross line position
middle_line_position = 225   
up_line_position = middle_line_position - 15
down_line_position = middle_line_position + 15


## Coco Names
classesFile = "coco.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)


## Model Files
modelConfiguration = 'yolov3-320.cfg'
modelWeights = 'yolov3-320.weights'

# configure the network model
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

# Configure the network backend

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# class index for our required detection classes
required_class_index = [2, 3, 5, 7]

# Define random colour for each class
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')


# Function for finding the center of a rectangle
def find_center(x, y, w, h):
    x1=int(w/2)
    y1=int(h/2)
    cx = x+x1
    cy=y+y1
    return cx, cy
    
temp_up_list = []
temp_down_list = []
up_list = [0, 0, 0, 0]
down_list = [0, 0, 0, 0]

# Function for count vehicle
def count_vehicle(box_id):

    x, y, w, h, id, index = box_id

    # Find the center of the rectangle for detection
    center = find_center(x, y, w, h)
    ix, iy = center
    
    # Find the current position of the vehicle
    if (iy > up_line_position) and (iy < middle_line_position):

        if id not in temp_up_list:
            temp_up_list.append(id)

    elif iy < down_line_position and iy > middle_line_position:

        if id not in temp_down_list:
            temp_down_list.append(id)
            
    elif iy < up_line_position:

        if id in temp_down_list:
            temp_down_list.remove(id)
            up_list[index] = up_list[index]+1

    elif iy > down_line_position:
        if id in temp_up_list:
            temp_up_list.remove(id)
            down_list[index] = down_list[index] + 1

    # Draw circle in the middle of the rectangle
    cv2.circle(img, center, 2, (0, 0, 255), -1)  # end here
    print(up_list, down_list)

# Function for finding the detected objects from the network output
def findObjects(outputs,img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    detection = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId in required_class_index:
                if confidence > confThreshold:
                    # print(classId)
                    w,h = int(det[2]*wT) , int(det[3]*hT)
                    x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                    bbox.append([x,y,w,h])
                    classIds.append(classId)
                    confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    # print(classIds)
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # print(x,y,w,h)

        
        # print(d)
        color = [int(c) for c in colors[classIds[i]]]
        name = classNames[classIds[i]]
        cv2.putText(img,f'{name.upper()} {int(confs[i]*100)}%',
                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        detection.append([x, y, w, h, required_class_index.index(classIds[i])])

    boxes_ids = tracker.update(detection)
    # print(boxes_ids)
    for box_id in boxes_ids:
        count_vehicle(box_id)


while True:
    success, img = cap.read()
    img = cv2.resize(img,(0,0),None,0.5,0.5)
    ih, iw, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)

    # Set the input of the network
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
    # Feed data to the network
    outputs = net.forward(outputNames)
   
    findObjects(outputs,img)

    # Draw the crossing lines

    cv2.line(img, (0, middle_line_position), (iw, middle_line_position), (255, 0, 255), 1)
    cv2.line(img, (0, up_line_position), (iw, up_line_position), (0, 0, 255), 1)
    cv2.line(img, (0, down_line_position), (iw, down_line_position), (0, 0, 255), 1)

    # Draw counting texts in the frame
    cv2.putText(img, "Up", (110, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    cv2.putText(img, "Down", (160, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    cv2.putText(img, "Car:        "+str(up_list[0])+"     "+ str(down_list[0]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    cv2.putText(img, "Motorbike:  "+str(up_list[1])+"     "+ str(down_list[1]), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    cv2.putText(img, "Bus:        "+str(up_list[2])+"     "+ str(down_list[2]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    cv2.putText(img, "Truck:      "+str(up_list[3])+"     "+ str(down_list[3]), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    cv2.imshow('Image', img)

    if cv2.waitKey(1) == ord('q'):
        break

# Write the vehicle counting information in a file and save it

with open("data.csv", 'w') as f1:
    cwriter = csv.writer(f1)
    cwriter.writerow(['Direction', 'car', 'motorbike', 'bus', 'truck'])
    up_list.insert(0, "Up")
    down_list.insert(0, "Down")
    cwriter.writerow(up_list)
    cwriter.writerow(down_list)
f1.close()
# Finally realese the capture object and destroy all active windows
cap.release()
cv2.destroyAllWindows()