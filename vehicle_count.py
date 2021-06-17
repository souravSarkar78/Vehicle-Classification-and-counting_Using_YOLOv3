import cv2
import numpy as np

from tracker import *
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture('video.mp4')
whT = 320
confThreshold =0.1
nmsThreshold= 0.2

car = 0
motorbike = 0
bus = 0
truck = 0

middle_line_position = 250
up_line_position = middle_line_position - 20
down_line_position = middle_line_position + 20

counter = 0


## Coco Names
classesFile = "coco.names"
f = open('coco.txt', 'r')
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

# Update the count for each object class
def update_count(name):
    global car, motorbike, bus, truck
    if name =='car':
        car+=1
    elif name == 'motorbike':
        motorbike+=1
    elif name == 'bus':
        bus +=1
    elif name == 'truck':
        truck +=1

up_list = []
down_list = []

up= 0
down = 0

# Function for count vehicle
def count_vehicle(box_id, name):
    global up, down

    x, y, w, h, id = box_id

    # Find the center of the rectangle for detection
    center = find_center(x, y, w, h)
    ix, iy = center
    
    if (iy > up_line_position) and (iy < middle_line_position):

        if id not in up_list:
            up_list.append(id)

    elif iy < down_line_position and iy > middle_line_position:

        if id not in down_list:
            down_list.append(id)
            
    elif iy < up_line_position:

        if id in down_list:
            down_list.remove(id)
            down += 1
            update_count(name)

    elif iy > down_line_position:
        if id in up_list:
            up_list.remove(id)
            up +=1
            update_count(name)

    print(up, down, car, motorbike, bus, truck)

    # Draw circle in the middle of the rectangle
    cv2.circle(img, center, 2, (0, 0, 255), -1)  # end here

# Function for finding the detected objects from the network output
def findObjects(outputs,img):
    global counter
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    cnt = 0
    detection = []
    for output in outputs:

        for det in output:
            cnt +=1
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

        detection.append([x, y, w, h])
        # print(d)
        
    boxes_ids = tracker.update(detection)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        color = [int(c) for c in colors[classIds[i]]]
        name = classNames[classIds[i]]
        cv2.putText(img,f'{name.upper()} {int(confs[i]*100)}%',
                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        count_vehicle(box_id, name)


while True:
    success, img = cap.read()
    img = cv2.resize(img,(0,0),None,0.5,0.5)
    ih, iw, c = img.shape
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)

    findObjects(outputs,img)

    cv2.line(img, (0, middle_line_position), (iw, middle_line_position), (255, 0, 255), 1)
    cv2.line(img, (0, up_line_position), (iw, up_line_position), (0, 0, 255), 1)
    cv2.line(img, (0, down_line_position), (iw, down_line_position), (0, 0, 255), 1)
    cv2.putText(img, "Car: "+str(car), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(img, "Motorbike: "+str(motorbike), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(img, "Bus: "+str(bus), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(img, "Truck: "+str(truck), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imshow('Image', img)

    if cv2.waitKey(1) == ord('q'):
        break

# Finally realese the capture object and destroy all active windows
cap.release()
cv2.destroyAllWindows()