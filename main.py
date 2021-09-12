import cv2
import numpy as np

#First we need to input the yolov3 files available

net = cv2.dnn.readNet('yolov3.weights','yolov3.cfg')

classes = []

#input the coco Dataset

with open('coco.names','r') as f:
    classes = f.read().splitlines()

#there are three types of inputs available in yolo algorithm
#1.image file
#2.vedio file
#3.webcam

cap = cv2.VideoCapture('sample.mp4')
#img = cv2.imread('obj.jfif')

while True:
    #capture the video in frames

    _, img = cap.read()
    height, width, _ = img.shape

    #we use blobfromimage function to input the pictures

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)

    net.setInput(blob)

    #getoutputlayers function to determine the output layers

    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    #two for loops to identify the objects in the frames

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    #NMS boxes to keep only the highest score boxes

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))

    #last for loop to pass all the information from the picture or videos

    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = colors[i]
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255, 255, 255), 2)

    #print the output image,video

    cv2.imshow("image",img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()