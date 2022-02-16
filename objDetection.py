import cv2
from cvlib.object_detection import YOLO
from keyboard import is_pressed

weights = r"C:\Users\seenusanjay\.cvlib\object_detection\yolo\yolov3\yolov3.weights"
config = r"C:\Users\seenusanjay\.cvlib\object_detection\yolo\yolov3\yolov3.cfg"
classes = r"D:\pythonProject2\yolov3_classes.txt"

model = YOLO(weights=weights, config=config, labels=classes)
print(model.labels)

camera = cv2.VideoCapture(0)

while not is_pressed("shift"):
    ret, img = camera.read()
    bbox, labels, confidence = model.detect_objects(image=img)
    print(labels, confidence, sep="\n")
    model.draw_bbox(img, bbox, labels, confidence)
    cv2.imshow("object detection", img)
    cv2.waitKey(1)

camera.release()
cv2.destroyAllWindows()
