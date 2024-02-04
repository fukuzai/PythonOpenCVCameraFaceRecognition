import cv2 as cv

#img = cv. imread("C:/Users/keirf/Pictures/Lenna.png")
#cv.imshow("Display window", img)
#k = cv.waitKey(0)

#Load the Classifier
face_classifier = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    #Perform the Face Detection
    face = face_classifier.detectMultiScale(
    frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    #Drawing a Bounding Box
    for (x, y, w, h) in face:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

    #Displaying the Image
    img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Display the resulting frame
    cv.imshow('frame', img_rgb)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()