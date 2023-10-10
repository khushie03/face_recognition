import cv2
import numpy as np
import os

dataset_path = "./data/"
facedata = []
labels = []
classid = 0
namemap = {}

for f in os.listdir(dataset_path):
    if f.endswith(".npy"):
        # Removing file extension only saving it with the file name
        namemap[classid] = f[:-4]
        
        # X-value
        data_item = np.load(os.path.join(dataset_path, f))
        m = data_item.shape[0]
        facedata.append(data_item)
        
        # Y-value
        target = classid * np.ones((m,))
        classid += 1
        labels.append(target)

Xt = np.concatenate(facedata, axis=0)
yt = np.concatenate(labels, axis=0).reshape((-1, 1))

# KNN Algorithm
def dist(p, q):
    return np.sqrt(np.sum((p - q) ** 2))

def knn(X, y, xt, k):
    m = X.shape[0]
    distances = np.zeros(m)
    
    for i in range(m):
        distances[i] = dist(X[i], xt)
    
    nearest_indices = np.argsort(distances)[:k]
    nearest_labels = y[nearest_indices]
    
    # Find the most common label among the nearest neighbors
    pred = np.bincount(nearest_labels.flatten().astype(int)).argmax()
    return int(pred)


# Predictions
cam = cv2.VideoCapture(0)
model = cv2.CascadeClassifier(r"C:\Users\Khushi Pandey\Downloads\facerecognition.xml")
offset = 20

while True:
    success, img = cam.read()

    if not success:
        print("Reading Camera Failed!")
        break

    # Detect faces in the grayscale image
    faces = model.detectMultiScale(img, 1.4, 5)

    # Choose the largest face based on the area
    for f in faces:
        x, y, w, h = f  # Extract face coordinates
        # Crop the detected face region
        cropped_image = img[y - offset : y + h + offset, x - offset : x + w + offset]
        cropped_image = cv2.resize(cropped_image, (100, 100))

        # Predict the name using knn
        classpredicted = knn(Xt, yt, cropped_image.flatten(), k=3)  # You can adjust k as needed
        name_predicted = namemap[classpredicted]
        cv2.putText(img, name_predicted, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 3), 2, cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x + w, y + h), (23, 45, 67), 2)

    cv2.imshow("Prediction Image window", img)
    # Exit loop when 'q' key is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()