
import cv2
import numpy as np
import os
from mtcnn import MTCNN
import tensorflow as tf
from tensorflow import keras
cam = cv2.VideoCapture(0)
filename = input("Enter your name: ")

dataset_path = "./data/"

offset = 20

faces_list = []

# Counter for skipping frames
skip = 0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(100, 100, 3)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

face_detector = MTCNN()

while True:
    success, img = cam.read()

    if not success:
        print("Reading Video Failed!")
        break

    faces = face_detector.detect_faces(img)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Iterate through detected faces
    for result in faces:
        x, y, w, h = result['box']

        # Check if the detected face region is valid
        if w > 0 and h > 0:
            # Draw a rectangle around the detected face
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

            # Crop the detected face region
            cropped_image = img[int(y - offset):int(y + h + offset), int(x - offset):int(x + w + offset)]
            cropped_image = cv2.resize(cropped_image, (100, 100))

            # Save every 10th face to the list
            if skip % 10 == 0:
                faces_list.append(cropped_image)
                print("Saved so far: " + str(len(faces_list)))
            skip += 1

    cv2.imshow("Video Window", img)

    # Exit loop when 'q' key is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

if len(faces_list) > 0:

    faces_list = np.asarray(faces_list)
    m = faces_list.shape[0]
    faces_list = faces_list.reshape(m, -1)

    # Create the dataset directory if it doesn't exist
    os.makedirs(dataset_path, exist_ok=True)
    filepath = os.path.join(dataset_path, filename + ".npy")
    np.save(filepath, faces_list)
    print("File Saved Successfully: " + filepath)

    # Create a subdirectory for the SavedModel
    model_save_dir = os.path.join(dataset_path, "saved_model")
    os.makedirs(model_save_dir, exist_ok=True)

    tf.saved_model.save(model, model_save_dir)

cam.release()
cv2.destroyAllWindows()
