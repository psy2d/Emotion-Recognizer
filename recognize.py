"""
Title: Emotion Recognition (Happy / Sad / Neutral)
Author: Sayali Deshpande
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.platform import gfile

# The file containing the frozen graph of the trained model
MODEL_FILE = 'pb/emotion_classifier.pb'

IMAGE_SIZE = 48
NUM_CHANNELS = 1
EMOTIONS=['Happy','Sad','Neutral']

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0)
with tf.Session() as sess:
    print("load graph")
    with gfile.FastGFile(MODEL_FILE,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        # Get input and output tensors from graph
        x_input = sess.graph.get_tensor_by_name("input:0")
        output = sess.graph.get_tensor_by_name("output:0")

        while True:
            # Read a frame
            success, frame = cap.read()

            # Convert it to grayscale for recognition
            gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY )

            # Detect faces
            faces = face_cascade.detectMultiScale(gray)

            for (x,y,w,h) in faces:

                cropped_frame = gray[y:y+h, x:x+w]

                # Crop out the face and do necessary preprocessing
                cropped_frame=cv2.resize(cropped_frame,(IMAGE_SIZE,IMAGE_SIZE)).reshape((1,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS))
                cropped_frame=(cropped_frame-np.mean(cropped_frame))*(2.0/255.0)

                # Feed the cropped and preprocessed frame to classifier
                result=sess.run(output,{x_input:cropped_frame})

                # Get the emotion and confidence
                emotion=EMOTIONS[np.argmax(result)]
                confidence=np.max(result)

                # Draw rectangle and write text on frame to be displayed
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.putText(frame,emotion+": %.1f"%confidence,(x,y-2),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2,cv2.LINE_AA)
                cv2.imshow('Image Fed to Classifier', cv2.resize(cropped_frame.reshape((48,48)), (48,48)))

            cv2.imshow('Emotion Recognition', cv2.resize(frame, (600,450)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
