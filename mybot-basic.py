import csv
import time
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.parse import CoreNLPParser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sem import Expression
from nltk.inference import ResolutionProver
import time
time.clock = time.time
from gtts import gTTS
import os
from playsound import playsound
from pyexpat import model
import aiml
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog
import urllib.request
import io
import kerastuner as kt

# Load AIML kernel
import aiml
kernel = aiml.Kernel()
kernel.learn("mybot-basic.xml")

# Function to play text as speech
def speak(text):
    text_to_speech_engine = gTTS(text=text, lang='en')
    text_to_speech_engine.save('response.mp3')
    os.system('start response.mp3')

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=2, activation="softmax"))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the CNN model
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train = train_datagen.flow_from_directory("dataset/train", target_size=(128, 128), batch_size=32, class_mode="categorical")
test = test_datagen.flow_from_directory("dataset/test", target_size=(128, 128), batch_size=32, class_mode="categorical")
model.fit(train, steps_per_epoch=len(train), epochs=5, validation_data=test, validation_steps=len(test))

# Save the trained CNN model
model.save("my_model.h5")

# Load the pre-trained model
pretrained_model = tf.keras.models.load_model("my_model.h5")

# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load Q/A pairs from csv file
question_answer_pairs = []
with open('mybot-basic.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        question_answer_pairs.append(row)

# Load knowledge base from csv file
knowledge_base = []
with open('knowledge_base.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        expr = Expression.fromstring(row[0])
        knowledge_base.append(expr)

# Define vectorizer using TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()


# Create a document-term matrix
document_term_matrix = tfidf_vectorizer.fit_transform([q[0] for q in question_answer_pairs])

# Set up CoreNLPParser
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('corenlp')
stop_words = set(stopwords.words('english'))
parser = CoreNLPParser(url='http://localhost:9000', tagtype='pos')
parser.parser_annotator='parse'
parser.parser_options=' -retainTmpSubcategories'

# Define a function to detect faces in an image
def detect_faces(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Define a function to capture video and perform face recognition
def video_face_recognition():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        faces = detect_faces(frame)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Video Face Recognition', frame)

        if cv2.waitKey(10) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Define a function to classify images
def classify_image(image_path, model, image_url=False):
    try:
        # Load and preprocess the image
        if not image_url:
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
        else:
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128), format='jpeg')
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.

        # Use the pre-trained model to predict the content of the image
        prediction = model.predict(img_array)
        print("Raw prediction:", prediction)
        prediction = prediction[0]


        # Return the predicted class and probabilities
        if prediction[0] > prediction[1]:
            return "This is a Lion.", prediction[0], prediction[1]
        else:
            return "This is a Tiger.", prediction[1], prediction[0]
    except:
        # If there was an error in classifying the image, return an error message
        return "Sorry, there was an error classifying the image.", None, None

# Welcome the user
opening_message = "Welcome to this chat bot. Please feel free to ask questions from me!"
print(opening_message)
speak(opening_message)
# Main loop
while True:
    # Get user input
    user_input = input("> ").upper()

    # Check if user wants to EXIT
    if user_input == "EXIT":
        print("goodbye!")
        break

    response = kernel.respond(user_input)

    if response == "VIDEO_FACE_RECOGNITION":
        print("Bot: Starting video face recognition. Press 'q' to quit.")
        video_face_recognition()

    elif response == "IMAGE_CLASSIFICATION":
        print("Choose the image source:")
        print("1. File")
        print("2. URL")
        print("3. Camera")

        source = int(input("Enter the number corresponding to the image source: "))

        if source == 1:
            root = tk.Tk()
            root.withdraw()
            image_path = filedialog.askopenfilename()
        elif source == 2:
            image_path = input("Please enter the file path for the image: ")
        elif source == 3:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cv2.imwrite("camera_image.jpg", frame)
            image_path = "camera_image.jpg"
            cap.release()
            cv2.destroyAllWindows()
        else:
            print("Invalid option, try again.")
            continue

        try:
            classification, prob_lion, prob_tiger = classify_image(image_path, pretrained_model)
            print("Bot: " + classification)
            print("Lion probability: {:.2f}%".format(prob_lion * 100))
            print("Tiger probability: {:.2f}%".format(prob_tiger * 100))
        except Exception as e:
            print("Bot: Sorry, there was an error classifying the image.")
            print(f"Error message: {str(e)}")

    # Process "I know that * is *" statements
    elif user_input.startswith("I KNOW THAT "):
        object, subject = user_input[12:].split(" IS ")
        expr = Expression.fromstring(subject + "(" + object + ")")
        # Check for contradiction before appending to knowledge_base
        contradiction = ResolutionProver().prove(Expression.fromstring('not(' + str(expr) + ')'), knowledge_base)
        if contradiction:
            print("Error: The knowledge base already contains a contradiction for the statement:", str(expr))
        else:
            knowledge_base.append(expr)
            # Append new knowledge to knowledge_base.csv
            with open('knowledge_base.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([str(expr)])

            print("OK, I will remember that", object, "is", subject)

    # Process "CHECK THAT * IS *" statements
    elif user_input.startswith("CHECK THAT "):
        object, subject = user_input[11:].split(" IS ")
        expr = Expression.fromstring(subject + "(" + object + ")")
        answer = ResolutionProver().prove(expr, knowledge_base, verbose=True)
        if answer:
            print("Correct.")
        else:
            # Check if expr is false
            contradiction = ResolutionProver().prove(Expression.fromstring('not(' + str(expr) + ')'), knowledge_base)
            if contradiction:
                print("Incorrect.")
            else:
                print("Sorry, I don't know.")

    # Process "What do you know?" statements
    elif user_input == "WHAT DO YOU KNOW?":
        print("Here is what I know:")
        for statement in knowledge_base:
            print(statement)

    else:
        # Match user input with AIML patterns
        aiml_response = kernel.respond(user_input)
    
        # If no AIML match, use similarity-based matching
        if aiml_response == "":
            # Convert user input to document-term frequency vector
            user_vector = tfidf_vectorizer.transform([user_input])
        
            # Compute cosine similarity between user vector and document-term matrix
            similarity_scores = cosine_similarity(user_vector, document_term_matrix)[0]
        
            # Find index of most similar question
            most_similar_index = similarity_scores.argmax()
        
            # Get the answer associated with the most similar question
            response = question_answer_pairs[most_similar_index][1]
        
            # Print answer
            print(response)
            speak(response)
        
        else:
            # Print AIML response
            print(aiml_response)
            speak(aiml_response)
