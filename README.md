# Chatbot Ai
This system is a chatbot that includes features like question-answering, facial recognition, image classification, and knowledge base management. It makes use of a variety of tools and packages, including NLTK for natural language processing, TensorFlow for image classification, Haar Cascade Classifier for face identification, and AIML for pattern matching.

# Below are some of the  system's essential parts:
- AIML (Artificial Intelligence Markup Language) Kernel: Used for pattern matching-based processing and user input response.
-	Text-to-speech software transforms text responses into speech using the gTTS library.
-	A pre-trained model of a convolutional neural network (CNN) is used to categorise pictures of lions and tigers.
-	Identifies faces in pictures and video streams using the Haar Cascade Classifier.
-	Knowledge base: A set of facts and assertions stored in a CSV file, which can be queried and updated by the user.
-	TfidfVectorizer: Transforms text to feature vectors that can be used to calculate the similarity between user inputs and the questions in the chatbot's knowledge base.
-	CoreNLPParser: A parser from the NLTK library, which is used for natural language processing tasks.

# The system should have the following requirements and functionalities from a user's perspective:
- Engage in a text-based conversation with the user.
-	Perform video face recognition and display the detected faces on the screen.
-	Classify images as either a lion or a tiger with associated probabilities.
-	Allow the user to provide new knowledge in the form of "I know that * is *" statements.
-	Enable the user to check if a statement is true, false, or unknown based on the system's knowledge base.
-	Respond to "What do you know?" query with a list of all known statements.
-	Match user input with AIML patterns and provide a response based on the pattern matching.
-	If no AIML match is found, use similarity-based matching to provide a response based on previously seen questions and answers.

# To use the system, users should have the following libraries and tools installed:
-	Python 3.x
-	nltk
-	scikit-learn
-	CoreNLP server
-	TensorFlow
-	Keras
-	gTTS
-	playsound
-	PyExpat
-	AIML
-	OpenCV
-	pandas
-	keras-tuner
-	Tkinter
-	urllib.request

# Screenshots
![image](https://github.com/Clive20/Chatbot-Ai/assets/74508019/1a67e4b3-81e4-48c9-a791-7c447df904bb)
![image](https://github.com/Clive20/Chatbot-Ai/assets/74508019/db88ad9c-5e99-42d2-9a3a-71e314c921bb)
![image](https://github.com/Clive20/Chatbot-Ai/assets/74508019/f950c2dd-fbf2-48fe-b976-a882f2cc0445)
![image](https://github.com/Clive20/Chatbot-Ai/assets/74508019/eda1cb1b-0bad-4af2-8d2f-d6643dc108f2)
![image](https://github.com/Clive20/Chatbot-Ai/assets/74508019/144421b9-ca67-489f-a97f-fd3652bf25ef)



