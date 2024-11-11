import json # to use json files
import random # aleatoire choices
import numpy as np # to use numpy
import tensorflow as tf # for IA
from data_preprocess import vocabulary , tags , generate_chatbot_response
from nltk_utils import tokenize , bag_of_words


#opening the json file

with open('intents.json','r') as file:
    intents = json.load(file)

#downloding the IA model
chatbot_brain = tf.keras.models.load_model("Baristas.keras")

bot_name= "jack"
print("Let's chat! (type 'quit' to exit)")

while True:
    sentence = input("You: ")
    if sentence == "quit":
        break

    # preprocessing 
    sentence = tokenize(sentence)
    x = bag_of_words(sentence,vocabulary)
    x = x.reshape(1,-1) # to make it a 2D array for the modal bech yfhemha

    #prediction the response


    # prediction = chatbot_brain.predict(X)
    # prediction_index = np.argmax(prediction)
    # tag = tags[prediction_index]
    # response = generate_chatbot_response(tag,intents)
    # print(f"{bot_name}: {response}")

    pred_proba , intent_tags = generate_chatbot_response(chatbot_brain,x,tags)

    # verification +affichage 
    if pred_proba > 0.75:
        for intent in intents['intents']:
            if intent['tag'] == intent_tags:
                print(f"{bot_name}: {random.choice(intent['responses'])}") # the f is like the template literals in js ${  }
    else:
        print(f"{bot_name}: I do not understand...")


    
    


    