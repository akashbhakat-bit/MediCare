from flask import Flask, render_template, request, redirect
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

def load_image(img_path):

    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]



    return img_tensor

def covid_test():
    # load model
    model = load_model('covid_model.h5')


    img_path = 'img.jpg'  

    new_image = load_image(img_path)

    # check prediction
    class1=['Covid 19 Detected','Normal']
    pred = model.predict(new_image)
    predicted_class_indices=np.argmax(pred,axis=1)
    print(class1[predicted_class_indices[0]])
    return (class1[predicted_class_indices[0]])

def pneumonia_test():
    
    # load model
    model = load_model('pneumonia_model.h5')


    img_path = 'img.jpg'  

    new_image = load_image(img_path)

    # check prediction
    class1=['Normal','Pneumonia Detected']
    pred = model.predict(new_image)
    predicted_class_indices=np.argmax(pred,axis=1)
    print(class1[predicted_class_indices[0]])
    return (class1[predicted_class_indices[0]])

def diabetes_retinopathy_test():
    # load model
    model = load_model('diabetes_retinopathy_model.h5')


    img_path = 'img.jpg'  

    new_image = load_image(img_path)

    # check prediction
    class1=['Diabetes Retinopathy Detected','Normal']
    pred = model.predict(new_image)
    predicted_class_indices=np.argmax(pred,axis=1)
    print(class1[predicted_class_indices[0]])
    return (class1[predicted_class_indices[0]])

def skin_cancer():
    # load model
    model = load_model('skin_cancer_model.h5')


    img_path = 'img.jpg'  

    new_image = load_image(img_path)

    # check prediction
    class1=['Benign','Malignant Detected']
    pred = model.predict(new_image)
    predicted_class_indices=np.argmax(pred,axis=1)
    print(class1[predicted_class_indices[0]])
    return (class1[predicted_class_indices[0]])

##########

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res




app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET', 'POST'])
def main():
    
    return render_template('footer.html',appoint=1,disease=0,machine=0)

@app.route('/change_appoint', methods=['GET', 'POST'])
def change_appoint():

    return render_template('footer.html',appoint=1,disease=0,machine=0)

@app.route('/change_disease', methods=['GET', 'POST'])
def change_disease():
    return render_template('footer.html',disease=1,appoint=0,machine=0)

@app.route('/change_machine', methods=['GET', 'POST'])
def change_machine():
    return render_template('footer.html',machine=1,disease=0,appoint=0)

#@app.route('/disease_diagnose', methods=['GET', 'POST'])
#def disease_diagnose():


#@app.route('/symptoms', methods=['GET', 'POST'])
#def symptoms_diagnose():
    

   



if __name__ == "__main__":
    app.run(debug=True, port=8000)