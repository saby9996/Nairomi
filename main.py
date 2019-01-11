## Basic Imports
import pandas as pd
import numpy as np
import scipy
import tensorflow as tf
import sklearn
import warnings
warnings.filterwarnings("ignore")
import simplejson as json
from IPython.core.display import Image, display

#Imports For Speech to Text Conversion
import speech_recognition as sr
#Inports For Text To Speech
import pyttsx3

#RASA Import
import rasa_nlu
import rasa_core
import spacy
from rasa_nlu.training_data import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer
from rasa_nlu import config
from rasa_core.actions import Action
from rasa_nlu.evaluate import run_evaluation
from rasa_core.events import SlotSet
from rasa_core.policies import FallbackPolicy, KerasPolicy, MemoizationPolicy
from rasa_core.agent import Agent


import requests

#Capturing Audio for Text
def audio_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.1)
        audio = r.listen(source)
    text=r.recognize_google(audio)
    return text

#Rendering Audio for Output From Text
def audio_output(text):
    engine = pyttsx3.init()
    engine.setProperty('voice', voice.id[1])
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()

#Training the Model
training_data = load_data("nlu.md")
trainer = Trainer(config.load("config.yml"))
interpreter = trainer.train(training_data)
model_directory = trainer.persist("./models/nlu", fixed_model_name="current")

#Evaluate NLU Model on Random Text
def pprint(o):
    print(json.dumps(o, indent=2))
pprint(interpreter.parse("I am very sad. Could you send me a cat picture? "))

#Evaluating on Test Data
run_evaluation("nlu.md", model_directory)


class ApiAction(Action):
    def name(self):
        return "action_retrieve_image"

    def run(self, dispatcher, tracker, domain):
        group = tracker.get_slot('group')

        r = requests.get('http://shibe.online/api/{}?count=1&urls=true&httpsUrls=true'.format(group))
        response = r.content.decode()
        response = response.replace('["', "")
        response = response.replace('"]', "")

        # display(Image(response[0], height=550, width=520))
        dispatcher.utter_message("Here is something to cheer you up: {}".format(response))


#Training Dialogue Model
fallback = FallbackPolicy(fallback_action_name="utter_unclear", core_threshold=0.2,nlu_threshold=0.1)
agent = Agent('domain.yml', policies=[MemoizationPolicy(), KerasPolicy(), fallback])
# loading our neatly defined training dialogues
training_data = agent.load_data('stories.md')
agent.train(training_data,validation_split=0.0)
agent.persist('models/dialogue')

#Starting the Bot
from rasa_core.agent import Agent
agent = Agent.load('models/dialogue', interpreter=model_directory)

#################################################Starting Nairomi Agent###############################################
audio_output("Hi There !!! , This is Nairomi, How can I help You ?'")
while True:
    a = audio_input().capitalize()
    print(a)
    if a == 'Stop':
        break
    responses = agent.handle_message(a)
    for response in responses:
        print(response["text"])
        audio_output(response["text"])























