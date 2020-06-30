import tensorflow as tf
from ktrain import text
import ktrain
import re
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import argparse
import os
import numpy as np
import subprocess
import time
import random
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn.utils.class_weight as k
import string

#Can't use tensorflow to recognize the GPUs as it will select all of them and you can't choose less later.
def get_available_gpus():
    if os.name == 'nt':
        #This does only work for windows.
        process = subprocess.Popen(
            'nvidia-smi -q', shell=True, stdout=subprocess.PIPE)
        output = str(process.stdout.read())
        k = output.split(' ')
        num_gpus = int(k[108][:1])
    else:
        num_gpus = input("How many GPUs are available? ")
    return num_gpus

parser = argparse.ArgumentParser()
parser.add_argument('-mg', '--multigpu', help="Multiple GPU support", nargs='?',
                    type=int, default=get_available_gpus(), const=get_available_gpus())
parser.add_argument('-m', '--modelname', help="The pre-trained model used for learning",
                    nargs='?', type=str, default="wietsedv/bert-base-dutch-cased", const="wietsedv/bert-base-dutch-cased")
parser.add_argument('-p', '--predictor', help="Boolean flag to decide whether or not to save a predictor",
                    default=False, action='store_true')
parser.add_argument('-l', '--losses', help="Boolean flag to decide whether or not to view top losses",
                    default=False, action='store_true')
parser.add_argument('-v', '--validate', help="Boolean flag to decide whether or not to save a validate report",
                    default=False, action='store_true')
parser.add_argument('-md', '--model', help="Boolean flag to decide whether or not to save the model",
                    default=False, action='store_true')
parser.add_argument('-b', '--batchsize', help="The size of the batches taken at one time",
                    nargs='?', type=int, default=16, const=16)
parser.add_argument('-lr', '--learningrate', help="The rate of which the model adjusts learning",
                    nargs='?', type=float, default=1e-4, const=1e-4)
parser.add_argument('-fr', '--findrate', help="Will estimate a good learning rate",
                    default=False, action='store_true')
parser.add_argument('-e', '--epochs', help="The amount of epochs the learner runs",
                    nargs='?', type=int, default=2, const=2)
parser.add_argument('-ml', '--maxlen', help="The maximum length for tokens",
                    nargs='?', type=int, default=200, const=200)
parser.add_argument('-c', '--compare', help="Value of > 0 to start compare with linearsvc and > 1 to compare with 3 more",
                    nargs='?', type=int, default=0, const=0)
parser.add_argument('-lp', '--policy', help="Choose a learning policy(autofit(Triangular policy), fit_onecycle or fit(SGDR policy))",
                    nargs="?", type=str, default='fit_onecycle', const='fit_onecycle')

args = parser.parse_args()

devices = ""
if get_available_gpus() < args.multigpu:
    print("You dont have that many GPUs available on your machine, taking all GPUs available...")
    args.multigpu = get_available_gpus()
    print("Using {} GPUs...".format(args.multigpu))
elif args.multigpu and args.multigpu > 1:
    print("Using {} GPUs...".format(args.multigpu))
    for i in range(0, args.multigpu):
        devices += str(i) + ","
else:
    print(
        "Single GPU mode activated. Use --multigpu or -mg {GPU AMOUNT (> 1)} for multi GPU support")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = devices[:-1]

#Removes special characters from all issues, as well as empty values or issues with only numbers.
def cleanup_data(data):
    data = data[pd.notnull(data['category'])]
    data = data[pd.notnull(data['description'])]

    patternDel = "^[0-9]*$"
    data = data[data['description'].str.match(patternDel) == False]
    spec_chars = ["!",'"',"#","%","&","'","(",")",
              "*","+","-","/",":",";","<",
              "=",">","[","\\","]","^","_",
              "`","{","|","}","~","–"]
    for char in spec_chars:
        data["description"] = data["description"].str.replace(char, '')
    return data

#To balance the input data in the case of imbalance.
def balance_data(data):
    max_size = data['category'].value_counts().max()
    lst = []
    for _, group in data.groupby('category'):
        if len(group) < 10000:
            lst.append(group.sample(
                int(random.randint(int(max_size / 2), max_size) / 3), replace=True))
        else:
            lst.append(group.sample(frac=0.5))
    frame_new = pd.concat(lst)
    return frame_new.sample(frac=1)

#An ID to append to a folder so there are no collisions and the file can be saved
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

#To create a 80, 10, 10 set-up.
def get_datasets(X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.111) 
    return x_train, x_test, x_val, y_train, y_test, y_val


fields = ['category', 'description']
data = pd.read_csv('DATA CSV', usecols=fields) #Input your data file here.

data = cleanup_data(data)

data = balance_data(data)

categories = data['category'].unique().tolist()

mirrored_strategy = tf.distribute.MirroredStrategy()

X = data['description']
y = data['category']

x_train, x_test, x_val, y_train, y_test, y_val = get_datasets(X, y)

t = text.Transformer(args.modelname, maxlen=args.maxlen,
                     class_names=categories)

t.lang = 'nl'

class_weight = k.compute_class_weight('balanced', categories, y_train)

le = preprocessing.LabelEncoder()

le.fit(y_train)

y_train, y_test, y_val = le.transform(
    y_train), le.transform(y_test), le.transform(y_val)

x_train, y_train = x_train.tolist(), y_train.tolist()
x_test, y_test = x_test.tolist(), y_test.tolist()
x_val, y_val = x_val.tolist(), y_val.tolist()

trn = t.preprocess_train(x_train, y_train)
test = t.preprocess_test(x_test, y_test)

model = None

if args.multigpu > 1:
    with mirrored_strategy.scope():
        model = t.get_classifier()
else:
    model = t.get_classifier()

learner = ktrain.get_learner(
    model, train_data=trn, val_data=test, batch_size=args.batchsize * args.multigpu)

#Plots the learning rate simulation graph
if args.findrate:
    learner.lr_find(show_plot=True, max_epochs=args.epochs)
    plt.show()

#Learns with one of the three policies
if args.policy == 'fit':
    learner.fit(args.learningrate, args.epochs, class_weight=class_weight)
elif args.policy == 'autofit':
    learner.autofit(args.learningrate, args.epochs, class_weight=class_weight)
else:
    args.policy = 'fit_onecycle'
    learner.fit_onecycle(args.learningrate, args.epochs,
                         class_weight=class_weight)

randomString = id_generator()

#Save the fine-tuned model
if args.model:
    try:
        os.makedirs('./models/' + randomString)
    except:
        pass
    learner.save_model('./models/' + randomString)

val = t.preprocess_test(x_val, y_val)

#Validates the model and creates a confusion matrix as well as a classification report
if args.validate:
    cm = learner.validate(val_data=val, print_report=True, class_names=t.get_classes())
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt='d',
                xticklabels=categories,
                yticklabels=categories)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title("CONFUSION MATRIX - BERT\n", size=16)
    plt.show()
    input()
    try:
        os.makedirs('../trainedpredictors/supersettests/' + randomString)
    except:
        pass
    f = open('../trainedpredictors/supersettests/' +
             randomString + '/result.txt', "a")
    f.write(str(vars(args)))
    f.close()

#Shows the top losses for the validation data
if args.losses:
    learner.view_top_losses(n=10, preproc=t, val_data=val)

#Saves a predictor object for the fine-tuned model
if args.predictor:
    try:
        os.makedirs('../trainedpredictors/supersettests/' + randomString)
    except:
        pass
    predictor = ktrain.get_predictor(learner.model, t)
    predictor.save('../trainedpredictors/supersettests/' +
                   randomString + '/predictor')