import ktrain
import pandas as pd
from sklearn import metrics
import argparse
from ktrain import text
import subprocess
from sklearn import preprocessing
import os
import matplotlib.pyplot as plt
import string
import seaborn as sns
import random
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import stopwordsiso as stopwords

# Can't use tensorflow to recognize the GPUs as it will select all of them and you can't choose less later.


def get_available_gpus():
    if os.name == 'nt':
        # This does only work for windows.
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

# An ID to append to a folder so there are no collisions and the file can be saved


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


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

fields = ['category', 'description']
train = pd.read_csv('TRAIN CSV', usecols=fields)

test = pd.read_csv('TEST CSV', usecols=fields)

val = pd.read_csv('VAL CSV', usecols=fields)

categories = train['category'].unique().tolist()

x_train, x_test, x_val = train['description'], test['description'], val['description']

y_train, y_test, y_val = train['category'].to_numpy(
), test['category'].to_numpy(), val['category'].to_numpy()

le = preprocessing.LabelEncoder()

le.fit(y_train)

y_train, y_test, y_val = le.transform(
    y_train), le.transform(y_test), le.transform(y_val)

englishclasses = ['Cemeteries', 'Animals', 'Enforcement & nuisance', 'Environment', 'Public green space', 'Public lighting', 'Other', 'Damage',
               'Public playgrounds', 'Vandalism', 'Traffic, Signage & parking', 'Trash', 'Fireworks', 'Water, sewage & bridges', 'Roads, sidewalks & bicycle lanes']

x_train, y_train = x_train.tolist(), y_train.tolist()
x_test, y_test = x_test.tolist(), y_test.tolist()
x_val, y_val = x_val.tolist(), y_val.tolist()

t = text.Transformer(args.modelname, maxlen=args.maxlen,
                     class_names=le.classes_)

t.lang = 'nl'

trn = t.preprocess_train(x_train, y_train)
test = t.preprocess_test(x_test, y_test)
val = t.preprocess_test(x_val, y_val)


model = t.get_classifier()

learner = ktrain.get_learner(
    model, train_data=trn, val_data=test, batch_size=args.batchsize * args.multigpu)

learner.load_model('./models/7UGTV3/', t)

randomString = id_generator()

cm = learner.validate(val_data=val, print_report=True,
                      class_names=list(englishclasses))

fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d',
            xticklabels=englishclasses,
            yticklabels=englishclasses)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title("CONFUSION MATRIX - BERT\n", size=16)
plt.show()
try:
    os.makedirs('./trainedpredictors/superset/' + randomString)
except:
    pass
f = open('./trainedpredictors/superset/' +
         randomString + '/result.txt', "a")
f.write(str(cm))
f.write('\n')
f.write(str(vars(args)))
f.close()


tfidf = TfidfVectorizer(sublinear_tf=True, min_df=10,
                        ngram_range=(1, 2),
                        stop_words=stopwords.stopwords("nl"), max_features=20000)

train_features = tfidf.fit_transform(x_train).toarray()
test_features = tfidf.transform(x_test).toarray()
val_features = tfidf.transform(x_val).toarray()

model = LinearSVC()
model.fit(train_features, y_train)
print("The mean accuracy on the test data is: " +
      str(model.score(test_features, y_test)))
y_pred = model.predict(val_features)

print('\t\t\t\tCLASSIFICATIION METRICS\n')
print(metrics.classification_report(y_val, y_pred,
                                    target_names=list(englishclasses)))

conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt='d',
            xticklabels=englishclasses,
            yticklabels=englishclasses)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title("CONFUSION MATRIX - LinearSVC\n", size=16)

plt.show()
input("Press enter to exit...")
