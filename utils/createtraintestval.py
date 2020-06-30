from sklearn.model_selection import train_test_split
import pandas as pd
import random

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

def get_datasets(X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.111)

    x_train, x_test, x_val, y_train, y_test, y_val = x_train.to_numpy(), x_test.to_numpy(), x_val.to_numpy(), y_train.to_numpy(), y_test.to_numpy(), y_val.to_numpy()
    return x_train, x_test, x_val, y_train, y_test, y_val


fields = ['category', 'description']
data = pd.read_csv('SAMPLE DATA CSV', usecols=fields) #Input your data file here.

data = cleanup_data(data)

data = balance_data(data)

X = data['description']
y = data['category']

x_train, x_test, x_val, y_train, y_test, y_val = get_datasets(X, y)

train = pd.DataFrame({"description": x_train, "category": y_train})

train.to_csv('train.csv')

test = pd.DataFrame({"description": x_test, "category": y_test})

test.to_csv('test.csv')

val = pd.DataFrame({"description": x_val, "category": y_val})

val.to_csv('val.csv')