from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import random
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Importing the dataset
training = pd.read_csv('C:\\Users\\user\\Downloads\\Healtho-Healthcare_Chatbot-main\\Healtho-Healthcare_Chatbot-main\\Training.csv')
testing = pd.read_csv('C:\\Users\\user\\Downloads\\Healtho-Healthcare_Chatbot-main\\Healtho-Healthcare_Chatbot-main\\Testing.csv')
# saving the information of columns
cols = training.columns
cols = cols[:-1]
# Slicing and Dicing the dataset to separate features from predictions
x = training[cols]
y = training['prognosis']
y1 = y

# dimensionality Reduction for removing redundancies
reduced_data = training.groupby(training['prognosis']).max()

# encoding/mapping String values to integer constants
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

# Splitting-the-dataset-into-training-set-and-test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        symptoms = request.form.getlist('symptoms')
        symptoms_present = list(map(int, symptoms))

        def print_disease(node):
            node = node[0]
            val = node.nonzero()
            disease = le.inverse_transform(val[0])
            return disease

        def tree_to_code(tree, feature_names):
            tree_ = tree.tree_

            feature_name = [
                feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
                for i in tree_.feature
            ]
            symptoms_present = []

            def recurse(node, depth):
                if tree_.feature[node] != _tree.TREE_UNDEFINED:
                    threshold = tree_.threshold[node]
                    if symptoms_present[node] <= threshold:
                        recurse(tree_.children_left[node], depth + 1)
                    else:
                        symptoms_present.append(feature_name[node])
                        recurse(tree_.children_right[node], depth + 1)
                else:
                    present_disease = print_disease(tree_.value[node])
                    for di in present_disease:
                        diss = di
                    for i in symptoms_present:
                        dis = i
                    return diss, dis

            return recurse(0, 1)

        def decide(symptoms_present):
            clf = DecisionTreeClassifier()
            clf.fit(x_train, y_train)
            symptoms_present = np.array(symptoms_present).reshape(1, -1)
            disease, symptoms = tree_to_code(clf, cols)(symptoms_present)
            return disease, symptoms

        disease, symptoms = decide(symptoms_present)
        return render_template('result.html', disease=disease, symptoms=symptoms)


if __name__ == '__main__':
    app.run(debug=True)
