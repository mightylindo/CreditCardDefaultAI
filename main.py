#Import the libraries
import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle
from sklearn.metrics import confusion_matrix

# Read in the csv file
data = pd.read_csv("default_of_credit_card_clients.csv", sep=",")

# Create the array with the desired attributes
data = data[["LIMIT_BAL", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1",
             "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6", "default payment next month"]]

print(data.head()) # This prints out the first 5 rows of the data frame and allows us to check that its reading in correctly


# This is the attribute we will try to predict
predict = "default payment next month"

# This creates the training sets and testing sets
X = np.array(data.drop([predict], 1))
print(X)
y = np.array(data[predict])
print(y)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.5)

# This code trains the model by going through 30 training runs and saves the best model
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.5)

    model = LogisticRegression()

    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    print(acc)

    if acc > best:
        best = acc
        with open("defaultmodel.pickle", "wb") as f:
            pickle.dump(model, f)

pickle_in = open("defaultmodel.pickle", "rb")
model = pickle.load(pickle_in)

p = "LIMIT_BAL"
style.use("ggplot")
pyplot.scatter(data[p], data["default payment next month"])
pyplot.xlabel(p)
pyplot.ylabel("default payment next month")
pyplot.show()

p = "BILL_AMT1"
style.use("ggplot")
pyplot.scatter(data[p], data["default payment next month"])
pyplot.xlabel(p)
pyplot.ylabel("default payment next month")
pyplot.show()

p = "PAY_AMT1"
style.use("ggplot")
pyplot.scatter(data[p], data["default payment next month"])
pyplot.xlabel(p)
pyplot.ylabel("default payment next month")
pyplot.show()

# This code predicts which customers will default next
predictions = model.predict(x_test)
# This stores the models predictions in a variable,
# and takes the full list of customers without the default attribute

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_true=y_test, y_pred=predictions)
fig, ax = pyplot.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=pyplot.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

pyplot.xlabel('Predictions', fontsize=18)
pyplot.ylabel('Actuals', fontsize=18)
pyplot.title('Confusion Matrix', fontsize=18)
pyplot.show()

for x in range(len(predictions)):
    # This loop walks through all the predictions and provides the customerIDs for the customers
    # predicted to default.
    if predictions[x] == 1:
        print("Customer ID: ", x)
