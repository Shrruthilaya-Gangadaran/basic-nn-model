# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The Neural network model contains input layer,two hidden layers and output layer. Input layer contains
2 single neuron. Output layer also contains two neuron.First hidden layer contains four neurons and
second hidden layer contains three neurons. A neuron in input layer is connected with every neurons in a
first hidden layer. Similarly, each neurons in first hidden layer is connected with all neurons in second
hidden layer. All neurons in second hidden layer is connected with output layered neuron. Relu
activation function is used here. It is linear neural network model(single input neuron forms single
output neuron).

## Neural Network Model

![](nn_model.webp)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
~~~
Program developed by : Shrruthilaya G
Register number : 212221230097

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df=pd.read_csv("/content/Datasheet - Sheet1.csv")

df.head()

X = df[["Input"]].values
X

Y = df[["Output"]].values
Y

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.33, random_state=33)

Scaler = MinMaxScaler()

Scaler.fit(X_train)
Scaler.fit(X_test)

X_train1 = Scaler.transform(X_train)
X_test1 = Scaler.transform(X_test)
X_train1

ai_brain = Sequential([
    Dense(8, activation = 'relu'),
    Dense(10, activation = 'relu'),
    Dense(1)])

ai_brain.compile(
    optimizer = 'rmsprop',
    loss = 'mse'
)

ai_brain.fit(X_train1, Y_train, epochs=2000)

lossai_brain=pd.DataFrame(ai_brain.history.history)
lossai_brain.plot()

ai_brain.evaluate(X_test1,Y_test)

Xn1=[[30]]
Xn11=Scaler.transform(Xn1)
ai_brain.predict(Xn11)

~~~
### Dataset Information

![](dataset.png)

## OUTPUT

### Training Loss Vs Iteration Plot

![](plot.png)

### Test Data Root Mean Squared Error

![](rmse.png)

### New Sample Data Prediction

![](newprediction.png)

## RESULT
Thus, the neural network model regression model for the given dataset is developed.