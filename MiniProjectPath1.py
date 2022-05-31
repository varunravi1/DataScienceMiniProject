import pandas
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
from sklearn import metrics
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso



''' 
The following is the starting code for path1 for data reading to make your first step easier.
'dataset_1' is the clean data for path1. 

'''
'''Functions we used'''
def get_train(Bridge ):
    arr = []
    for i in range(0, 173, 1):
        arr.append(Bridge[i])
    return arr
def get_test(Bridge):
    arr = []
    for i in range(173, 214):
        arr.append(Bridge[i])
    return arr


def normalize_train(X_train):


    mean = np.mean(X_train,axis=0)
    std = np.std(X_train, axis=0)
    X = (X_train - mean ) / std

    return X

def train_model(X, y):

    # fill in
    model = Ridge(fit_intercept=True)
    model.fit(X, y)

    return model
def trainLo_model(X, y):
    model = LogisticRegression(fit_intercept=True)
    model.fit(X, y.ravel())
    return model


def error(X, y, model):

    # Fill in
    y_pred = model.predict(X)
    total = (y - y_pred)**2
    mse = total.mean()
    return mse

def pred(X, model):
    return model.predict(X)

def rSquared (X, y, model,y_test):
    y_pred = model.predict(X)
    #print(y_pred)
    return r2_score(y_test, y_pred)
def trainL_model(X, y):
    model = LinearRegression(fit_intercept=True)
    model = model.fit(X, y)
    return model
def trainLasso_model(X,y):
    model = Lasso(alpha = 0.1)
    model = model.fit(X,y)
    return model


dataset_1 = pandas.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
dataset_1['Brooklyn Bridge']      = pandas.to_numeric(dataset_1['Brooklyn Bridge'].replace(',','', regex=True))
dataset_1['Manhattan Bridge']     = pandas.to_numeric(dataset_1['Manhattan Bridge'].replace(',','', regex=True))
dataset_1['Queensboro Bridge']    = pandas.to_numeric(dataset_1['Queensboro Bridge'].replace(',','', regex=True))
dataset_1['Williamsburg Bridge']  = pandas.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))
#dataset_1['Williamsburg Bridge']  = pandas.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))
dataset_1['Total']  = pandas.to_numeric(dataset_1['Total'].replace(',','', regex=True))




"""The use of the code provided is optional, feel free to add your own code to read the dataset. The use (or lack of use) of this code is optional and will not affect your grade."""

############################################ Question 1 #####################################################################
 #create feature matrices

matrixBMQ = np.column_stack((dataset_1['Brooklyn Bridge'] , dataset_1['Manhattan Bridge'], dataset_1['Queensboro Bridge']))
matrixMQW = np.column_stack((dataset_1['Williamsburg Bridge'], dataset_1['Manhattan Bridge'], dataset_1['Queensboro Bridge']))
matrixBQW = np.column_stack((dataset_1['Williamsburg Bridge'], dataset_1['Brooklyn Bridge'], dataset_1['Queensboro Bridge']))
matrixBMW = np.column_stack((dataset_1['Brooklyn Bridge'], dataset_1['Manhattan Bridge'], dataset_1['Williamsburg Bridge']))
total = np.array(dataset_1['Total'])
total = total.reshape(214, 1)
#print(np.column_stack((dataset_1['Brooklyn Bridge'] , dataset_1['Manhattan Bridge'], dataset_1['Williamsburg Bridge'], dataset_1['Queensboro Bridge'])))

matrixBMQ_train, matrixBMQ_test,yBMQ_train, yBMQ_test = train_test_split(matrixBMQ, total, test_size=0.2, random_state=42)
matrixMQW_train, matrixMQW_test,yMQW_train, yMQW_test = train_test_split(matrixMQW, total, test_size =0.2,random_state=42)
matrixBQW_train, matrixBQW_test,yBQW_train, yBQW_test = train_test_split(matrixBQW, total, test_size=0.2,random_state=42)
matrixBMW_train, matrixBMW_test,yBMW_train, yBMW_test = train_test_split(matrixBMW, total, test_size=0.2, random_state=42)

y_test = [yBMQ_test, yMQW_test, yBQW_test, yBMW_test]
y_testNorm = [normalize_train(x) for x in y_test]
y_train = [yBMQ_train, yMQW_train, yBQW_train, yBMW_train]
y_trainNorm = [normalize_train(x) for x in y_train]


test_matrix = [matrixBMQ_test, matrixMQW_test, matrixBQW_test, matrixBMW_test]
#normalize test

test_matrixNorm = [normalize_train(x) for x in test_matrix]

train_matrix = [matrixBMQ_train, matrixMQW_train, matrixBQW_train, matrixBMW_train]

mse = []
models = []
#normalize train

list_of_Norm_Matrix = [normalize_train(x) for x in train_matrix]

#get models of test

for i in range(0, 4):
    models.append(train_model(train_matrix[i],y_trainNorm[i]))

#get all MSEs
for x in range(0, 4):
    mse.append(error(test_matrixNorm[x], y_testNorm[x], models[x]))

print(mse)
min_val = min(mse)
min_index = mse.index(min_val)
print(min_index)
print("minimum is with BMW")

#creating feature matrices for all our models

################################## Question 2 #######################################

Feature = np.column_stack((dataset_1['High Temp'], dataset_1['Low Temp'], dataset_1['Precipitation']))
#reshaping the matrices with only one feature to make it 2 Dimensions
FeaturePP = np.column_stack((dataset_1['Precipitation']))
FeaturePP = FeaturePP.reshape((214,1))

FeatureHighTemp = np.column_stack((dataset_1['High Temp']))
FeatureHighTemp = FeatureHighTemp.reshape((214,1))

FeatureLowTemp = np.column_stack((dataset_1['Low Temp']))
FeatureLowTemp = FeatureLowTemp.reshape((214,1))

#Splitting the feature matrices into train and test data sets
matrix_train, matrix_test, yMatrix_train, yMatrix_test = train_test_split(Feature,dataset_1['Total'],test_size = 0.2, shuffle = False)
matrix_train1, matrix_test1, yMatrix_train1, yMatrix_test1 = train_test_split(FeatureHighTemp,dataset_1['Total'], test_size = 0.2, shuffle = False)
matrix_train2, matrix_test2, yMatrix_train2, yMatrix_test2 = train_test_split(FeatureLowTemp,dataset_1['Total'], test_size = 0.2, shuffle = False)
matrix_train3, matrix_test3, yMatrix_train3, yMatrix_test3 = train_test_split(FeaturePP,dataset_1['Total'], test_size = 0.2, shuffle = False)

#Normalizing all sets
y_testNorm = normalize_train(yMatrix_test)
y_trainNorm = normalize_train(yMatrix_train)
test_MatrixNorm =  normalize_train(matrix_test)
train_MatrixNorm = normalize_train(matrix_train)

#Ridge Regression Model
Model = train_model(train_MatrixNorm,y_trainNorm)
#Rsquared for ridge regression
Rsquared = rSquared(test_MatrixNorm, y_trainNorm, Model, y_testNorm)
#Linear Regression Model and Lasso Model.
Model11 = trainL_model(train_MatrixNorm,y_trainNorm)
Model12 = trainLasso_model(train_MatrixNorm,y_trainNorm)
#Rsquared for the above.
Rsquared11 = rSquared(test_MatrixNorm, y_trainNorm, Model11, y_testNorm)
Rsquared12 = rSquared(test_MatrixNorm, y_trainNorm, Model12, y_testNorm)

#Normalizing other variables
y_testNorm1 = normalize_train(yMatrix_test1)
y_trainNorm1 = normalize_train(yMatrix_train1)
test_MatrixNorm1 =  normalize_train(matrix_test1)
train_MatrixNorm1 = normalize_train(matrix_train1)

#Rsquared and model for High Temp
Model1 = train_model(train_MatrixNorm1,y_trainNorm1)
Rsquared1 = rSquared(test_MatrixNorm1, y_trainNorm1, Model1, y_testNorm1)

#Normalize Low Temp Variables
y_testNorm2 = normalize_train(yMatrix_test2)
y_trainNorm2 = normalize_train(yMatrix_train2)
test_MatrixNorm2 =  normalize_train(matrix_test2)
train_MatrixNorm2 = normalize_train(matrix_train2)
#Model and Rquared for Low Temp Variables
Model2 = train_model(train_MatrixNorm2,y_trainNorm2)
Rsquared2 = rSquared(test_MatrixNorm2, y_trainNorm2, Model2, y_testNorm2)

#Normalize Precipitation
y_testNorm3 = normalize_train(yMatrix_test3)
y_trainNorm3 = normalize_train(yMatrix_train3)
test_MatrixNorm3 =  normalize_train(matrix_test3)
train_MatrixNorm3 = normalize_train(matrix_train3)
#Model and Rsquared for precipitation
Model3 = train_model(train_MatrixNorm3,y_trainNorm3)
Rsquared3 = rSquared(test_MatrixNorm3, y_trainNorm3, Model3, y_testNorm3)

#Printing Rsquared for 3 types of models for all three features
print(Rsquared)
print(Rsquared11)
print(Rsquared12)
print()
#Printing Rsquared for the three features using ridge regression individually in the order - high temp, low temp, precipitation.m
print(Rsquared1)
print(Rsquared2)
print(Rsquared3)
#print()
#print(PPrSquared)
#################################### Question 3 ######################################################
dataset_1['Precipitation']  = pandas.to_numeric(dataset_1['Precipitation'].replace(',','', regex=True))

Precip = []
for i in dataset_1['Precipitation']: #Make it binary
    if(i > 0):
        Precip.append(1)
    else:
        Precip.append(0)

#print(Precip)
Precip = np.array(Precip)
Precip = Precip.reshape(214,1)
rng = np.random.default_rng()
num = rng.integers(low = 0, size = 1, high = 10000)
print(num)
matrix_train, matrix_test,y_train, y_test = train_test_split(total,Precip , test_size=0.3, random_state= 5373) #get parameters


model2 = trainLo_model(matrix_train, y_train)
pred = pred(matrix_test, model2) #get prediction model with logistic regression
#print("here is y_test")
print(max(dataset_1['Total']))

plt.figure(1)
plt.clf()
plt.scatter(dataset_1['Total'].ravel(), Precip , color="black", zorder=20) # plot creator
X_test = np.linspace(0, 30000)

loss = expit(X_test * model2.coef_ + model2.intercept_).ravel()
plt.plot(X_test, loss, color="red", linewidth=3)

plt.ylabel("Likelyness of Rain")
plt.xlabel("Amount of Bikers Outside")

plt.show()


cnf_matrix = metrics.confusion_matrix(y_test,pred) # confusion matrix creation
class_names = [0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pandas.DataFrame(cnf_matrix), annot=True, fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix for Total', y=1.2)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
















