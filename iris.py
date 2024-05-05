import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

def Soft_Max(z):
        row, col = z.shape
        z1 = np.zeros(shape = (row, col))
        for i in range (row):
            for j in range(col):
                z1[i][j] = np.exp(z[i][j]) / np.sum(np.exp(z[i]))
        return z1

def oneHotVector(y):
        y = np.array([y])
        rows, n_samples = np.shape(y)
        classes = np.unique(y)
        n_classes = len(classes)
        ohv = np.zeros(shape = (n_samples, n_classes))
        for j in range (rows):
            for m in range (n_samples):
                index = y[j][m]
                ohv[m][index] = 1
        return ohv

class SoftmaxRegression():

    def __init__(self, lr = 0.01, iterations = 1000):
        self.lr = lr
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        samples, features = X.shape
        classes = np.unique(y)
        self.weights = np.zeros(shape = (len(classes), features))
        self.bias = np.array([np.zeros(len(classes))])
        ohv = oneHotVector(y)

        for i in range (self.iterations):
            linear_model = np.dot(X, self.weights.T) + self.bias
            prob = Soft_Max(linear_model)
            dw = (1/samples)*np.dot((prob - ohv).T, X)
            db = (prob - ohv).T
            db1 = np.zeros(shape = (1, len(classes)))
            for i in range(db.shape[0]):
                db1[0][i] =(1/samples) * np.sum(db[i])
            self.weights -= self.lr * dw
            self.bias -= self.lr * db1

    def predict(self, someX):
        model = np.dot(someX, self.weights.T) + self.bias
        probs = Soft_Max(model)
        preds = np.argmax(probs, axis = 1)
        return preds
    
    def accuracy(self, test, preds):
        count = 0.0
        for i in range(len(test)):
            if(test[i] == preds[i]):
                count += 1
        accuracy_percent = count / len(test)
        return accuracy_percent
    
model = SoftmaxRegression()

model.fit(X_train, y_train)
model.predict(X_test)
y_predicted = model.predict(X_test)

Accuracy = model.accuracy(y_test, y_predicted)
print(f"Accuracy: {Accuracy:.4f}")

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)

print("Confusion Matrix :-")
for i in range(len(cm)):
    print(cm[i])
