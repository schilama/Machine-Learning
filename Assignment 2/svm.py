from __future__ import division
import numpy as np


class SVM(object):
    """The mini-batch pegasos algorithm with a bias term.

    Arguments:
        C: regularization parameter (default: 1)
        iterations: number of training iterations (default: 500)
    """
    def __init__(self, C=1, iterations=800):
        self.C = C
        self.iterations = iterations
        

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                Training target. Each entry is either -1 or 1.
        """
        self.w = np.zeros(X.shape[1])
        self.b = 0
        epoch = 1
        alpha = 0.0001
        while (epoch <= self.iterations):
            batch_y = y
            batch_X = X
            subgrad = self.subgradient(batch_X,batch_y)
            subgrad_w = subgrad[0]
            subgrad_b = subgrad[1]
            self.w = self.w - alpha*subgrad_w
            self.b = self.b - alpha*subgrad_b
            #print self.b
            predicted_y = self.predict(batch_X)
            correct_predictions = 0
            for i in range(0,len(batch_y)):
                if predicted_y[i] == batch_y[i]:
                    correct_predictions = correct_predictions + 1
            accuracy = correct_predictions/len(batch_y)
            #print accuracy
            #print epoch
            epoch = epoch + 1
            
            
            
    def objective(self, X, y):
        """Compute the objective function for the SVM.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                Training target. Each entry is either -1 or 1.

        Returns:
            obj (float): value of the objective function evaluated on X and y.
        """
        objval = 0
        normsquare = np.linalg.norm(self.w,ord=2)**2
        for i in range(0,len(y)):
            if 1 - np.dot(y[i],(np.dot(self.w,X[i].T)+self.b)) > 0.0 :
                objval = objval + self.C*(1 - np.dot(y[i],(np.dot(self.w,X[i].T)+self.b))) 
        return objval + normsquare       
                
    def subgradient(self, X, y):
        """Compute the subgradient of the objective function.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                Training target. Each entry is either -1 or 1.

        Returns:
            subgrad_w (ndarray, shape = (n_features,)):
                subgradient of the objective function with respect to
                the coefficients of the linear model.
            subgrad_b (float):
                subgradient of the objective function with respect to
                the bias term.
        """
        subgrad_w = np.zeros(X.shape[1])
        subgrad_b = 0
        for i in range(0,len(y)):
            if 1 - np.dot(y[i],(np.dot(self.w,X[i].T)+self.b)) > 0.0 :
                subgrad_w = subgrad_w - self.C*y[i]*X[i] 
                subgrad_b = subgrad_b - self.C*y[i]
        subgrad_w = subgrad_w + 2*self.w
        return subgrad_w, subgrad_b

    def predict(self, X):
        """Predict class labels for samples in X.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)): test data

        Returns:
            y (ndarray, shape = (n_samples,):
                Predictions with values of -1 or 1.
        """
        raw_prediction = np.dot(self.w,X.T) + self.b
        predicted_y = []
        for item in raw_prediction:
            if item > 0:
                predicted_y.append(1)
            else:
                predicted_y.append(-1)
        return np.array(predicted_y)

    def get_model(self):
        """Get the parameters of the model.

        Returns:
            w (ndarray, shape = (n_features,)):
                coefficient of the linear model.
            b (float): bias term.
        """
        return self.w, self.b

    def set_model(self, w, b):
        """Set the parameters of the model.

        Arguments:
            w (ndarray, shape = (n_features,)):
                coefficient of the linear model.
            b (float): bias term.
        """
        self.w, self.b = w, b


def main():
    np.random.seed(0)

    train_X = np.load('../data/q1_train_X.npy')
    train_y = np.load('../data/q1_train_y.npy')
    test_X = np.load('../data/q1_test_X.npy')
    test_y = np.load('../data/q1_test_y.npy')

    cls = SVM()
    #print train_X.shape
    #print train_y.shape
    #print train_X.shape[1]
    """
    cls.w = np.zeros(train_X.shape[1])
    print cls.w.shape
    cls.b = 0
    
    #temp = cls.subgradient(train_X,train_y)[0]
    """
    """
    for item in temp:
        print item
    """
    cls.fit(train_X, train_y)
    objval = cls.objective(train_X,train_y)
    print "SVC objective on training data: %f" % objval
    #print cls.objective(train_X,train_y)
    predicted_y = cls.predict(train_X)
    correct_predictions = 0
    for i in range(0,len(train_y)):
        if predicted_y[i] == train_y[i]:
            correct_predictions = correct_predictions + 1
    accuracy = correct_predictions/len(train_y)
    classificationerror = 1 - accuracy
    print "SVC classification error on training data: %f" % classificationerror 
    #print classificationerror    
    
    from sklearn.linear_model import LogisticRegression
    logregobj = LogisticRegression()
    returnedobj = logregobj.fit(train_X,train_y)
    logregw = np.asarray(returnedobj.coef_[0])
    logregb = np.asarray(returnedobj.intercept_[0])
    
    expcoeff = - train_y*(np.dot(logregw,train_X.T) + logregb)
    first_term = np.log(1 + np.exp(expcoeff)).sum()
    second_term = 0.5*(np.linalg.norm(logregw,ord=2)**2)
    valueoflogregobj = first_term + second_term
    print "Logistic regression objective from scikitlearn params: %f" % valueoflogregobj
    
    SVCwithlogregparam = SVM()
    SVCwithlogregparam.w = logregw
    SVCwithlogregparam.b = logregb
    SVCobjlogregparamval = SVCwithlogregparam.objective(train_X,train_y)
    print "SVM objective from scikitlearn params: %f" % SVCobjlogregparamval
    
    predicted_y = returnedobj.predict(train_X)
    correct_predictions = 0
    for i in range(0,len(train_y)):
        if predicted_y[i] == train_y[i]:
            correct_predictions = correct_predictions + 1
    accuracy = correct_predictions/len(train_y)
    classificationerror = 1 - accuracy
    print "Classification error rate on training data: %f" % classificationerror         
    
    """
    predicted_y = SVCwithlogregparam.predict(train_X)
    correct_predictions = 0
    for i in range(0,len(train_y)):
        if predicted_y[i] == train_y[i]:
            correct_predictions = correct_predictions + 1
    accuracy = correct_predictions/len(train_y)
    classificationerror = 1 - accuracy
    print "SVM classification error with logistic regression optimal parameters on training data:%f" % classificationerror     
    #Same error rate as scikitlearn predict. SVM() uses the same function to predict.
    
    """
    #verifying performance on test data for SVM
    predicted_y = cls.predict(test_X)
    correct_predictions = 0
    for i in range(0,len(test_y)):
        if predicted_y[i] == test_y[i]:
            correct_predictions = correct_predictions + 1
    accuracy = correct_predictions/len(test_y)
    classificationerror = 1 - accuracy
    print "SVC classification error on test data: %f" % classificationerror   
    
    #verifying performance on test data for logistic regression
    predicted_y = returnedobj.predict(test_X)
    correct_predictions = 0
    for i in range(0,len(test_y)):
        if predicted_y[i] == test_y[i]:
            correct_predictions = correct_predictions + 1
    accuracy = correct_predictions/len(test_y)
    classificationerror = 1 - accuracy
    print "Logistic regression error on test data: %f" % classificationerror  
    
if __name__ == '__main__':
    main()
