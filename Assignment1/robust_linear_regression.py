from __future__ import division
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

class RobustLinearRegression(object):
    """Generalized robust linear regression.

    Arguments:
        delta (float): the cut-off point for switching to linear loss
        k (float): parameter controlling the order of the polynomial par of the loss 
    """

    def __init__(self, delta, k):
        self.delta = delta #cut-off point
        self.k = k #polynomial order parameter
        self.w = 0 #learned coefficients
        self.b = 0 #learned bias
          
    def fit(self, X, y):
        """Fit the model according to the given training data.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                Training output vector. Each entry is either -1 or 1.
        """
        wb = np.append(self.w,self.b)
        returnedlist = fmin_l_bfgs_b(self.objective,wb,self.objective_grad,args=(X,y),approx_grad=0, bounds=None, m=10, factr=10000000.0, pgtol=1e-05, epsilon=1e-08, iprint=-1, maxfun=15000, maxiter=15000)
        #print returnedlist
        self.set_params(returnedlist[0][:len(returnedlist[0]) - 1], returnedlist[0][len(returnedlist[0])-1])
        return self

    def predict(self, X):
        """Predict using the linear model.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)): test data

        Returns:
            y (ndarray, shape = (n_samples,): predicted values
        """
        return X.dot(self.w) + self.b


    def objective(self, wb, X, y):
        """Compute the loss function.

        Arguments:
            wb (ndarray, shape = (n_features + 1,)):
                concatenation of the coefficient and the bias parameters: wb = [w,b]
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                target values.

        Returns:
            loss (float):
                the objective function evaluated on w.
        """
        x = np.append(X,np.ones([len(X),1]),1)
        loss = 0
        for i in range(0,len(y)):
            if np.absolute(y[i] - np.dot(wb,x[i])) <= self.delta:
                loss = loss + (1/2*self.k)*np.power((y[i] - np.dot(wb,x[i])),2*self.k)
            else:
                loss = loss + np.power(self.delta,2*self.k-1) * ((np.absolute(y[i]-np.dot(wb,x[i])) - (self.delta*((2*self.k) - 1)/2*self.k)))
        return loss

    def objective_grad(self, wb, X, y):
        """Compute the derivative of the loss function.

        Arguments:
            wb (ndarray, shape = (n_features + 1,)):
                concatenation of the coefficient and the bias parameters: wb = [w,b]
            X (ndarray, shape = (n_samples, n_features )):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                target values.

        Returns:
            loss_grad (ndarray, shape = (n_features + 1,)):
                derivative of the objective function with respect to w.
        """
        x = np.append(X,np.ones([len(X),1]),1)
        #print wbxT.shape
        
        first_term = np.array(np.zeros(wb.shape))
        second_term = np.array(np.zeros(wb.shape))
        third_term = np.array(np.zeros(wb.shape))
        
        for i in range(0,len(y)):
            if np.absolute(y[i] - np.dot(wb,x[i])) <= self.delta:
                first_term = first_term + (np.power(y[i] - np.dot(wb,x[i]),2*self.k -1)* (-x[i].T))
            elif y[i] - np.dot(wb,x[i]) > 0:
                second_term = second_term + (np.power(self.delta,2*self.k -1)*(-x[i].T))
            elif  y[i] - np.dot(wb,x[i]) < 0:
                third_term = third_term + (np.power(self.delta,2*self.k -1)*(x[i].T))
        #print first_term
        #print second_term
        #print third_term
        return first_term + second_term + third_term

    def get_params(self):
        """Get learned parameters for the model. Assumed to be stored in 
           self.w, self.b.

        Returns: 
            A tuple (w,b) where w is the learned coefficients (ndarray)
            and b is the learned bias (float).
        """
        return (self.w, self.b)

    def set_params(self, w, b):
        """Set the parameters of the model. When called, this
           function sets the model parameters tha are used
           to make predictions. Assumes parameters are stored in 
           self.w, self.b.

        Arguments:
            w (ndarray, shape = (n_features,)): coefficient prior
            b (float): bias prior
        """
        self.w = w
        self.b = b
        

def main():

    np.random.seed(0)

    #Example code for loading data
    train_X = np.load('../Data/q3_train_X.npy')
    train_y = np.load('../Data/q3_train_y.npy')
    
    delta = 1
    k = 1
    
    #print train_X.shape
    #print train_y.shape
    
    myobject = RobustLinearRegression(delta,k)
    
    wb = np.append(myobject.w,myobject.b)
    #print wb
    gradient = myobject.objective_grad(wb,train_X,train_y)
    #print gradient
    
    objloss = myobject.objective(wb,train_X,train_y)
    #print objloss
    
    optima = myobject.fit(train_X,train_y)
    #print optima.w
    #print optima.b
    
    predicted_y = myobject.predict(train_X)
    #print predicted_y.shape
    #print train_y.shape
    #computing mean square error from robust regression
    
    sumofsquares = 0
    for i in range(0,len(predicted_y)):
        sumofsquares = sumofsquares + np.power((predicted_y[i] - train_y[i]),2)
    
    robustMSE = sumofsquares/len(train_y)
    
    from sklearn import linear_model
    regrobj = linear_model.LinearRegression()
    regrobj.fit(train_X,train_y)
    linearmodelprediction_y = regrobj.predict(train_X)
    sumofsquares = 0
    for i in range(0,len(linearmodelprediction_y)):
        sumofsquares = sumofsquares + np.power((linearmodelprediction_y[i] - train_y[i]),2)
    linearMSE = sumofsquares/len(train_y)
    
    print robustMSE
    print linearMSE
    
    import matplotlib.pyplot as plt
    plt.scatter(train_X,train_y)
    plt.plot(train_X,predicted_y,color='k')
    plt.plot(train_X,linearmodelprediction_y,color='r')
    plt.title("Scatter plot for question 3(d)")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    
if __name__ == '__main__':
    main()
