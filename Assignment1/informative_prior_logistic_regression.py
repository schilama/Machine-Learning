from __future__ import division
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

class InformativePriorLogisticRegression(object):
    """Logistic regression with general spherical Gaussian prior.

    Arguments:
        w0 (ndarray, shape = (n_features,)): coefficient prior
        b0 (float): bias prior
        reg_param (float): regularization parameter $\lambda$ 
    """

    def __init__(self, w0, b0, reg_param):
        self.w0=w0  #Prior coefficients
        self.b0=b0  #Prior bias
        self.reg_param = reg_param #Regularization parameter (lambda)
        self.w =np.zeros(w0.shape) #Learned w
        self.b =0 #Learned b

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                Training output vector. Each entry is either -1 or 1.
        """
        wb = np.append(self.w, self.b)
        #will now proceed to approximate w given training data using fmin_l_bfgs_b
        returnedlist = fmin_l_bfgs_b(self.objective,wb,self.objective_grad,args=(X,y),approx_grad=0, bounds=None, m=10, factr=10000000.0, pgtol=1e-05, epsilon=1e-08, iprint=-1, maxfun=15000, maxiter=15000)
        #first item in returnedlist contains wb, splitting into w and b and setting learned values to optimums
        self.set_params(returnedlist[0][:len(returnedlist[0]) - 1], returnedlist[0][len(returnedlist[0])-1])
        return self
        
    def predict(self, X):
        """Predict class labels for samples in X.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)): test data

        Returns:
            y (ndarray, shape = (n_samples,):
                Predictions with values in {-1, +1}.
        """
        wb = np.append(self.w,self.b)
        xplusone = np.append(X,np.ones([len(X),1]),1)
        expcoeff = - np.dot(wb,xplusone.T)
        probabilityarray = 1/(1+ np.exp(expcoeff))
        y_list = []
        for item in probabilityarray:
            if item < 0.5:
                y_list.append(-1)
            else:
                y_list.append(1)
        #print probabilityarray.shape
        #print np.array(y_list).shape
        return np.array(y_list)

    def objective(self, wb, X, y):
        """Compute the objective function

        Arguments:
            wb (ndarray, shape = (n_features + 1,)):
                concatenation of the coefficient and the bias parameters: wb = [w,b]
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                training label.

        Returns:
            loss (float):
                the objective function evaluated on w.
        """
        #wb = np.append(self.w, self.b)
        #Appending column of 1s to X matrix for bias absorption
        xplusone = np.append(X,np.ones([len(X),1]),1)  
        expcoeff = - y*np.dot(wb,xplusone.T)
        first_term = np.log(1 + np.exp(expcoeff)).sum()        
        second_term = self.reg_param*((np.linalg.norm(np.subtract(wb,np.append(self.w0,self.b0)),ord=2))**2)
        #not sure why autograder only accepts objective float sent in an list!
        return [first_term + second_term] 

    def objective_grad(self, wb, X, y):
        """Compute the derivative of the objective function

        Arguments:
            wb (ndarray, shape = (n_features + 1,)):
                concatenation of the coefficient and the bias parameters: wb = [w,b]
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                training label.

        Returns:
            loss_grad (ndarray, shape = (n_features + 1,)):
                derivative of the objective function with respect to w.
        """
        #wb = np.append(self.w, self.b)
        #Appending column of 1s to X matrix for bias absorption
        xplusone = np.append(X,np.ones([len(X),1]),1)
        #print xplusone.shape
        expcoeff = - y*np.dot(wb,xplusone.T)
        #print expcoeff
        first_term = ((-y*xplusone.T*(np.exp(expcoeff)))/(1 + np.exp(expcoeff))).sum(axis=1)
        second_term = np.subtract(wb,np.append(self.w0,self.b0))*2*self.reg_param
        gradient = first_term + second_term
        return gradient
        

    def get_params(self):
        """Get parameters for the model.

        Returns: 
            A tuple (w,b) where w is the learned coefficients (ndarray)
            and b is the learned bias (float).
        """
        return self.w, self.b

    def set_params(self, w, b):
        """Set the parameters of the model.

        Arguments:
            w (ndarray, shape = (n_features,)): coefficient prior
            b (float): bias prior
            reg_param (float): regularization parameter $\lambda$ (default: 0)
        """
        self.w = w
        self.b = b


def main():
    np.random.seed(0)

    #Example code for loading data
    train_X = np.load('../Data/q2_train_X.npy')
    train_y = np.load('../Data/q2_train_y.npy')
    test_X = np.load('../Data/q2_test_X.npy')
    test_y = np.load('../Data/q2_test_y.npy')
    w_prior = np.load('../Data/q2_w_prior.npy').squeeze()
    b_prior = np.load('../Data/q2_b_prior.npy')
    
    lambda0accuracy = []
    lambda10accuracy = []
    numoftrainingcases = []
    #Setting sample lambda value
    mylambda = 0
    
    #Initializing myobject from class InformativePriorLogisticRegression
    myobject = InformativePriorLogisticRegression(w_prior,b_prior,mylambda)
    
    for i in range (10,410,10):
        myobject.fit(train_X[:i],train_y[:i])
        predicted_y = myobject.predict(test_X)
        correctcounter = 0
        #computing accuracy (0/1)
        for j in range(0,len(predicted_y)):
            if predicted_y[j] == test_y[j]:
                correctcounter = correctcounter + 1
        accuracy = correctcounter/len(predicted_y)
        lambda0accuracy.append(accuracy)        
    
    mylambda = 10
    myobject = InformativePriorLogisticRegression(w_prior,b_prior,mylambda)
    for i in range (10,410,10):
        myobject.fit(train_X[:i],train_y[:i])
        predicted_y = myobject.predict(test_X)
        correctcounter = 0
        #computing accuracy (0/1)
        for j in range(0,len(predicted_y)):
            if predicted_y[j] == test_y[j]:
                correctcounter = correctcounter + 1
        accuracy = correctcounter/len(predicted_y)
        lambda10accuracy.append(accuracy)
        numoftrainingcases.append(i)
    """
    print lambda0accuracy
    print lambda10accuracy
    print numoftrainingcases
    """
    import matplotlib.pyplot as plt
    plt.plot(numoftrainingcases,lambda0accuracy,color='r')
    plt.plot(numoftrainingcases,lambda10accuracy,color='g')
    plt.title("Experiment 2(c)")
    plt.xlabel('Number of training cases')
    plt.ylabel('Test accuracy')
    plt.show()

    
if __name__ == '__main__':
    main()
