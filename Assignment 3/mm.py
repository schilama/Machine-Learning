from __future__ import division
from __future__ import print_function
import numpy as np
from scipy.misc import logsumexp
from scipy.misc import factorial

class MixtureModel(object):
    """Mixture Model
        Arguments:
            labels_in_category (dic {category (string): labels (list (string))}):
                mapping from category name to list of labels within category
            n_components (int):
                number of mixture components
            n_iter (int):
                number of iterations for EM algorithm
    """
    def __init__(self, labels_in_category, n_components=10, n_iter=50):
        """Initializes parameters
            Note that the nested dictionary parameter format is for avoiding ambiguity during grading.
            You should convert these paramters to a format that is convenient for computation.
        """
        self.n_components = n_components
        self.n_iter=n_iter
        self.labels_in_category = labels_in_category

        # initialize parameters for mixture component
        self.component = {}
        for z in range(self.n_components):
            self.component[z] = 1 / self.n_components

        # initialize Gaussian means
        self.gaussian_mean = {}
        for key in ['age', 'hr']:
            self.gaussian_mean[key] = {}
            for z in range(self.n_components):
                self.gaussian_mean[key][z] = np.random.rand()

        # intialize Gaussain covariance with identity matrix
        self.gaussian_cov = {}
        for key1 in ['age', 'hr']:
            self.gaussian_cov[key1] = {}
            for key2 in ['age', 'hr']:
                self.gaussian_cov[key1][key2] = {}
                for z in range(self.n_components):
                    value = 1. if key1 == key2 else 0.
                    self.gaussian_cov[key1][key2][z] = value

        # initialize Poisson paramters
        self.poisson = {}
        self.poisson['edu-num'] = {}
        for z in range(self.n_components):
            self.poisson['edu-num'][z] = np.random.rand()

        # initialize Bernoulli paramters
        self.bernoulli = {}
        for key in ['income', 'sex']:
            self.bernoulli[key] = {}
            for label in labels_in_category[key]:
                self.bernoulli[key][label] = {}
                for z in range(self.n_components):
                    self.bernoulli[key][label][z] = 0.5

        # initialize multinomial paramters
        self.multinomial = {}
        for key in ['workclass', 'marital', 'occup', 'relation', 'country']:
            n_labels = len(labels_in_category[key])
            self.multinomial[key] = {}
            for label in labels_in_category[key]:
                self.multinomial[key][label] = {}
                for z in range(self.n_components):
                    self.multinomial[key][label][z] = 1.0/n_labels
                


    def fit(self, X):
        """Fit the mixture model according to the given data using EM algorithm.

        Arguments:
            X (structured_array, shape = (n_samples, 1)):
                Training input data where each row is a sample stored as a tuple of data entries.
                    The categories within X can be obtained through X.dtype.names
                    The data within each category can be accessed through X[category]
                    See https://docs.scipy.org/doc/numpy-1.13.0/user/basics.rec.html for more detail

            You should create a function that encodes and groups the data into meaningful chunks for computing.
            You are allowed to use sklearn's label encoder to encode the data.
            Try to avoid more then two for loops within the E-step and M-step.
        """        
        #print (X.dtype.names)
        
        N = X.shape[0]
        for i in range(1):
            print ('iter:',i)
            #E-step
            #Building phizn which is a matrix of n samples as rows and z values as columns
            tphil = []
            for zdim in range(self.n_components):
                tphil.append(self.posterior(X,np.full(X.shape[0],zdim)))
                #print (tphil[zdim].shape)
            #phizn is a matrix with n samples as rows,z values as columns
            #print (np.array(tphil[0]).shape)
            phizn = np.stack(tphil,axis=1)
            #print (phizn)
            phiznsumn = np.sum(phizn,axis=0)
            #print ((1/N)*phiznsumn)
           
            #M-step
            bernincf = np.zeros(self.n_components)
            bernincsum = np.zeros(self.n_components)
            bernsexf = np.zeros(self.n_components)
            bernsexsum = np.zeros(self.n_components)
            poissonf = np.zeros(self.n_components)
            poissonsum = np.zeros(self.n_components)
            #print (bernincf.shape)
            for zdim in range(self.n_components):
                #M step for model parameters
                #print (phiznsumz[zdim])
                self.component[zdim]=(1/N)*phiznsumn[zdim]
                bsum=0
                for ndim in range(N):
                    #M step for Bernoulli Variables
                    #print (phizn[ndim][zdim])
                    if(X['income'][ndim]=='<=50K'):
                        bernincsum[zdim] = bernincsum[zdim] + (phizn[ndim][zdim])
                    if(X['sex'][ndim]=='Female'):    
                        bernsexsum[zdim] = bernsexsum[zdim] + (phizn[ndim][zdim])
                    #M step for Poisson Variable    
                    poissonsum[zdim] = poissonsum[zdim] + X['edu-num'][ndim]
                bernincf[zdim] = (1 + 100*bernincsum[zdim])/(2 + 100*phiznsumn[zdim])
                bernsexf[zdim] = (1 + 100*bernsexsum[zdim])/(2 + 100*phiznsumn[zdim])
                poissonf[zdim] = (1 + 100*poissonsum[zdim])/(100 + 100*phiznsumn[zdim]) 
                #print (np.array(bernincf[zdim]).shape,np.array(1 + 100*bernincsum[zdim]).shape,np.array(2 + 100*phiznsumn[zdim]).shape)
            #print (self.component)
            #print (bernincf,bernsexf)
            #print (poissonf)
            
            
              
                    
                    
              

    def log_marginal_likelihood(self, X):
        """Calculate log marginal likehood.

        Arguments:
            X (structured_array, shape = (n_samples, 1)):
                Input data where each row is a sample stored as a tuple of data entries.
        Returns:
            log_marginal_likelihood (float)
        """
        total = 0.0
        # initialize my gcov matrix dict with respect to z
        my_gcovz = {}   
        keys = ['age', 'hr']
        for z in range(self.n_components):
            cov = np.array([[self.gaussian_cov['age']['age'][z],self.gaussian_cov['age']['hr'][z]],[self.gaussian_cov['hr']['age'][z],self.gaussian_cov['hr']['hr'][z]]])
            my_gcovz[z] = cov
        
        # initialize my mu matrix dict with respect to z
        my_muz = {}
        for z in range(self.n_components):
            my_muz[z] = np.array([self.gaussian_mean['age'][z], self.gaussian_mean['hr'][z]]).astype(np.float_)
        
        #print (my_muz)
        
        for x in X:
            #x1,x2,x3,x4,x5,x6,x7,x8,x9,x10 = x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]
            xg,xp,xb,xc= np.array([x[0],x[1]]).astype(np.float), x[2], [x[3],x[4]], [x[5],x[6],x[7],x[8],x[9]]
            #print (x1,x2,x3,x4,x5,x6,x7,x8,x9,x10)
            #look at page for order of terms
            fir,sec,thi,fou,fiv,six,sev,eig,nin,ten,ele,twe = 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
            firz,secz,thiz,fouz,fivz,sixz,sevz,eigz,ninz,tenz,elez,twez = [],[],[],[],[],[],[],[],[],[],[],[]
            for z in range(self.n_components):
                #Gaussian exp term
                adj_xg = np.array(xg - my_muz[z])[:,np.newaxis]
                firz.append((-1/2)*np.transpose(adj_xg).dot(np.linalg.inv(my_gcovz[z])).dot((adj_xg))) 
                #print (np.array(firz).shape)
                #Gaussian log term
                thiz.append((-1/2)*np.log(np.linalg.det(2*np.pi*my_gcovz[z])))
                #Poisson exp term
                secz.append(-self.poisson['edu-num'][z])
                #Poisson log term
                elez.append(xp*np.log(self.poisson['edu-num'][z])-np.log(factorial(xp)))
                #First bernoulli dist (income) log term
                if xb[0] == '<=50K':
                    fouz.append(np.log(self.bernoulli['income']['<=50K'][z]))                    
                else:
                    fouz.append(np.log(self.bernoulli['income']['>50K'][z]))
                #Second bernoulli dist (sex) log term
                if xb[1] == 'Female':
                    fivz.append(np.log(self.bernoulli['sex']['Female'][z]))                    
                else:
                    fivz.append(np.log(self.bernoulli['sex']['Male'][z]))    
                #Multinoulli workclass
                sixz.append(np.log(self.multinomial['workclass'][xc[0].decode("utf-8")][z]))
                #Multinoulli marital
                sevz.append(np.log(self.multinomial['marital'][xc[1].decode("utf-8")][z]))
                #Multinoulli occup
                eigz.append(np.log(self.multinomial['occup'][xc[2].decode("utf-8")][z]))
                #Multinoulli relation
                ninz.append(np.log(self.multinomial['relation'][xc[3].decode("utf-8")][z]))
                #Multinoulli country
                tenz.append(np.log(self.multinomial['country'][xc[4].decode("utf-8")][z]))
                #theta zM 
                twez.append(np.log(self.component[z]))

            #print (np.array(firz).squeeze(),np.array(secz).shape,np.array(thiz).shape,np.array(fouz).shape,np.array(fivz).shape,np.array(sixz).shape,np.array(sevz).shape,np.array(eigz).shape,np.array(ninz).shape,np.array(tenz).shape,np.array(elez).shape)
            myarr = np.array(firz).squeeze()+np.array(secz)+np.array(thiz)+np.array(fouz)+np.array(fivz)+np.array(sixz)+np.array(sevz)+np.array(eigz)+np.array(ninz)+np.array(tenz)+np.array(elez)+np.array(twez)
            
            total = total + logsumexp(myarr)
            #print (fir,sec,thi,fou,fiv,six,sev,eig,nin,ten,ele)
            #total = total + fir+sec+thi+fou+fiv+six+sev+eig+nin+ten+ele
        #print ("Log marginal likelihood:",total)
        return total

    def posterior(self, X, z):
        """Calculate poseterior probability.

        Arguments:
            X (structured_array, shape = (n_samples, 1)):
                Input data where each row is a sample stored as a tuple of data entries.
            z (ndarray, shape = (n_samples, 1)):
                Mixture component number for each sample, z \in {0,1,...,n_components-1}
        Returns:
            poseterior probability (ndarray, shape = (n_samples,))
        """
        exparg = self.log_posterior(X,z) 
        total = np.exp(exparg)
        #print ("Posterior:",total," Exparg=self.logposterior",exparg)
        #print (total.shape)
        return total
    
    #def sv_log_posterior(self,X,z):

    def log_posterior(self, X, z):
        """Calculate log poseterior probability.

        Arguments:
            X (structured_array, shape = (n_samples, 1)):
                Input data where each row is a sample stored as a tuple of data entries.
            z (ndarray, shape = (n_samples, 1)):
                Mixture component number for each sample, z \in {0,1,...,n_components-1}
        Returns:
            log poseterior probability (ndarray, shape = (n_samples,))
        """
        paramz = z
        # initialize my gcov matrix dict with respect to z
        my_gcovz = {}   
        keys = ['age', 'hr']
        for z in range(self.n_components):
            cov = np.array([[self.gaussian_cov['age']['age'][z],self.gaussian_cov['age']['hr'][z]],[self.gaussian_cov['hr']['age'][z],self.gaussian_cov['hr']['hr'][z]]])
            my_gcovz[z] = cov
        
        # initialize my mu matrix dict with respect to z
        my_muz = {}
        for z in range(self.n_components):
            my_muz[z] = np.array([self.gaussian_mean['age'][z], self.gaussian_mean['hr'][z]]).astype(np.float_)
        
        lpparr = []
        count = 0
        
        for x in X:
            total = 0.0
            z = paramz[count]
            xg,xp,xb,xc= np.array([x[0],x[1]]).astype(np.float), x[2], [x[3],x[4]], [x[5],x[6],x[7],x[8],x[9]]
            #look at page for order of terms
            fir,sec,thi,fou,fiv,six,sev,eig,nin,ten,ele,twe = 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
            firz,secz,thiz,fouz,fivz,sixz,sevz,eigz,ninz,tenz,elez,twez = [],[],[],[],[],[],[],[],[],[],[],[]            
            #Gaussian exp term
            adj_xg = np.array(xg - my_muz[z])[:,np.newaxis]
            fir = (-1/2)*np.transpose(adj_xg).dot(np.linalg.inv(my_gcovz[z])).dot((adj_xg)).squeeze()
            #Gaussian log term
            thi = (-1/2)*np.log(np.linalg.det(2*np.pi*my_gcovz[z]))
            #Poisson exp term
            sec = -self.poisson['edu-num'][z]
            #Poisson log term
            ele = xp*np.log(self.poisson['edu-num'][z])-np.log(factorial(xp))
            #First bernoulli dist (income) log term
            if xb[0] == '<=50K':
                fou = np.log(self.bernoulli['income']['<=50K'][z])                
            else:
                fou = np.log(self.bernoulli['income']['>50K'][z])
            #Second bernoulli dist (sex) log term
            if xb[1] == 'Female':
                fiv = np.log(self.bernoulli['sex']['Female'][z])                   
            else:
                fiv = np.log(self.bernoulli['sex']['Male'][z])
            #Multinoulli workclass
            six = np.log(self.multinomial['workclass'][xc[0].decode("utf-8")][z])
            #Multinoulli marital
            sev = np.log(self.multinomial['marital'][xc[1].decode("utf-8")][z])
            #Multinoulli occup
            eig = np.log(self.multinomial['occup'][xc[2].decode("utf-8")][z])
            #Multinoulli relation
            nin = np.log(self.multinomial['relation'][xc[3].decode("utf-8")][z])
            #Multinoulli country
            ten = np.log(self.multinomial['country'][xc[4].decode("utf-8")][z])
            #theta zM 
            twe = np.log(self.component[z])
            myarr = fir+sec+thi+fou+fiv+six+sev+eig+nin+ten+ele+twe
            total = myarr - self.log_marginal_likelihood([x])
            lpparr.append(total)
            count = count + 1
        finalarr = np.array(lpparr)
        #print(finalarr.shape)
        return finalarr


    def predict(self, X):
        """Predict missing information for samples in X.
            Predict and fill in the missing entries.

        Arguments:
            X (structured_array, shape = (n_samples, 1)):
                Input data where each row is a sample stored as a tuple of data entries.
                Each sample has one missing entry in either category 'age' or 'hr'. (hr is working hour).
        Returns:
            X_filled (structured_array, shape = (n_samples, 1)):
                X with missing data filled
        """

        return X

    def get_model_params(self):
        """Get the parameters of the model.
            See set_model_params for more information on parameter types.

        Returns:
            component (dict, {component_id (int): component probability (float)}):
                mixture component parameters
            gaussian_mean (dict, {category (string): (dict, {component_id (int): mean (float)})}):
                Gaussian means
            gaussian_cov (dict, {category (string): (dict, {category (string): (dict, {component_id (int): covariance (float)})})})
                Gaussian covariances
            poisson (dict, {component_id (int): lambda (float)})
                Poisson parameters
            bernoulli (dict, {category (string): (dict, {label (string): (dict, {component_id (int): probability of label (float)})})})
                Bernoulli parameters
            multinomial (dict, {category (string): (dict, {label (string): (dict, {component_id (int): probability of label (float)})})})
                multinomial parameters
        """

        return self.component, self.gaussian_mean, self.gaussian_cov, self.poisson, self.bernoulli, self.multinomial

    def set_model_params(self, component, gaussian_mean, gaussian_cov, poisson, bernoulli, multinomial):
        """Set the parameters of the model.
            Each argument is a nested dictionary. See intialization to understand more on the structure.
            e.g, the gaussian mean of the "age" category of the zth component is stored in gaussian_mean['age'][z].
            The gaussian covariance sigma_{age,hr} of the zth component is stored in gaussian_cov['age']['hr'][z].
            The multinomial paramter for "Tech-support" label in the "occup" category of the zth component is stored in multinomial['occup']['Tech-support'][z].
            Labels for each categories can be accesed through labels_in_category.

        Arguments:
            component (dict, {component_id (int): component probability (float)}):
                mixture component parameters
            gaussian_mean (dict, {category (string): (dict, {component_id (int): mean (float)})}):
                Gaussian means
            gaussian_cov (dict, {category (string): (dict, {category (string): (dict, {component_id (int): covariance (float)})})})
                Gaussian covariances
            poisson (dict, {component_id (int): lambda (float)})
                Poisson parameters
            bernoulli (dict, {category (string): (dict, {label (string): (dict, {component_id (int): probability of label (float)})})})
                Bernoulli parameters
            multinomial (dict, {category (string): (dict, {label (string): (dict, {component_id (int): probability of label (float)})})})
                multinomial parameters
        """

        self.component = component
        self.gaussian_mean = gaussian_mean
        self.gaussian_cov = gaussian_cov
        self.poisson = poisson
        self.bernoulli = bernoulli
        self.multinomial = multinomial

def main():
    np.random.seed(0)

    train_X = np.load('../data/q1_train_X.npy')
    test_X = np.load('../data/q1_test_X.npy')
    labels_in_category = np.load('../data/labels_in_category.npy').item()
    #print (train_X.shape)
    #print (labels_in_category)    
    mm = MixtureModel(labels_in_category, n_components=3, n_iter=50)
    #lml = mm.log_marginal_likelihood(train_X)
    #lp = mm.log_posterior(train_X,np.ones(train_X.shape[0]))
    #p = mm.posterior(train_X,np.ones(train_X.shape[0]))
    #print("LML:",lml," LP:",lp," P:",p)
    mm.fit(train_X[:5])
    
    prediction = mm.predict(test_X)
    np.save('q1_prediction', prediction)
    print("prediction file saved to q1_prediction.npy")
    

if __name__ == '__main__':
    main()
