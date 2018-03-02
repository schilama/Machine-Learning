from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import matplotlib.lines as mlines

def rejection_sampler(S,X,Y,mu,sigma):
    #Inputs S = number of samples (int), X,Y = training data (numpy arrays)
    #Mu = mean vector (array), sigma = covariance matrix (2D array)
    #Output = array of S samples from P(theta|D) parameter posterior
    thetars = []
    logregobj = LogisticRegression(tol=1e-2, C=1e+33)
    returnedobj = logregobj.fit(X,Y)
    wmle = np.asarray(returnedobj.coef_[0])
    bmle = np.asarray(returnedobj.intercept_[0])
    pmlesum = 0
    for i in range(len(X)):
        expcoeff = np.exp(-(wmle.dot(X[i].T)+bmle))
        if Y[i] == 1:
            pmlesum = pmlesum - np.log(1 + expcoeff)
        else:
            pmlesum = pmlesum + np.log(expcoeff/(1 + expcoeff))
    pmle = np.exp(pmlesum)
    for s in range(S):
        accept = False
        while(accept == False):
            thetas = np.random.multivariate_normal(mu,sigma)
            ws = thetas[:2]
            bs = thetas[2]
            u = np.random.uniform()
            pssum = 0
            for i in range(len(X)):
                expcoeff = np.exp(-(ws.dot(X[i].T)+bs))
                if Y[i] == 1:
                    pssum = pssum - np.log(1 + expcoeff)
                else:
                    pssum = pssum + np.log(expcoeff/(1 + expcoeff))
            ps = np.exp(pssum)
            if (ps/pmle >= u):
                accept = True
                thetars.append(thetas)
    #print thetars        
    return thetars
    
def plot_boundary(thetars):
    #Input = array thetars = set of theta values (can be a single theta value)
    #Each theta is a vector containing w1,w2,b
    #Output = Plot of the decision boundaries for each theta
    axes = plt.gca()
    axes.set_xlim([-1,1])
    axes.set_ylim([-1,1])
    x1 = np.arange(-1,1.1,0.1)
    for i in range(len(thetars)):
        w1 = thetars[i][0]
        w2 = thetars[i][1]
        b = thetars[i][2]
        x2 = (-w1/w2)*x1 - b/w2
        plt.plot(x1,x2,alpha=0.4,color='b') 
    #plt.title('Plot Of Decision Boundary')
    plt.xlabel('X1')
    plt.ylabel('X2')
    blue_line = mlines.Line2D([], [], color='blue',label='Decision Boundary',alpha=0.4)
    plt.legend(handles=[blue_line],loc=2)
    plt.show()

def predictive_distribution(S,X,Y,theta):
    #Inputs S = Number of samples in theta (int)
    #X = single data point (array [x1,x2]), Y = class label for the data point
    #Theta = Set of theta values (array), each theta is a vector containing [w1,w2,b]
    #Output = P(Y=y|X=x,D,mu,sigma)
    psum = 0
    for s in range(S):
        w = np.array(theta[s][:2])
        b = theta[s][2]
        expcoeff = np.exp(-(w.dot(X.T)+b))
        if Y == 1:
            psum = psum + (1/(1 + expcoeff))
        else:
            psum = psum + (expcoeff/(1 + expcoeff))
    return psum/S
    
def main():
    #Load the data
    data = np.load("../data/data.npz")
    Xte = data["Xte"] #Test feature vectors
    Yte = data["Yte"] #Test labels
    Xtr = data["Xtr"] #Train feature vectors
    Ytr = data["Ytr"] #Train labels
    """
    thetarsarr = []
    i = 10
    #Add your code here
    for i in range(10,60,10):
        thetars = rejection_sampler(100, Xtr[:i], Ytr[:i], mu=np.array([0,0,0]), sigma=np.array([[100,0,0],[0,100,0],[0,0,100]]))
        thetarsarr.append(thetars)
    np.save('thetarsvalues.npy',np.array(thetarsarr)) 
    """    
    allthetars = np.load('thetarsvalues.npy')
    #print allthetars.shape
    
    """
    thetamean = []
    for j in range(5):
        thetamean.append(0.01*allthetars[j].sum(axis=0))
    np.save('thetameanvalues.npy',np.array(thetamean))
    """
    allthetamean = np.load('thetameanvalues.npy')
    #print allthetamean.shape
    """
    thetamle = []
    for i in range(10,60,10):
        onethetamle = []
        logregobj = LogisticRegression(tol=1e-2, C=1e+33)
        returnedobj = logregobj.fit(Xtr[:i],Ytr[:i])
        wmle = returnedobj.coef_[0]
        bmle = returnedobj.intercept_[0]
        for item in wmle:
            onethetamle.append(item)
        onethetamle.append(bmle)
        print onethetamle
        thetamle.append(onethetamle)
    print thetamle
    np.save('thetamlevalues.npy',np.array(thetamle))
    """
    allthetamle = np.load('thetamlevalues.npy')
    #print allthetamle
    
    """
    thetamap = []
    for i in range(10,60,10):
        onethetamap = []
        logregobj = LogisticRegression(tol=1e-2, C=100)
        returnedobj = logregobj.fit(Xtr[:i],Ytr[:i])
        wmap = returnedobj.coef_[0]
        bmap = returnedobj.intercept_[0]
        for item in wmap:
            onethetamap.append(item)
        onethetamap.append(bmap)
        print onethetamap
        thetamap.append(onethetamap)
    print thetamap
    np.save('thetamapvalues.npy',np.array(thetamap))
    """
    allthetamap = np.load('thetamapvalues.npy')
    #print allthetamap  
    
    """
    #2a plots
    thetaindex = 0
    for i in range(10,60,10):
        plt.title("Experiment 2(a)\nScatter Plot for M = " + str(i))
        arrx1 = Xtr[:i,0]
        arrx2 = Xtr[:i,1]
        labely = Ytr[:i]
        color= ['red' if l == 0 else 'green' for l in labely]
        plt.scatter(arrx1, arrx2, color=color)
        proxy_artist0 = mlines.Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="red", markersize=8,label='Class 0')
        proxy_artist1 = mlines.Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="green", markersize=8,label='Class 1')
        first_legend = plt.legend(handles=[proxy_artist0,proxy_artist1],loc=4)
        ax = plt.gca().add_artist(first_legend)
        plot_boundary(allthetars[thetaindex])
        thetaindex = thetaindex + 1
    """
    """
    #2b plots
    thetaindex = 0
    for i in range(10,60,10):
        plt.title("Experiment 2(b)\nScatter Plot for M = " + str(i))
        arrx1 = Xtr[:i,0]
        arrx2 = Xtr[:i,1]
        labely = Ytr[:i]
        color= ['red' if l == 0 else 'green' for l in labely]
        plt.scatter(arrx1, arrx2, color=color)
        proxy_artist0 = mlines.Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="red", markersize=8,label='Class 0')
        proxy_artist1 = mlines.Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="green", markersize=8,label='Class 1')
        first_legend = plt.legend(handles=[proxy_artist0,proxy_artist1],loc=4)
        ax = plt.gca().add_artist(first_legend)
        plot_boundary([allthetamean[thetaindex]])
        thetaindex = thetaindex + 1
    """
    """
    #2c plots
    thetaindex = 0
    for i in range(10,60,10):
        plt.title("Experiment 2(c)\nScatter Plot for M = " + str(i))
        arrx1 = Xtr[:i,0]
        arrx2 = Xtr[:i,1]
        labely = Ytr[:i]
        color= ['red' if l == 0 else 'green' for l in labely]
        plt.scatter(arrx1, arrx2, color=color)
        proxy_artist0 = mlines.Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="red", markersize=8,label='Class 0')
        proxy_artist1 = mlines.Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="green", markersize=8,label='Class 1')
        first_legend = plt.legend(handles=[proxy_artist0,proxy_artist1],loc=4)
        ax = plt.gca().add_artist(first_legend)
        plot_boundary([allthetamap[thetaindex]])
        thetaindex = thetaindex + 1
    """
    """
    #2d plots
    thetaindex = 0
    for i in range(10,60,10):
        plt.title("Experiment 2(d)\nScatter Plot for M = " + str(i))
        arrx1 = Xtr[:i,0]
        arrx2 = Xtr[:i,1]
        labely = Ytr[:i]
        color= ['red' if l == 0 else 'green' for l in labely]
        plt.scatter(arrx1, arrx2, color=color)
        proxy_artist0 = mlines.Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="red", markersize=8,label='Class 0')
        proxy_artist1 = mlines.Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="green", markersize=8,label='Class 1')
        first_legend = plt.legend(handles=[proxy_artist0,proxy_artist1],loc=4)
        ax = plt.gca().add_artist(first_legend)
        plot_boundary([allthetamle[thetaindex]])
        thetaindex = thetaindex + 1
    """    
    #print allthetamean,"\n",allthetamap,"\n",allthetamle
    """
    #2e plots
    thetaindex = 0
    for i in range(10,60,10):
        plt.title("Experiment 2(e)\nScatter Plot for M = " + str(i))
        plt.xlabel('W1')
        plt.ylabel('W2')
        axes = plt.gca()
        axes.set_xlim([np.amax(allthetamle)+1,np.amin(allthetamle)-1])
        axes.set_ylim([np.amax(allthetamle)+1,np.amin(allthetamle)-1])
        arrw1 = allthetars[thetaindex][:,0]
        arrw2 = allthetars[thetaindex][:,1]
        plt.scatter(arrw1, arrw2, color='k',alpha=0.1)
        plt.scatter(allthetamean[thetaindex][0],allthetamean[thetaindex][1],color="green")
        plt.scatter(allthetamap[thetaindex][0],allthetamap[thetaindex][1],color="blue")
        plt.scatter(allthetamle[thetaindex][0],allthetamle[thetaindex][1],color="red")
        proxy_artist0 = mlines.Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="black", markersize=8,label='ThetaRS',alpha=0.5)
        proxy_artist1 = mlines.Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="green", markersize=8,label='ThetaMean')
        proxy_artist2 = mlines.Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="blue", markersize=8,label='ThetaMAP')
        proxy_artist3 = mlines.Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="red", markersize=8,label='ThetaMLE')
        first_legend = plt.legend(handles=[proxy_artist0,proxy_artist1,proxy_artist2,proxy_artist3],loc=2)
        ax = plt.gca().add_artist(first_legend)
        plt.show()
        thetaindex = thetaindex + 1
    """
    #3b computing test set log likelihoods
    """
    testllrs, testllmean, testllmap, testllmle = [],[],[],[]    
    #print predictive_distribution(100,Xte[0],Yte[0],allthetars[0])
    #print predictive_distribution(1,Xte[0],Yte[0],[allthetamean[0]])
    for m in range(5):
        testllrssum, testllmeansum, testllmapsum, testllmlesum = 0,0,0,0
        for i in range(len(Xte)):
            testllrssum = testllrssum + np.log(predictive_distribution(100,Xte[i],Yte[i],allthetars[m]))
            testllmeansum = testllmeansum + np.log(predictive_distribution(1,Xte[i],Yte[i],[allthetamean[m]]))
            testllmapsum = testllmapsum + np.log(predictive_distribution(1,Xte[i],Yte[i],[allthetamap[m]]))
            testllmlesum = testllmlesum + np.log(predictive_distribution(1,Xte[i],Yte[i],[allthetamle[m]]))
        testllrs.append(testllrssum)
        testllmean.append(testllmeansum)
        testllmap.append(testllmapsum)
        testllmle.append(testllmlesum)
    np.save('testllrsvalues.npy',np.array(testllrs))
    np.save('testllmeanvalues.npy',np.array(testllmean))
    np.save('testllmapvalues.npy',np.array(testllmap))
    np.save('testllmlevalues.npy',np.array(testllmle))
    print testllrs,"\n",testllmean,"\n", testllmap,"\n",testllmle
    """
    tllrs = np.load('testllrsvalues.npy')
    tllmean = np.load('testllmeanvalues.npy')
    tllmap = np.load('testllmapvalues.npy')
    tllmle = np.load('testllmlevalues.npy')
    #print tllrs,"\n",tllmean,"\n",tllmap,"\n",tllmle
    
    """
    #3b plot
    m = [10,20,30,40,50]
    plt.title("Experiment (3b)")
    plt.xlabel("Number Of Training Cases")
    plt.ylabel("Test Set Log Likelihood")
    bi, = plt.plot(m,tllrs,color='k',label='Bayesian Inference')
    plmean, = plt.plot(m,tllmean,color='g',label='Plug-In Mean')
    plmap, = plt.plot(m,tllmap,color='b',label='Plug-In Map')
    plmle, = plt.plot(m,tllmle,color='r',label='Plug-In MLE')
    plt.legend(handles=[bi,plmean,plmap,plmle])
    plt.show()
    """
if __name__ == "__main__":
  main()
