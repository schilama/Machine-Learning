from __future__ import division
from __future__ import print_function
import argparse
import math
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

def save_prediction(predict_y):
    assert type(predict_y) is np.ndarray
    np.save('q2_prediction', predict_y)
    print("prediction file saved to q2_prediction.npy")


class Autoencoder(object):
    """An autoencoder architecture

    Arguments:
        alpha: trade-off parameter for the composite objective function.
        epochs: number of epochs for training
    """
    def __init__(self, alpha=.5, epochs=5):
        self.alpha = alpha
        self.epochs = epochs
        tf.reset_default_graph()  
        tf.set_random_seed(0)
        
        #Initializing model parameters to random values (default dtype = tf.float32)
        self.w1 = tf.get_variable("w1", [784,32], trainable = True, initializer=tf.contrib.layers.xavier_initializer(seed=0))
        self.b1 = tf.get_variable("b1", [32], trainable = True, initializer=tf.contrib.layers.xavier_initializer(seed=0))
        self.w2 = tf.get_variable("w2", [32,784], trainable = True, initializer=tf.contrib.layers.xavier_initializer(seed=0))
        self.b2 = tf.get_variable("b2", [784], trainable = True, initializer=tf.contrib.layers.xavier_initializer(seed=0))
        self.w3 = tf.get_variable("w3", [32,10], trainable = True, initializer=tf.contrib.layers.xavier_initializer(seed=0))
        self.b3 = tf.get_variable("b3", [10], trainable = True, initializer=tf.contrib.layers.xavier_initializer(seed=0))     
        
        #Creating placeholder tensors 
        self.tf_X = tf.placeholder(tf.float32)
        self.tf_y = tf.placeholder(tf.float32)
        
        #Computing the model now
        h = tf.nn.relu(tf.matmul(self.tf_X,self.w1) + self.b1)
        self.xcap = tf.matmul(h,self.w2) + self.b2
        self.ycap = tf.nn.softmax(tf.matmul(h,self.w3) + self.b3)
        self.yclabel = tf.argmax(self.ycap,1)
        #self.yclabel = tf.argmax(tf.nn.softmax(tf.matmul(h,self.w3) + self.b3,dim=0),1)
        #self.ycap = tf.one_hot(self.yclabel,depth=10)
        #self.yclabelshape = tf.shape(self.yclabel)
        #self.ycapshape = tf.shape(self.ycap)
        self.xcapshape = tf.shape(self.xcap)
        
        self.nvalue = tf.cast(tf.shape(self.tf_X)[0],tf.float32)
        self.dvalue = tf.cast(tf.shape(self.tf_X)[1],tf.float32)
        
        #computing loss
        self.aeloss = (1.0 - self.alpha)*(1/self.dvalue)*tf.reduce_sum(tf.square(self.tf_X - self.xcap))
        self.closs = (-self.alpha)*tf.reduce_sum(tf.multiply(self.tf_y,tf.log(self.ycap + 1e-10)))
        self.loss = (1/self.nvalue)*(self.closs + self.aeloss)
        
        #Gradient Descent
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9, beta2=0.999, epsilon=1e-08)
        self.train = optimizer.minimize(self.loss)
        
        self.sess = tf.Session()
        self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op)
        
    def objective(self, X, y):
        """Objective function.

        Arguments:
            X (numpy ndarray, shape = (samples, 784)):
                Training input matrix where each row is a feature vector.
            y (numpy ndarray, shape = (samples, 10)):
                one hot encoded class label

        Returns:
            Composite objective function value. (float)
        """
        objval = self.sess.run([self.loss],{self.tf_X:X, self.tf_y:y.astype('float32')})
        print (objval[0],np.array(objval).squeeze())
        return objval[0]
            
    def predict_x(self, X):
        """Predict decoded X for samples in X.

        Arguments:
            X (numpy ndarray, shape = (samples, 784)):
                Input matrix where each row is a feature vector.

        Returns:
            decode_X (numpy ndarray, shape = (samples, 784)):
                Decoded matrix where each row is a feature vector.
        """
        decode_X = self.sess.run([self.xcap,self.xcapshape], {self.tf_X: X})
        """
        xcaps = decode_X[1]
        dx = decode_X[0]
        xbla = np.array(decode_X[0])
        xblah = xbla.squeeze()
        print (dx.shape,xcaps,xcaps.shape,xblah.shape,xbla.shape)
        """
        return np.array(decode_X[0])

    def predict_y(self, X):
        """Predict class label for samples in X.

        Arguments:
            X (numpy ndarray, shape = (samples, 784)):
                Input matrix where each row is a feature vector.

        Returns:
            y (numpy ndarray, shape = (samples,)):
                class labels
        """
        y = self.sess.run([self.yclabel], {self.tf_X: X})
        blah = np.array(y).squeeze()
        print (blah.shape)
        return np.array(y).squeeze()

    def fit(self, X, y):
        """Train the model according to the given training data.

        Arguments:
            X (numpy ndarray, shape = (samples, 784)):
                Training input matrix where each row is a feature vector.
            y (numpy ndarray, shape = (samples, 10)):
                one hot encoded class label
        """
        print ("lalala")
        """
        templist = self.sess.run([self.yclabelshape,self.ycapshape], {self.tf_X: X[:2], self.tf_y: y.astype('float32')[:2]})
        print (templist[0])
        print (templist[1])
    
        #print (templist[2])
        #print (templist[3])
        """
        for epoch in range(self.epochs):
            print ("Epoch", epoch)
            for i in range(0, y.shape[0], 50):
                mylist = self.sess.run([self.train,self.loss,self.aeloss,self.closs], {self.tf_X: X[i:i+50], self.tf_y: y.astype('float32')[i:i+50]})
            print ("Total loss:",mylist[1]," Class loss:",mylist[2],"AEloss: ",mylist[3])  

        
        
    def get_model_params(self):
        """Get the parameters of the model.

        Returns:
            w1 (numpy ndarray, shape = (784, 32)):
            b1 (numpy ndarray, shape = (32,)):
                weights and bias for FC(784, 32)

            w2 (numpy ndarray, shape = (32, 784)):
            b2 (numpy ndarray, shape = (784,)):
                weights and bias for FC(32, 784)

            w3 (numpy ndarray, shape = (32, 10)):
            b3 (numpy ndarray, shape = (10,)):
                weights and bias for FC(32, 10)
        """
        return self.sess.run(self.w1),self.sess.run(self.b1),self.sess.run(self.w2),self.sess.run(self.b2), self.sess.run(self.w3),self.sess.run(self.b3)


    def set_model_params(self, w1, b1, w2, b2, w3, b3):
        """Set the parameters of the model.

        Arguments:
            w1 (numpy ndarray, shape = (784, 32)):
            b1 (numpy ndarray, shape = (32,)):
                weights and bias for FC(784, 32)

            w2 (numpy ndarray, shape = (32, 784)):
            b2 (numpy ndarray, shape = (784,)):
                weights and bias for FC(32, 784)

            w3 (numpy ndarray, shape = (32, 10)):
            b3 (numpy ndarray, shape = (10,)):
                weights and bias for FC(32, 10)
        """
        assign_w1 = self.w1.assign(w1) 
        assign_w2 = self.w2.assign(w2) 
        assign_w3 = self.w3.assign(w3)
        assign_b1 = self.b1.assign(b1)  
        assign_b2 = self.b2.assign(b2)
        assign_b3 = self.b3.assign(b3)

        self.sess.run([assign_w1,assign_w2,assign_w3, assign_b1, assign_b2, assign_b3])   
        
        
def main():
    train_data_path = '../data/q2_train.npz'
    data = np.load(train_data_path)
    train_X = data['X_train']
    test_X = data['X_test']
    train_y = data['y']
    np.random.seed(0)
    """
    model = Autoencoder(alpha=0.5,epochs=50)
    model.fit(train_X, train_y)
    #model.objective(train_X,train_y)
    decode_X = model.predict_x(train_X)
    #print (decode_X.shape)
    predict_y = model.predict_y(train_X)
    # save q2prediciton.txt
    predict_test_y = model.predict_y(test_X)
    save_prediction(predict_test_y)
    
    """
    rng_state = np.random.get_state()
    np.random.shuffle(train_X)
    np.random.set_state(rng_state)
    np.random.shuffle(train_y)
    
    rtrain_X = train_X[:48000]
    rtrain_y = train_y[:48000]
    cvtrain_X = train_X[48000:54000]
    cvtrain_y = train_y[48000:54000]
    rtest_X = train_X[54000:]
    rtest_y = train_y[54000:]
    
    ep = [10,20,30,40,50]
    classerrorlist = []
    mselist = []
    #for i in range(5):
    if(1==1):
        model = Autoencoder(alpha=0.5,epochs=50)
        model.fit(rtrain_X, rtrain_y)
        #model.objective(train_X,train_y)
        #decode_X = model.predict_x(cvtrain_X)
        #print (decode_X.shape)
        y_class_predict_raw = model.predict_y(cvtrain_X)
        y_class_predict = []
        train_y_cnum = []
        for j in range(0,cvtrain_y.shape[0]):
            if 1 in cvtrain_y[j]:
                y_class_predict.append(y_class_predict_raw[j])
                train_y_cnum.append(np.argmax(cvtrain_y[j]))
        #print (len(train_y_cnum),len(y_class_predict))
        classerror = 1- accuracy_score(train_y_cnum,y_class_predict)
        classerrorlist.append(classerror)
        
        x_ae_predict = model.predict_x(cvtrain_X)
        mse = mean_squared_error(cvtrain_X,x_ae_predict)
        mselist.append(mse)
    
    print (classerrorlist)
    print (mselist)    
    
    
    """
    classerrorlist = []
    mselist = []
    alphavalue = []
    myalpha = 0.0
    for i in range(11):
        print (myalpha)
        mymodel = Autoencoder(myalpha,epochs=50)
        mymodel.fit(train_X, train_y)      
        
        y_class_predict_raw = mymodel.predict_y(train_X) 
        y_class_predict = []
        train_y_cnum = []
        for j in range(0,train_y.shape[0]):
            if 1 in train_y[j]:
                y_class_predict.append(y_class_predict_raw[j])
                train_y_cnum.append(np.argmax(train_y[j]))
        print (len(train_y_cnum),len(y_class_predict))
        classerror = 1- accuracy_score(train_y_cnum,y_class_predict)
        classerrorlist.append(classerror)
        
        x_ae_predict = mymodel.predict_x(train_X)
        mse = mean_squared_error(train_X,x_ae_predict)
        mselist.append(mse)
        
        alphavalue.append(myalpha)
        myalpha = myalpha + 0.1
        
    print (classerrorlist)
    print (mselist)
    
    import matplotlib.pyplot as plt
    
    plt.figure(1)
    plt.plot(alphavalue,classerrorlist,color='g')
    plt.title("Experiment 2(c)")
    plt.xlabel('Value of alpha')
    plt.ylabel('Classification error')
    
    plt.figure(2)
    plt.plot(alphavalue,mselist,color='b')
    plt.title("Experiment 2(c)")
    plt.xlabel('Value of alpha')
    plt.ylabel('MSE')    
    
    plt.show()
    """

if __name__ == '__main__':
    main()
