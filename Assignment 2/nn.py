from __future__ import division
from __future__ import print_function
import argparse
import math
import numpy as np
from utils import load_data
import tensorflow as tf

np.random.seed(0)


class NN(object):
    """A network architecture of simultaneous localization and
       classification of objects in images.

    Arguments:
        alpha: trade-off parameter for the composite objective function.
        epochs: number of epochs for training
    """
    def __init__(self, alpha=0.5, epochs=5):
        #Wipes the graph clean
        tf.reset_default_graph()  
        #Tradeoff between classification and localization loss
        self.alpha = alpha
        #For gradient descent
        self.epochs = epochs
        tf.set_random_seed(0)
        #Initializing model parameters to random values (default dtype = tf.float32)
        self.w1 = tf.get_variable("w1", [3600,256], trainable = True, initializer=tf.contrib.layers.xavier_initializer(seed=0))
        self.b1 = tf.get_variable("b1", [256], trainable = True, initializer=tf.contrib.layers.xavier_initializer(seed=0))
        self.w2 = tf.get_variable("w2", [256,64], trainable = True, initializer=tf.contrib.layers.xavier_initializer(seed=0))
        self.b2 = tf.get_variable("b2", [64], trainable = True, initializer=tf.contrib.layers.xavier_initializer(seed=0))
        self.w3 = tf.get_variable("w3", [64,32], trainable = True, initializer=tf.contrib.layers.xavier_initializer(seed=0))
        self.b3 = tf.get_variable("b3", [32], trainable = True, initializer=tf.contrib.layers.xavier_initializer(seed=0))     
        self.w4 = tf.get_variable("w4", [32,2], trainable = True, initializer=tf.contrib.layers.xavier_initializer(seed=0))
        self.b4 = tf.get_variable("b4", [2], trainable = True, initializer=tf.contrib.layers.xavier_initializer(seed=0))    
        self.w5 = tf.get_variable("w5", [32,1], trainable = True, initializer=tf.contrib.layers.xavier_initializer(seed=0))
        self.b5 = tf.get_variable("b5", [1], trainable = True, initializer=tf.contrib.layers.xavier_initializer(seed=0))        
        
        #Creating placeholder tensors 
        self.tf_X = tf.placeholder(tf.float32)
        self.tf_y_class = tf.placeholder(tf.float32)
        self.tf_y_loc = tf.placeholder(tf.float32)
        
        #Computing the model now
        h1 = tf.nn.relu(tf.matmul(self.tf_X,self.w1) + self.b1)
        h2 = tf.nn.relu(tf.matmul(h1,self.w2) + self.b2)
        h3 = tf.nn.relu(tf.matmul(h2,self.w3) + self.b3)
        self.yc = tf.matmul(h3,self.w5) + self.b5
        #self.yprob = tf.sigmoid(yclout)
        self.yl = tf.matmul(h3,self.w4) + self.b4        
        
        #computing loss
        self.class_loss = self.alpha*tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.squeeze(self.yc),labels=self.tf_y_class))
        #local_loss = (1.0 - self.alpha)*tf.reduce_sum(tf.square(tf.norm(self.tf_y_loc - self.yl, axis=1)))
        self.local_loss = (1.0 - self.alpha)*tf.reduce_sum(tf.square(self.tf_y_loc - self.yl))
        self.loss = self.class_loss + self.local_loss
        
        #Gradient Descent
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9, beta2=0.999, epsilon=1e-08)
        self.train = optimizer.minimize(self.loss)
        
        self.sess = tf.Session()
        self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op)
        
        """
        with self.sess.as_default():
            print (self.w1.eval())
        """    
        
    def objective(self, X, y_class, y_loc):
        """Objective function.

        Arguments:
            X (numpy ndarray, shape = (samples, 3600)):
                Training input matrix where each row is a feature vector.
            y_class (numpy ndarray, shape = (samples,)):
                Training labels. Each entry is either 0 or 1.
            y_loc (numpy ndarray, shape = (samples, 2)):
                Training (vertical, horizontal) locations of the objects.
                This is the coordinate of the top-left corner of the 28x28
                region where the object presents.

        Returns:
            Composite objective function value.
        """
        return self.sess.run(self.loss,{self.tf_X: X, self.tf_y_class: y_class.astype('float32'), self.tf_y_loc: y_loc})
        

    def predict(self, X):
        """Predict class labels and object locations for samples in X.

        Arguments:
            X (numpy ndarray, shape = (samples, 3600)):
                Input matrix where each row is a feature vector.

        Returns:
            y_class (numpy ndarray, shape = (samples,)):
                predicted labels. Each entry is either 0 or 1.
            y_loc (numpy ndarray, shape = (samples, 2)):
                The predicted (vertical, horizontal) locations of the
                objects.
        """
        mylist = self.sess.run([self.yc,self.yl],{self.tf_X: X})
        y_predict = mylist[0] 
        y_predict[y_predict>0]=1
        y_predict[y_predict<=0]=0   
        offset_predict = mylist[1]
        return np.array(y_predict).squeeze(), np.array(offset_predict)

    def fit(self, X, y_class, y_loc):
        """Train the model according to the given training data.

        Arguments:
            X (numpy ndarray, shape = (samples, 3600)):
                Training input matrix where each row is a feature vector.
            y_class (numpy ndarray, shape = (samples,)):
                Training labels. Each entry is either 0 or 1.
            y_loc (numpy ndarray, shape = (samples, 2)):
                Training (vertical, horizontal) locations of the
                objects.
        """    
        for epoch in range(self.epochs):
            print ("Epoch", epoch)
            """
            seed_state = np.random.get_state()
            #print (seed_state)
            np.random.shuffle(X)
            np.random.set_state(seed_state)
            np.random.shuffle(y_class)
            np.random.set_state(seed_state)
            np.random.shuffle(y_loc)
            #print (seed_state)"""
            for i in range(0, y_class.shape[0], 50):
                mylist = self.sess.run([self.train, self.loss, self.class_loss, self.local_loss, self.yc, self.tf_y_class], {self.tf_X: X[i:i+50], self.tf_y_class: y_class.astype('float32')[i:i+50], self.tf_y_loc: y_loc[i:i+50]})
                #print self.sess.run(self.loss, {self.tf_X: X, self.tf_y_class: y_class.astype('float32'), self.tf_y_loc: y_loc})
                #print (self.get_model_params()[5])
                # evaluate training accuracy
                #curr_loss = self.sess.run([self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4, self.b4, self.w5, self.b5, self.objective], {self.tf_X: X, self.tf_y_class: y_class.astype('float32'), self.tf_y_loc: y_loc})
                #print("loss: %s"% curr_loss)
        
    def get_model_params(self):
        """Get the parameters of the model.

        Returns:
            w1 (numpy ndarray, shape = (3600, 256)):
            b1 (numpy ndarray, shape = (256,)):
                weights and bias for FC(3600, 256)

            w2 (numpy ndarray, shape = (256, 64)):
            b2 (numpy ndarray, shape = (64,)):
                weights and bias for FC(256, 64)

            w3 (numpy ndarray, shape = (64, 32)):
            b3 (numpy ndarray, shape = (32,)):
                weights and bias for FC(64, 32)

            w4 (numpy ndarray, shape = (32, 2)):
            b4 (numpy ndarray, shape = (2,)):
                weights and bias for FC(32, 2) for location outputs

            w5 (numpy ndarray, shape = (32, 1)):
            b5 (float):
                weights and bias for FC(32, 1) for the logit for
                class probability output
        """
        #Is there a more elegant way?        
        return self.sess.run(self.w1),self.sess.run(self.b1), self.sess.run(self.w2),self.sess.run(self.b2), self.sess.run(self.w3),self.sess.run(self.b3), self.sess.run(self.w4),self.sess.run(self.b4), self.sess.run(self.w5),self.sess.run(self.b5)

    def set_model_params(self, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5):
        """Set the parameters of the model.

        Arguments:
            w1 (numpy ndarray, shape = (3600, 256)):
            b1 (numpy ndarray, shape = (256,)):
                weights and bias for FC(3600, 256)

            w2 (numpy ndarray, shape = (256, 64)):
            b2 (numpy ndarray, shape = (64,)):
                weights and bias for FC(256, 64)

            w3 (numpy ndarray, shape = (64, 32)):
            b3 (numpy ndarray, shape = (32,)):
                weights and bias for FC(64, 32)

            w4 (numpy ndarray, shape = (32, 2)):
            b4 (numpy ndarray, shape = (2,)):
                weights and bias for FC(32, 2) for location outputs

            w5 (numpy ndarray, shape = (32, 1)):
            b5 (float):
                weights and bias for FC(32, 1) for the logit for
                class probability output
        """
        
        assign_w1 = self.w1.assign(w1) 
        assign_w2 = self.w2.assign(w2) 
        assign_w3 = self.w3.assign(w3)
        assign_w4 = self.w4.assign(w4)      
        assign_w5 = self.w5.assign(w5)  
        assign_b1 = self.b1.assign(b1)  
        assign_b2 = self.b2.assign(b2)
        assign_b3 = self.b3.assign(b3)
        assign_b4 = self.b4.assign(b4)
        assign_b5 = self.b5.assign(b5)
        self.sess.run([assign_w1,assign_w2,assign_w3,assign_w4,assign_w5, assign_b1, assign_b2, assign_b3, assign_b4, assign_b5])   
        
        
def main():
    train_data_path = '../data/q2_train.npz'
    test_data_path = '../data/q2_test.npz'
    train_X, train_y_class, train_y_loc = load_data(train_data_path)
    test_X, test_y_class, test_y_loc = load_data(test_data_path)
    
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import mean_squared_error
    
    model = NN(alpha=0.5,epochs=20)
    model.fit(train_X, train_y_class.astype('float32'), train_y_loc)
    
    y_class_predict, y_loc_predict = model.predict(train_X)
    
    print (1- accuracy_score(train_y_class,y_class_predict))
    print (mean_squared_error(train_y_loc,y_loc_predict))
    
    y_class_predict, y_loc_predict = model.predict(test_X)
    
    print (1- accuracy_score(test_y_class,y_class_predict))
    print (mean_squared_error(test_y_loc,y_loc_predict))
    
    classerrorlist = []
    mselist = []
    alphavalue = []
    myalpha = 0.0
    for i in range(11):
        print (myalpha)
        mymodel = NN(myalpha,epochs=20)
        mymodel.fit(train_X, train_y_class.astype('float32'), train_y_loc)      
        y_class_predict, y_loc_predict = mymodel.predict(test_X)
        classerror = 1- accuracy_score(test_y_class,y_class_predict)
        mse = mean_squared_error(test_y_loc,y_loc_predict)
        classerrorlist.append(classerror)
        mselist.append(mse)
        alphavalue.append(myalpha)
        myalpha = myalpha + 0.1
        
    print (classerrorlist)
    print (mselist)
    
    import matplotlib.pyplot as plt
    
    plt.figure(1)
    plt.plot(alphavalue,classerrorlist,color='g')
    plt.title("Experiment 3(c)")
    plt.xlabel('Value of alpha')
    plt.ylabel('Classification error')
    
    plt.figure(2)
    plt.plot(alphavalue,mselist,color='b')
    plt.title("Experiment 3(c)")
    plt.xlabel('Value of alpha')
    plt.ylabel('MSE')    
    
    plt.show() 
    
if __name__ == '__main__':
    main()
