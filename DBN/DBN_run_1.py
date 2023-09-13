#Deep activities recognition model

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import sys
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))   # Join parent path to import library
from Shared.MLP import HiddenLayer, MLP
from Shared.logisticRegression2 import LogisticRegression 
from Shared.rbm_har import  RBM, GRBM
import timeit
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report 
import warnings
warnings.filterwarnings('ignore') 
from Shared.read_dataset import read_dataset


class DBN(object):
    """Deep belief network
    A deep belief network is obtained by stacking several RBMs on top of the each other.
    The hidden layer of the RBM at layer 'i' becomes the input of the RBM at layer 'i+1'.
    The first layer RBM gets as input the input of the network, and the hidden layer of
    the last RBM represents the output. When used for classification, the DBN is treated
    as a MLP, by adding a logistic regression layer on top.
    """

    def __init__(self, n_inp = 784, n_out = 10, hidden_layer_sizes = [500, 500]):
        """ This class is made to support a variable number of layers.
        :param n_inps: int, dimension of the input to the DBN
        :param n_outs: int, demension of the output of the network
        :param hidden_layer_sizes: list of ints, intermediate layers size, must contain
        at least one value
        """

        self.sigmoid_layers = []
        self.layers = []
        self.params = []
        self.n_layers = len(hidden_layer_sizes)

        assert self.n_layers > 0

        #define the grape
        height, weight, channel = n_inp
        self.x = tf.placeholder(tf.float32, [None, height, weight, channel])
        self.y = tf.placeholder(tf.float32, [None, n_out])

        for i in range(self.n_layers):
            # Construct the sigmoidal layer

            # the size of the input is either the number of hidden units of the layer
            # below or the input size if we are on the first layer

            if i == 0:
                input_size = height * weight *channel
            else:
                input_size = hidden_layer_sizes[i - 1]

            # the input to this layer is either the activation of the hidden layer below
            # or the input of the DBN if you are on the first layer
            if i == 0:
                layer_input = tf.reshape(self.x, [-1, height*weight*channel])

            else:
                layer_input = self.sigmoid_layers[-1].output


            sigmoid_layer = HiddenLayer(input = layer_input, n_inp = input_size, 
                n_out = hidden_layer_sizes[i], activation = tf.nn.sigmoid)

            #add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)

            # Its arguably a philosophical question... but we are going to only
            # declare that the parameters of the sigmoid_layers are parameters of the DBN.
            # The visible biases in the RBM are parameters of those RBMs, but not of the DBN

            self.params.extend(sigmoid_layer.params)
            if i == 0:
                rbm_layer = GRBM(inp = layer_input, n_visible = input_size, n_hidden = hidden_layer_sizes[i], W = sigmoid_layer.W, hbias = sigmoid_layer.b) 
            else:
                rbm_layer = RBM(inp = layer_input, n_visible = input_size, n_hidden = hidden_layer_sizes[i], W = sigmoid_layer.W, hbias = sigmoid_layer.b)  
            self.layers.append(rbm_layer)
            
        self.logLayer = LogisticRegression(input= self.sigmoid_layers[-1].output, 
            n_inp = hidden_layer_sizes[-1], n_out = n_out)
        self.params.extend(self.logLayer.params)
        #print(self.sigmoid_layers[-1].output)
        #print(hidden_layer_sizes[-1], n_out)
        #compute the cost for second phase of training, defined as the cost of the
        # logistic regression output layer

        self.finetune_cost = self.logLayer.cost(self.y)

        #compute the gradients with respect to the model parameters symbolic variable that
        # points to the number of errors made on the minibatch given by self.x and self.y
        self.pred = self.logLayer.pred
        self.accuracy = self.logLayer.accuracy(self.y)
        """
        # Initialize with 0 the weights W as a matrix of shape (n_inp, n_out)
        out_weights = tf.Variable(tf.zeros([hidden_layer_sizes[-1], n_out]))
        # Initialize the biases b as a vector of n_out 0s
        out_biases  = tf.Variable(tf.zeros([n_out]))
        out_ = tf.nn.softmax(tf.matmul(self.sigmoid_layers[-1].output, out_weights) + out_biases)
        self.finetune_cost = -tf.reduce_mean(tf.reduce_sum(self.y * tf.log(out_)))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(self.finetune_cost)
        correct_prediction = tf.equal(tf.argmax(out_,1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        """

    def pretraining(self, sess, train_set_x, batch_size = 100, pretraining_epochs = 100, 
        learning_rate = 0.001, k = 1, display_step = 1):
        """ Generates a list of functions, for performing one step of gradient descent at
        a given layer. The function will require as input the minibatch index, and to train
        an RBM you just need to iterate, calling the corresponding function in all minibatch 
        indexes.
        :param train_set_x: tensor, contains all datapoints used for traing the RBM
        :param batch_size: int, size of a minibatch
        :param k: number of Gibbs steps to do in CD-k/ PCD-k
        :param learning_rate: learning rate
        :param pretraining_epochs: int, maximal number of epochs to run the optimizer
        """

        #begining of a batch, given index
        start_time = timeit.default_timer()
        batch_num = int(train_set_x.train.num_examples / batch_size)

        #Pretraining layer by layer
        for i in range(self.n_layers):
            # Get the cost and the updates list
            #Using CD-k here for training each RBM
            #TODO: change cost function to reconstruction error
            
            #cost = self.layers[i].get_reconstruction_cost()
            #train_ops = self.layers[i].get_train_ops(lr = learning_rate, persistent = None, k = k)
            #print(self.rbm_layers[i].n_visible, self.rbm_layers[i].n_hidden)
            
            if i ==0:
                learning_rate = 0.0001
            else:
                learning_rate = 0.0001
            
            cost, train_ops = self.layers[i].get_train_ops(lr = learning_rate, persistent = None, k = k)
            #cost = self.layers[i].get_reconstruction_cost()
            for epoch in range(pretraining_epochs):
                avg_cost = 0.0
                for j in range(batch_num):
                    batch_xs, batch_ys = train_set_x.train.next_batch(batch_size)
                    _ = sess.run(train_ops, feed_dict = {self.x : batch_xs,})
                    c = sess.run(cost, feed_dict = {self.x: batch_xs, })
                    avg_cost += c / batch_num
                    
                if epoch % display_step == 0:
                    print("Pretraining layer {0} Epoch {1}".format(i+1, epoch +1) + " cost {:.9f}".format(avg_cost))        
                    
                    #plt.imsave("new_filters_at_{0}.png".format(epoch),tile_raster_images(X = sess.run(tf.transpose(self.rbm_layers[i].W)), img_shape = (21, 21), tile_shape = (10,10), tile_spacing = (1,1)), cmap = 'gray')
                #plt.show()
                
        end_time = timeit.default_timer()
        print("time {0} minutes".format((end_time - start_time)/ 60.))


    def fine_tuning(self, sess, train_set_x, batch_size = 100 , training_epochs = 1000 , learning_rate = 0.01, display_step = 1):
        """ Genarates a function train that implements one step of finetuning, a function validate 
        that computes the error on a batch from the validation set, and a function test that computes
        the error on a batch from the testing set
        :param datasets: tensor, a list contain all the dataset
        :param batch_size: int, size of a minibatch
        :param learning_rate: int, learning rate
        :param training_epochs: int, maximal number of epochs to run the optimizer
        """

        start_time = timeit.default_timer()
        saver = tf.train.Saver(max_to_keep=4)
        train_ops = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(self.finetune_cost, var_list= self.params)

        #Accuracy
        accuracy = self.accuracy
        batch_num = int(train_set_x.train.num_examples/batch_size)
        ACC_max = 0
        pre_max = 0
        rec_max = 0
        accu    = []
        for epoch in range(training_epochs):
            avg_cost = 0.0
            for i in range(batch_num):
                b =[]
                d = []
                batch_xs, batch_ys = train_set_x.train.next_batch(batch_size)
                _= sess.run(train_ops, feed_dict = {self.x :batch_xs, self.y : batch_ys} )
                c =sess.run(self.finetune_cost, feed_dict = {self.x :batch_xs, self.y : batch_ys} )
                avg_cost += c / batch_num
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch +1), "cost:", "{:.9f}".format(avg_cost))
                acc = sess.run(accuracy, feed_dict = {self.x: train_set_x.test.segments, self.y: train_set_x.test.labels})
                pr= sess.run(self.pred, feed_dict = {self.x: train_set_x.test.segments, self.y: train_set_x.test.labels})
                #------
                d = np.append(d, sess.run(tf.argmax(pr, axis =1))) 
                np.savetxt('d.txt', d ,delimiter=',') 
                b = np.append(b , sess.run(tf.argmax(self.y, axis =1 ), feed_dict ={self.x: train_set_x.test.segments, self.y: train_set_x.test.labels})) 
                a = confusion_matrix(b, d) 
                print(classification_report(b, d))

                #--------
                # print("Accuracy:", acc)

                # Please double check the evaluation methods
                FP = a.sum(axis=0) - np.diag(a)
                FN = a.sum(axis=1) - np.diag(a)
                TP = np.diag(a)
                TN = a.sum() - (FP + FN + TP)
                ac = (TP + TN) / (TP + FP + FN + TN)
                ACC = ac.sum() / 2
                precision = precision_score(b, d, average='weighted')
                recall = recall_score(b, d, average='weighted')
                accu = np.append(accu, ac.sum() / 2)
                np.savetxt('accu.txt', accu,delimiter=',') 
                print("ACCURACY: {0}, PRECISION: {1}, RECALL: {2}:".format(ACC, precision, recall))
                end_time = timeit.default_timer()
                print("Time {0} minutes".format((end_time- start_time)/ 60.))

        end_time = timeit.default_timer()
        print("Time {0} minutes".format((end_time- start_time)/ 60.))


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    filename1 = dir_path + "/Dataset/Train.csv"
    filename2 = dir_path + "/Dataset/Test.csv"

    dataset = read_dataset(filename1, filename2)
    
    learning_rate = 0.0001
    pre_epochs = 100
    training_epochs = 100
    batch_size = 10000
    display_step = 10

    #DBN structure
    n_inp = [1, 1, 21]
    hidden_layer_sizes = [500, 500, 500]
    n_out = 2

    dbn = DBN(n_inp = n_inp, hidden_layer_sizes = hidden_layer_sizes, n_out = n_out)

    tf.set_random_seed(seed = 999)

    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = "0"
    with tf.Session(config=config) as sess:
        sess.run(init)
        dbn.pretraining(sess, train_set_x = dataset, pretraining_epochs=pre_epochs)
        dbn.fine_tuning(sess, train_set_x = dataset)