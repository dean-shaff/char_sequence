from NN import NN 
from dataset import TextDataset
import theano
import theano.tensor as T 
import numpy as np 
import h5py
import time 


class Trainer(object):

    def __init__(self,model,dataset):
        """
        Initalize a SGD trainer. 
        args:
            - model: the "model" or neural net we want to train 
            - dataset: on object of dataset.Dataset 
        """
        self.model = model
        self.dataset = dataset
        # input data is of the following dimension: (mini_batch_size, sequence_length, char_len)
        self.dataset.cut_by_sequence(self.model.seq_len) 

    def compile_functionsSGD(self,x,y,reg=True):
        """
        compile functions for training using SGD (stochastic gradient descent)
        args:
            - x: a theano symbolic variable that corresponds to network input. Note that we 
                have to pass this, despite already forming the network before initialization of 
                the trainer object. 
            - y: a theano symbolic variable whose dimensionality is equal to that of 
                the model input 
        kwargs:
            - reg: Whether or not this is a regression or classification task. If regression, use 
                square error cost. If classification, use negative log likelihood. 
        """
        lr = T.scalar("lr",dtype=theano.config.floatX) 
        mb = T.scalar("mb", dtype='int64')
        index = T.scalar('index',dtype='int64') 
        if reg:
            self.cost = self.model.sqr_error(y) 
        elif not reg:
            self.cost = self.model.negloglikelihood(y) 
        self.error = self.model.error(y) 
        t0 = time.time()
        gparams = [T.grad(self.cost,param) for param in self.model.params] 
        updates = [(param, param-lr*gparam) for param, gparam in zip(self.model.params, gparams)]
        print("Starting to compile theano functions...") 
        
        self.train_model = theano.function(
                inputs = [index, mb, lr],
                outputs = self.cost,
                updates = updates, 
                givens = {
                    x:self.dataset.in_train[index*mb:(index+1)*mb],
                    y:self.dataset.obs_train[index*mb:(index+1)*mb]
               }
            )
        self.test_model = theano.function(
                    inputs = [x,y],
                    outputs = self.error
            ) 
        print("Theano functions compiled! Took {:.2f} seconds.".format(time.time()- t0))
    
    def train_SGD(self,**kwargs):
        """
        kwargs:
            - mb: minibatch size. The size of the third dimension of the input vectors.  
            - lr: learning rate. The learning rate for network training. 
            - nepochs: number of epochs for training
            - test_rate: Rate at which to test network 
            - mom: momentum (not yet implemented)  
        """
        print("\n\nStarting to train neural net...")
        t0 = time.time() 
        mb = kwargs.get('mb',25)
        lr = kwargs.get('lr',0.01)
        nepochs = kwargs.get('nepochs',50)
        test_rate = kwargs.get('test_rate',1) 
        mom = kwargs.get('mom',None)
    
        train_batches = self.dataset.in_train.get_value().shape[0] // mb 
        print("\n\nEach epoch has {} batches".format(train_batches)) 
        for epoch in xrange(nepochs):
            t1 = time.time()
            e_cost = 0 
            for b in xrange(train_batches):
                b_cost = self.train_model(b,mb,lr)
                e_cost += b_cost
                if b % 1 == 0:
                    print("Current mini-batch cost: {}".format(b_cost))
#                    print("Current test error: {}".format(self.test_model(self.dataset.in_test.get_value(),self.dataset.obs_test.get_value())))
            print("Current training cost: {}".format(e_cost / train_batches))
            print("Epoch {} took {:.2f} seconds".format(epoch, time.time() - t1))
            if epoch % test_rate == 0:
                #test the net 
                print("Current test cost: {}".format(self.test_model(self.dataset.in_test.get_value(),self.dataset.obs_test.get_value())))


if __name__ == '__main__':
    x = T.tensor3('x') 
    y = T.tensor3('y') 
    dat = TextDataset('shakespeare.hdf5') 
    seq_len = 50 
    nn = NN(x,[[seq_len,dat.char_len],[40,30],[seq_len,dat.char_len]])
    trainer = Trainer(nn,dat)
    trainer.compile_functionsSGD(x,y,reg=True)
    trainer.train_SGD(mb=1000) 
