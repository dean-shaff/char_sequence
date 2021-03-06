import theano 
import theano.tensor as T 
import numpy as np 
import h5py
import time 

class HiddenLayer(object):
    """
    Hidden Layer for NN below. 
    """
    def __init__(self,input,dim_in, dim_out,name,rng,trans_func):
        """
        Initialize a single hidden layer. 
        args:
            input: a symbolic Theano variable, with 3 dimensions 
            dim_in: The input dimensionality
            dim_out: The output dimensionality of the layer 
        """
        # W_shape and W_bound from LeNet tutorial. Don't know if they're any good 
        W_shape = dim_in + dim_out
        W_bound = 1
        for dim in W_shape:
            W_bound *= dim 
        self.W = theano.shared(np.asarray(rng.uniform(low=-1.0/W_bound,
                                                        high = 1.0/W_bound,size=W_shape),
                                                        dtype=theano.config.floatX),name='W')

        self.b = theano.shared(np.asarray(np.zeros(dim_out), dtype=theano.config.floatX), name='b')
        self.params = [self.W, self.b] 
        self.lin_output = T.tensordot(input,self.W) + self.b  
        self.nonlin_output = trans_func(self.lin_output)  
    
    def set_params(self):
        pass

    def get_params(self):
        pass 


class NN(object):
    """
    A neural net that takes as input stacked two dimensional tensors. 
    The output is the same dimension as the output. Dealing with a regression task 
    """
    def __init__(self,input,dim,**kwargs):
        """
        Initialize layers, and their respective symbolic output. 
        args:
            -dim: A nested list containing the dimensions of the neural net to be constructed.
                If the final layer and the input layer don't have the same dimensions it won't work. 
        kwargs:
            -t_func: Transfer function -- the function that will turn linear transformations into
                    nonlinear outputs. 
        """
        assert dim[0] == dim[-1], "You're not building a sequence encoder"
        
        self.seq_len = dim[0][0] # seqence length  
        self.char_len = dim[0][1] # number of characters  
        trans_func = kwargs.get('t_func',T.nnet.sigmoid)
        rng = np.random.RandomState(1234) 
        hiddenLayers = [HiddenLayer(input,dim[0], dim[1],'h0', rng, trans_func)]
        params = hiddenLayers[-1].params

        for i in xrange(1,len(dim)-1):
            name_i = "h_{}".format(i) 
            hiddenLayers.append(HiddenLayer(hiddenLayers[-1].nonlin_output,dim[i],dim[i+1],name_i,rng, trans_func))
            params += hiddenLayers[-1].params 
        
        self.params = params 
        self.hiddenLayers = hiddenLayers 
        self.nn_nonlin_output = self.hiddenLayers[-1].nonlin_output
        self.nn_lin_output = self.hiddenLayers[-1].lin_output

    def sqr_error(self,y):
        """
        Calculate the error between the input (x) and the observation (y) 
        args:
            -y: A theano shared variable corresponding to the observation. Will be 3 dimensional
        """
        return T.mean((y - self.nn_nonlin_output)**2)


    def negloglikelihood(self,y):
        """
        Calculate the negative log likelihood error between the input and the observation 
        ***This is really darn slow ATM***
        args:
            -y: A theano shared variable corresponding to the observation. Will be 3-dimensional. 
        """
        softmax = T.nnet.sigmoid(self.nn_lin_output) 
        log_tot = T.log(softmax)
        y_arg_max = T.argmax(y, axis=2) # assuming y is in hot max form. Dim = (minibatch size, sequence len)  
        # first index is minibatch, second is sequence index, last is character
        log_prob, updates = theano.scan(fn=lambda i,yi: log_tot[i,T.arange(y_arg_max.shape[1]),yi],
                                        sequences = [T.arange(y_arg_max.shape[0]), y_arg_max]) 

                                # self.log_prob = log_prob
        return -T.mean(log_prob)

    def error(self, y):
        """
        Calculate classification error.
        """
        y_arg_max = T.argmax(y, axis=2) # assuming y is one hot 
        softmax = T.nnet.sigmoid(self.nn_lin_output) #is this right??? 
        y_guess = T.argmax(softmax, axis=2) 
        error = T.sum(T.neq(y_guess, y_arg_max))
        return error 

    def save_params(self,*args):
        """
        save model parameters to a hdf5 file. 
        args:
            values that we want written to the name of the hdf5 file 
        kwargs:
            values that we want set as attributes of the hdf5 file 
        """
        print("Starting to save model parameters...") 
        t0 = time.time() 
        name = 'model'
        if args:
            for arg in args:
                model += "_{}"
                model.format(arg) 
            model += ".hdf5"  
        else:
            name = 'model.hdf5'
        f = h5py.File(name, 'w')
        for h in self.hiddenLayers:
            grp = f.create_group(h.name)
            grp.create_dataset('W',data=h.W.get_value())
            grp.create_dataset('b',data=h.b.get_value()) 
        for key in kwargs.keys():
            f.attrs[key] = kwargs[key] 

        f.close() 
       
    def __getattr__(self, val):
        """
        Get an attribute that was loaded into a model file.
        Will raise an error if the key hasn't been set yet. 
        args:
            -val: A key that will get a value
        """
        pass 

    def load_params(self,filename):

        f = h5py.File(filename) 
        
        f.close() 





        


if __name__ == '__main__':
    x = T.tensor3('x') 
    y = T.tensor3('y') 
    nn = NN(x,[[10,40],[50,20],[10,40]]) 
    cost = nn.sqr_error(y)
    f = theano.function([x,y],cost)

    
    #rng = np.random.RandomState(1234)
    #h0 = HiddenLayer(x,[10,40],[50,20],rng) 
    #f = theano.function([x],h0.output)
    print(f(np.arange(5*10*40).reshape((5,10,40)),np.arange(5*10*40).reshape((5,10,40)))) 









