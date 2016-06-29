import numpy as np 
import theano.tensor as T
import theano 
import h5py


class Dataset(object):

    def __init__(self, filename):
        """
        args:
            -filename: should be an hdf5 file 
        """
        self.filename 
        

    def get_data(self):
        """
        Grab the data from the dataset file. 
        Store it into two Theano shared variables.
        """
        f = h5py.File(self.filename,'r') 
        dat = f['dat'][...]
        input_dat = dat[:-1]
        obs_dat = dat[1:]
        input_dat = theano.shared(input_dat)
        obs_dat = theano.shared(obs_dat) 

