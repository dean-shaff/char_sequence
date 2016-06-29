import numpy as np 
import theano.tensor as T
import theano 
import h5py
import time 
from preprocessor import TextProcessor


class TextDataset(object):

    def __init__(self, filename,**kwargs):
        """
        args:
            -filename: should be an hdf5 file 
        """
        self.filename = filename  
        self.get_data(**kwargs)  

    def get_data(self,shared=False):
        """
        Grab the data from the dataset file and turn it into one hot vectors.
        kwargs:
            -shared: If true, load data into shared variable.s 
        """
        print("Loading data into memory")
        t0 = time.time() 
        f = h5py.File(self.filename,'r') 
        dat = f['dat'][...]
        print("Data loaded from hdf5 file. Took {:.2f} seconds".format(time.time() - t0))
        chars = f.attrs['chars'] 
        dat_one_hot = np.zeros((dat.shape[0], len(chars)),dtype=theano.config.floatX) #need float because we're doing regression 
        indices = np.arange(dat.shape[0]) 
        dat_one_hot[indices,dat] = 1.0 
        print("One hot vectors generated. Took {:.2f} second".format(time.time() - t0))
        input_dat = dat_one_hot[:-1]
        obs_dat = dat_one_hot[1:]
        if shared:
            input_dat = theano.shared(input_dat)
            obs_dat = theano.shared(obs_dat) 
            print("Theano shared variables created. Took {:.2f} seconds.".format(time.time()  - t0))
        else:
            pass 
        f.close() 
        self.dat = dat 
        self.data_one_hot = dat_one_hot 
        self.obs_dat = obs_dat 
        self.input_dat = input_dat
if __name__ == '__main__':
    filename = 'shakespeare.hdf5'
    txter = TextDataset(filename) 
