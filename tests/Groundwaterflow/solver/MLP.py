import pickle
import numpy as np

from itertools import product

from pyDOE import lhs as lhcs
from scipy.stats.distributions import norm as norm_dist

from keras.models import Sequential
from keras.layers import Dense

class MLP:
    def __init__(self, model, datapoints, n_samples):
        
        # Internalise the parameters.
        self.model = model
        self.datapoints = datapoints
        self.n_samples = n_samples
        
        # Get the number of modes.
        self.mkl = self.model.mkl
        self.eigenvalues = self.model.random_process.eigenvalues
        self.eigenvectors = self.model.random_process.eigenvectors
        
        self.lamb = self.model.lamb
        
        # Create a matrix of random samples from a Gaussian Latin Hypercube.
        self.samples = lhcs(self.mkl, samples = self.n_samples)
        self.samples = norm_dist(loc=0, scale=1).ppf(self.samples)
        
    def compute_model_response(self):
        
        # Solve the model using every sample in the hypercube
        self.data = np.zeros((self.n_samples, len(self.datapoints)))
        for i in range(self.n_samples):

            self.model.solve(self.samples[i,:])
            
            # Extract data and save it in the matrix.
            self.data[i,:] = self.model.get_data(self.datapoints)
            
            #print('Processing... {0:.2f}%'.format(i/self.n_samples*100), end='\r')
        
        #print('\nDone.')
        
        # Split the data into test and training data
        self.X_train = self.samples[:int(0.9*self.n_samples),:]; self.y_train = self.data[:int(0.9*self.n_samples),:]
        self.X_test  = self.samples[int(0.9*self.n_samples):,:]; self.y_test  = self.data[int(0.9*self.n_samples):,:]
        

    def fit(self, epochs=500, batch_size=100):

        # Set up an ANN
        self.mlp = Sequential()
        self.mlp.add(Dense(4*self.mkl, input_dim=self.mkl, activation='sigmoid'))
        self.mlp.add(Dense(8*self.mkl, activation='relu'))
        self.mlp.add(Dense(4*self.mkl, activation='relu'))
        self.mlp.add(Dense(len(self.datapoints), activation='exponential'))

        self.mlp.compile(loss='mse', optimizer='rmsprop')
        
        # Train the ANN
        self.history = self.mlp.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose = 1)
        
        # Check the performance of the ANN
        self.score = self.mlp.evaluate(self.X_test, self.y_test)
        
    def pickle(self, filename):
        
        data_dict = {'X_train': self.X_train, 
                     'y_train': self.y_train, 
                     'X_test': self.X_test, 
                     'y_test': self.y_test}
        
        pickle_dict = {'MLP': self.mlp, 
                       'Datapoints': self.datapoints,
                       'Data': data_dict, 
                       'lambda': self.lamb,
                       'MKL': self.mkl,
                       'Eigenvalues': self.eigenvalues, 
                       'Eigenvectors': self.eigenvectors}
        
        with open(filename, 'wb') as f:
            
            pickle.dump(pickle_dict, f)
        
    def predict(self, parameters):
        
        # This method allows predicting the model response from a vector of modes.
        return self.mlp.predict(np.expand_dims(parameters, axis = 0)).flatten()
        

    def solve(self, parameters):
        
        # This is simply a proxy-method which allows the model to plugged into the MCMC.
        self.output = self.predict(parameters)
        

    def get_data(self, datapoints = None):
        
        # Another proxy.
        return self.output


    def get_outflow(self):
        
        # Another proxy.
        return None
        

class PickledMLP(MLP):
    def __init__(self, filename):
        
        with open(filename, 'rb') as f:
            pickle_dict = pickle.load(f)
            
        self.mlp = pickle_dict['MLP']
        
        self.datapoints = pickle_dict['Datapoints']
        data_dict = pickle_dict['Data']
        
        self.X_train = data_dict['X_train']
        self.y_train = data_dict['y_train']
        self.X_test = data_dict['X_test']
        self.y_test = data_dict['y_test']
        
        self.lamb = pickle_dict['lambda']
        self.mkl = pickle_dict['MKL']
        self.eigenvalues = pickle_dict['Eigenvalues']
        self.eigenvectors = pickle_dict['Eigenvectors']
        
        self.history = None
        
    def compute_model_response(self):
        raise TypeError('Cannot compute model response for a pickled MLP')
        return
        
    def fit(self, **kwargs):
        raise TypeError('Cannot fit a pickled MLP')
        return
        
    #def pickle(self, *args):
    #    raise TypeError('Why would you pickle a pickle?')
    #    return
