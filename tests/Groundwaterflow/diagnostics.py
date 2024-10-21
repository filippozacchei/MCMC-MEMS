import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from numpy.linalg import det

class MLPDiagnostics:
    
    '''
    MLPDiagnostics gathers and prints diagnostics from an ANN.
    Diagnostics will only be returned from the indexes listed in the
    data-points vector.
    '''
    
    def __init__(self, model, sampling_points):
        
        # Internalise the MLP
        self.model = model
        self.sampling_points = sampling_points
        
        # Predict for the training dataset.
        self.predictions_train = np.zeros(self.model.y_train.shape)
        for i, sample in enumerate(self.model.X_train):
            self.predictions_train[i,:] = self.model.predict(sample)
        
        # Save the predictions in a dataframe and gather it for easy plotting with Seaborn-FacetGrid.
        df_train_pred = pd.DataFrame(self.predictions_train[:,sampling_points], columns = sampling_points)
        df_train_pred['Sample'] = np.arange(0, self.model.y_train.shape[0])
        df_train_pred = pd.melt(df_train_pred, id_vars=['Sample'], value_vars=sampling_points, var_name='Sampling Point', value_name='Predicted Value')
        
        # Getthe true values and gather the dataframe.
        df_train_true = pd.DataFrame(self.model.y_train[:,sampling_points], columns = sampling_points)
        df_train_true['Sample'] = np.arange(0, self.model.y_train.shape[0])
        df_train_true = pd.melt(df_train_true, id_vars=['Sample'], value_vars=sampling_points, var_name='Sampling Point', value_name='True Value')
        
        # Merge the two training dataframes according to Sample and Smapling Point (output index)
        self.df_train = pd.merge(df_train_pred, df_train_true, on = ['Sample', 'Sampling Point'])
        
        # Predict for the testing dataset.
        self.predictions_test = np.zeros(self.model.y_test.shape)
        for i, sample in enumerate(self.model.X_test):
            self.predictions_test[i,:] = self.model.predict(sample)
        
        # Save the predictions in a dataframe and gather it.
        df_test_pred = pd.DataFrame(self.predictions_test[:,sampling_points], columns = sampling_points)
        df_test_pred['Sample'] = np.arange(0, self.model.y_test.shape[0])
        df_test_pred = pd.melt(df_test_pred, id_vars=['Sample'], value_vars=sampling_points, var_name='Sampling Point', value_name='Predicted Value')
        
        # Save the true values in a dataframe and gather it.
        df_test_true = pd.DataFrame(self.model.y_test[:,sampling_points], columns = sampling_points)
        df_test_true['Sample'] = np.arange(0, self.model.y_test.shape[0])
        df_test_true = pd.melt(df_test_true, id_vars=['Sample'], value_vars=sampling_points, var_name='Sampling Point', value_name='True Value')
        
        # Merge the two testing-dataframes.
        self.df_test = pd.merge(df_test_pred, df_test_true, on = ['Sample', 'Sampling Point'])
        
    def xy_line(self, color):
        x = y = np.linspace(0,1)
        plt.plot(x,y, c = color)
        
    def compute_bias(self):
        
        self.bias = self.model.y_test - self.predictions_test
        
        self.mu_bias = self.bias.mean(axis = 0)
        
        self.SIGMA_bias = np.zeros((len(self.mu_bias), len(self.mu_bias)))
        for b in self.bias:
            self.SIGMA_bias += np.outer(b - self.mu_bias, b - self.mu_bias)
        
        self.SIGMA_bias = self.SIGMA_bias/(len(self.bias) - 1)
        
    def plot_convergence(self, epoch_start = 0):
        
        try:
            history = self.model.history.history['loss'][epoch_start:]
        except:
            print('Found no convergence history')
            return
        
        # Plot the converge according to training epochs.
        plt.title('MSE Convergence')
        plt.xlabel('Epoch'); plt.ylabel('MSE')
        plt.plot(history)
        plt.show()
        
    def plot_performance_train(self):
        
        # Plot the performance on the training dataset.
        sns.set(font_scale=1.4); sns.set_style('ticks')
        g = sns.FacetGrid(self.df_train, col='Sampling Point', col_wrap=4, height = 4)
        g.map(self.xy_line, color = 'k')
        g.map(plt.scatter, 'True Value', 'Predicted Value', color = 'k', alpha = 0.2)
        
    def plot_performance_test(self):
        
        # Plot performance on the testing dataset.
        sns.set(font_scale=1.4); sns.set_style('ticks')
        g = sns.FacetGrid(self.df_test, col='Sampling Point', col_wrap=4, height = 4)
        g.map(self.xy_line, color = 'k')
        g.map(plt.scatter, 'True Value', 'Predicted Value', color = 'k', alpha = 0.2)
    
    def plot_distributions_train(self, cutoff = None):
        
        # Plot distribution from the training dataset.
        g = sns.FacetGrid(self.df_train, col='Sampling Point', col_wrap=5)
        g.map(plt.hist, 'True Value', color = 'c', alpha = 0.5, label = 'True')
        g.map(plt.hist, 'Predicted Value', color = 'm', alpha = 0.5, label = 'Predicted')
        g.add_legend()
        for i in range(len(self.sampling_points)):
            g.axes[i].set_xlabel('')

    def plot_distributions_test(self, cutoff = None):
        
        # Plot distribution from the testing dataset.
        g = sns.FacetGrid(self.df_test, col='Sampling Point', col_wrap=5)
        g.map(plt.hist, 'True Value', color = 'c', alpha = 0.5, label = 'True')
        g.map(plt.hist, 'Predicted Value', color = 'm', alpha = 0.5, label = 'Predicted')
        g.add_legend()
        for i in range(len(self.sampling_points)):
            g.axes[i].set_xlabel('')
            
    def plot_bias(self, datapoints):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (16, 6))
        axes[0].set_title('\u03bc-bias', fontdict = {'fontsize': 16})
        mu_bias = axes[0].scatter(datapoints[:,0], datapoints[:,1], c = self.mu_bias, s = 100); fig.colorbar(mu_bias, ax = axes[0])

        axes[1].set_title('\u03a3-bias', fontdict = {'fontsize': 16})
        SIGMA_bias = axes[1].imshow(self.SIGMA_bias); fig.colorbar(SIGMA_bias, ax = axes[1])

class ChainDiagnostics:

    def __init__(self, chain, model, burnin):
        
        self.chain = chain
        self.model = model
        self.random_process = self.model.random_process
        self.burnin = burnin
        
        self.N = len(self.chain) - burnin
        
        self.parameter_mean = np.array([link.parameters for link in self.chain[burnin:]]).mean(axis = 0)
        
    def get_chain_parameters(self, parameters):
        
        self.parameters = parameters
        
        # Extract chain development for each parameter.
        chain_par = np.zeros((self.N, len(self.parameters)))
        
        for i in range(self.N):
            for j, par in enumerate(self.parameters):
                chain_par[i,j] = self.chain[self.burnin+i].parameters[par]
        
        # Save the development in a dataframe.
        self.df_par = pd.DataFrame(chain_par, columns = self.parameters)
        # Compute the expanding mean.
        self.df_par_mean = self.df_par.expanding().mean()
        
        self.df_par['Link'] = np.arange(self.burnin, self.N+self.burnin)
        self.df_par = pd.melt(self.df_par, id_vars=['Link'], value_vars=self.parameters, var_name='Mode', value_name='Value')
        
        self.df_par_mean['Link'] = np.arange(self.burnin, self.N+self.burnin)
        self.df_par_mean = pd.melt(self.df_par_mean, id_vars=['Link'], value_vars=self.parameters, var_name='Mode', value_name='Mean')

    def plot_means(self):
        g = sns.FacetGrid(self.df_par_mean, col='Mode', col_wrap=5)
        g.map(plt.plot, 'Link', 'Mean')
    
    def plot_caterpillars(self):
        g = sns.FacetGrid(self.df_par, col='Mode', col_wrap=5)
        g.map(plt.plot, 'Link', 'Value')
        
    def plot_distributions(self):
        g = sns.FacetGrid(self.df_par, col='Mode', col_wrap=5)
        g.map(plt.hist, 'Value')
        
    def get_chain_statistics(self):
        
        # Extract chain development for link statistics.
        chain_statistics = np.zeros((self.N, 4))
        for i in range(self.N):
            chain_statistics[i, 0] = int(self.burnin+i)
            chain_statistics[i, 1] = self.chain[self.burnin+i].prior
            chain_statistics[i, 2] = self.chain[self.burnin+i].likelihood
            chain_statistics[i, 3] = self.chain[self.burnin+i].posterior

        self.df_stats = pd.DataFrame(chain_statistics, columns=['Link', 'Prior', 'Likelihood', 'Posterior'])
        
    def plot_chain_statistics(self):
        
        # Plot it!
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize = (16, 4))

        self.df_stats.plot('Link', 'Prior', ax=axes[0], title = 'logPrior', legend = False)
        self.df_stats.plot('Link', 'Likelihood', ax=axes[1], title = 'logLikelihood', legend = False)
        self.df_stats.plot('Link', 'Posterior', ax=axes[2], title = 'logPosterior', legend = False)

    def get_fields(self):
        
        # Get the coords of the domain
        self.coords = np.c_[self.random_process.x, self.random_process.y]
        
        # Stack all the field from the chain.
        self.random_fields = np.zeros((self.N, len(self.coords)))
        for i in range(self.N):
            self.random_process.generate(self.chain[self.burnin+i].parameters)
            self.random_fields[i] = self.random_process.random_field

        # Compute the mean and the variance.
        self.field_mean = np.mean(self.random_fields, axis = 0)
        self.field_var = np.var(self.random_fields, axis = 0)
        
    def plot_fields(self, transform_field = False):
        
        # Transform the field accrording to the field parameters.
        if transform_field:
            field_mean = self.model.field_mean*np.exp(self.model.field_stdev*self.field_mean)
        else:
            field_mean = self.field_mean
        
        # Plot mean and variance.
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (24,9))
        
        axes[0].set_title('Field Mean', fontdict = {'fontsize': 24})
        axes[0].tick_params(labelsize=16)
        mean = axes[0].tricontourf(self.coords[:,0], 
                                   self.coords[:,1], 
                                   field_mean, 
                                   levels = np.linspace(min(field_mean), max(field_mean), 100),
                                   cmap = 'magma'); 
        fig.colorbar(mean, ax=axes[0])
        
        axes[1].set_title('Field Variance', fontdict = {'fontsize': 24})
        axes[1].tick_params(labelsize=16)
        var = axes[1].tricontourf(self.coords[:,0], 
                                  self.coords[:,1], 
                                  self.field_var, 
                                  levels = np.linspace(min(self.field_var), max(self.field_var), 100),
                                  cmap = 'magma');
        fig.colorbar(var, ax=axes[1])
        plt.show()
        
    def compute_mESS(self, k):
        
        self.b_n = int(self.N**(1/k))
        self.a_n = int(self.N/self.b_n)
        
        n_pars = len(self.chain[0].parameters)
        parameters = np.array([link.parameters for link in self.chain[self.burnin:]])
        parameter_means = parameters.mean(axis = 0)
        
        LAMBDA = np.zeros((n_pars, n_pars))
        for i in parameters:
            LAMBDA += np.outer(i,i)
        LAMBDA = 1/self.N * (LAMBDA - (self.N+1)*np.outer(parameter_means, parameter_means))
        
        batch_means = np.zeros((self.a_n, n_pars))
        for i in range(self.a_n):
            batch_means[i,:] = parameters[i*self.b_n:(i+1)*self.b_n,:].mean(axis = 0)
        
        SIGMA = np.zeros((n_pars, n_pars))
        for i in range(self.a_n):
            SIGMA += np.outer(batch_means[i,:] - parameter_means, 
                              batch_means[i,:] - parameter_means)
        SIGMA = self.b_n/(self.a_n-1) * SIGMA
        
        return self.N * (det(LAMBDA)/det(SIGMA))**(1/n_pars)

    def compute_ESS(self):
        
        N = len(self.chain[self.burnin:])
        qois = np.zeros(N)
        for i, link in enumerate(self.chain[self.burnin:]):
            qois[i] = link.qoi
            
        qoi_mean = np.mean(qois)
        t_max = int(np.floor(N/2))
        S_tau = 1.5
        
        time_vec = np.zeros(N+t_max)
        
        for i, qoi in enumerate(qois):
            time_vec[i] = qoi - qoi_mean
        
        freq_vec = np.fft.fft(time_vec)
        
        # Compute out1*conj(out2) and store in out1 (x+yi)*(x-yi)
        for i in range(N + t_max):
            freq_vec[i] = np.real(freq_vec[i])**2 + np.imag(freq_vec[i])**2
        
        # Now compute the inverse fft to get the autocorrelation (stored in timeVec)
        time_vec = np.fft.ifft(freq_vec)
        
        for i in range(t_max+1):
            time_vec[i] = np.real(time_vec[i])/(N - i)
        
        # The following loop uses ideas from "Monte Carlo errors with less errors." by Ulli Wolff 
        # to figure out how far we need to integrate
        G_int = 0.0
        W_opt = 0
        
        for i in range(1, t_max+1):
            G_int += np.real(time_vec[i]) / np.real(time_vec[0])
            
            if G_int <= 0:
                tau_W = 1e-15
            else:
                tau_W = S_tau / np.log((G_int+1) / G_int)
            
            g_W = np.exp(-i/tau_W) - tau_W/np.sqrt(i*N)
            
            if g_W < 0:
                W_opt = i
                t_max = min([t_max, 2*i])
                break
        
        # Correct for bias
        CFbb_opt = np.real(time_vec[0])
        
        for i in range(W_opt + 1):
            CFbb_opt += 2*np.real(time_vec[i+1])
        
        CFbb_opt = CFbb_opt/N
        
        for i in range(t_max+1):
            time_vec[i] += CFbb_opt
        
        scale = np.real(time_vec[0])
        
        for i in range(W_opt):
            time_vec[i] = np.real(time_vec[i])/scale
        
        tau_int = 0.0    
        for i in range(W_opt):
            tau_int += np.real(time_vec[i])    
        tau_int -= 0.5
        
        frac = min([1.0, 1/(2*tau_int)])
        
        return N*frac
        
