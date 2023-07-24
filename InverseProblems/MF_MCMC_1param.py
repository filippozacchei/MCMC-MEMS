# Bayesian model twinning

# Author: Matteo Torzoni

#Load some libraries
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import pickle
keras.backend.clear_session()

esempio         = 'L_FRAME_MF_HF_RETE' # Specify the example you are dealing with
ID_test         = '1_0_MCMC' #ID of the data to test
work_ID         = 'MF_HF_RETE_1_0_MCMC_1par_noscatter\\' #name of the working directory (to be created in advance and provided with the file Recordings_MCMC.pkl)
path            = 'D:\\Users\\Matteo\\Corigliano\\' + esempio + '\\Dati\\' #local directory
path_data_test  = path + 'istantest_' + ID_test #path for the labels 
work_path       = path + work_ID
ID_HF           = 'MF_HF_RETE_1_0' #ID of the HF NN
HF_NN_path      = path + ID_HF #Path to load the HF NN
ID_LF           = '1_0'  #ID of the LF NN
LF_NN_path      = 'D:\\Users\\Matteo\\Corigliano\\L_FRAME_MF_LF_RETE\\Dati\\MF_LF_RETE_'+ID_LF #Path to load the LF NN
ID_basis        = '1_0' #ID of the basis needed to expand the POD coefficient provided by the LF NN
LF_basis_path   = 'D:\\Users\\Matteo\\Corigliano\\L_FRAME_MF_LF_RETE\\Dati\\istantrain_'+ID_basis #Path to load the LF basis

# Options for the plots 
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rc('font', size=24)          # controls default text sizes
plt.rc('axes', titlesize=24)     # fontsize of the axes title
plt.rc('axes', labelsize=24)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=24)    # fontsize of the tick labels
plt.rc('ytick', labelsize=24)    # fontsize of the tick labels
plt.rc('legend', fontsize=20)    # legend fontsize
plt.rc('figure', titlesize=24)  # fontsize of the figure title
plt.rc('hatch', linewidth = 0.5)  # previous pdf hatch linewidth

# ----------------------------------------------------------------------------

which_inst = 3 #identifier of the instance to consider            

n_channels     = 8 #number of sensors recording the data
N_entries = 200 #length of each signal (time instants)
signal_resolution = 0.005 #sampling period in seconds

removed_ratio = 0.2 #ratio of N_entries to be removed in the computation of the likelihood from the first initial phase of the signals
limit=int(N_entries*removed_ratio) #time instants to be removed

#Neurons of the HF NN
HF_neurons_LSTM_1 = 16
HF_neurons_LSTM_2 = 16
HF_neurons_LSTM_3 = 32

#some data relevant to the modeling of the problem necessary to the normalization of the inputs of the surrogate model
freq_min = 10 # minimum value for the load frequency 
freq_delta_max = 50 # delta max for the load frequency 
Ampl_min = 1000 # minimum value for the load amplitude 
Ampl_delta_max = 4000 # delta max for the load amplitude 
Coord_x_min = 0.15 # minimum value for the x coordinate of the damage 
Coord_x_delta_max = 3.7 # delta max for the x coordinate of the damage 
Coord_y_min = 0.15 # minimum value for the y coordinate of the damage  
Coord_y_delta_max = 3.7 # delta max for the y coordinate of the damage 

#MCMC parameters
niters = 5000 #length of the chain
burnin = 500 #transitory period to be removed
burnin_cov = 250 #transitory period to initialize the tuining of the standard deviation of the proposal
MC_coord_start = 0.5 #starting point - initialize the chain at x = 3.7, y = 0
sigma_proposal= 5e-2 # initial value of the standard deviation of the proposal pdf (Truncated Gaussian centered on the previous accepted sample)
naccept = 0 # initialization of the number of accepted samples
N_obs = 3 # number of observations collected for each MCMC simulation
thinning = 4 #subsampling ratio to make samples uncorrelated

n_parameters = 1 #Number of parameters to be identified
sp = 2.4**2/n_parameters #coefficient used in the adaptation procedure 

# ----------------------------------------------------------------------------

def read_HF_input(path_data): # Load HF inputs
    Frequency_path = path_data + '\\Frequency.csv' #load the data from .csv file        
    Frequency      = np.genfromtxt(Frequency_path)
    Amplitude_path = path_data + '\\Amplitude.csv' #load the data from .csv file
    Amplitude      = np.genfromtxt(Amplitude_path)
    Coord_x_path = path_data + '\\Coord_x.csv' #load the data from .csv file
    Coord_x      = np.genfromtxt(Coord_x_path)     
    Coord_y_path = path_data + '\\Coord_y.csv' #load the data from .csv file                           
    Coord_y      = np.genfromtxt(Coord_y_path)
    #NORMALIZZO DATI IN INGRSSO
    Frequency = (Frequency - freq_min) / freq_delta_max #normalization in 0,1 
    Amplitude = (Amplitude - Ampl_min) / Ampl_delta_max #normalization in 0,1 
    Coord_x = (Coord_x - Coord_x_min) / Coord_x_delta_max #normalization in 0,1 
    Coord_y = (Coord_y - Coord_y_min) / Coord_y_delta_max #normalization in 0,1 
    N_ist = len(Frequency) #compute the total number of loaded instances 
    
    X_HF = np.zeros((N_ist,4)) #Build the empty tensor collecting the input for the HF NN
    X_HF[:,0] = Frequency
    X_HF[:,1] = Amplitude
    X_HF[:,2] = Coord_x
    X_HF[:,3] = Coord_y
        
    return X_HF, N_ist

def load_HF_signals(path_data, N_ist, N_entries, N_obs): # Load HF observations
    Recordings = np.load(path_data+'Recordings_MCMC.pkl',allow_pickle=True) #load the data from .pkl file
    Observations_HF = np.zeros((N_ist,N_obs,n_channels,N_entries)) #Build the empty tensor collecting the HF observations
    for i1 in range(N_ist):
        for i2 in range(N_obs):
            for i3 in range(n_channels):
                Observations_HF[i1,i2,i3,:] = Recordings[i1*N_obs+i2,i3*N_entries:(i3+1)*N_entries]
    return Observations_HF

# ----------------------------------------------------------------------------

#USAGE PREDICTION:
    
X_HF, N_ist = read_HF_input(path_data_test) #Load LF and HF inputs 
Y_HF  = load_HF_signals(work_path,N_ist//N_obs,N_entries,N_obs) #Load HF observations
    
# ---------------------------------------------------------------------------- 
    
#SETUP OF THE SURROGATE MODEL

basis = np.load(LF_basis_path+'\\basis.pkl',allow_pickle=True) # Load the LF basis
N_basis = np.size(basis,1)  # Size of the LF basis

LF_net = keras.models.load_model(LF_NN_path + '\\LF_model') #Load the LF NN
LF_mean = np.load(LF_NN_path+'\\mean_LF_POD.npy') #Load the means of the POD coeffs required to renormalize 
LF_std = np.load(LF_NN_path+'\\std_LF_POD.npy') #Load the stds of the POD coeffs required to renormalize 
LF_signals_means = np.load(LF_NN_path+'\\LF_signals_means.npy') #Load the means of the LF signals required for normalization 
LF_signals_stds = np.load(LF_NN_path+'\\LF_signals_stds.npy') #Load the stds of the LF signals required for normalization 

HF_net_trained = keras.models.load_model(HF_NN_path + '\\HF_model') #Load the HF NN
HF_signals_means = np.load(HF_NN_path+'\\HF_signals_means.npy') #Load the means of the HF signals required to renormalize 
HF_signals_stds = np.load(HF_NN_path+'\\HF_signals_stds.npy') #Load the stds of the HF signals required to renormalize 

#Recompile the HF NN to produce signals for the entire length of the considered time interval
HF_input = layers.Input(shape=(N_entries,5+n_channels), name='input_hf')
x = layers.LSTM(units=HF_neurons_LSTM_1, return_sequences=True, name='Reccurrent_1')(HF_input)                 
x = layers.LSTM(units=HF_neurons_LSTM_2, return_sequences=True, name='Reccurrent_2')(x)                         
x = layers.LSTM(units=HF_neurons_LSTM_3, return_sequences=True, name='Reccurrent_3')(x)                        
x = layers.LSTM(units=n_channels, return_sequences=True, name='Reccurrent_4')(x)                         
HF_output = layers.Dense(units=n_channels, activation=None, name='Linear_mapping')(x)   
HF_net_to_pred = keras.Model(inputs=HF_input, outputs=HF_output, name="HF_model_prediction")
HF_net_to_pred.summary()

#Assign to the new compiled HF NN the trained weights
for i1 in range(len(HF_net_trained.layers)):
    HF_net_to_pred.layers[i1].set_weights(HF_net_trained.layers[i1].get_weights())

# ---------------------------------------------------------------------------- 
    
#INITIALIZE SOME VARIABLES

LF_signals = np.zeros((N_obs,N_entries,n_channels)) #structure to collect the reconstructed LF signals
Input_HF = np.zeros((N_obs,N_entries,5+n_channels)) #structure to collect the HF input
Input_HF_start = np.zeros((N_obs,N_entries,5+n_channels)) #structure to collect the HF input only used at the first iteration

like_hist = np.zeros((niters+1)) #initialize the history of the computed likelihoods
proposed_params = np.zeros((niters+1)) #initilize the history of the values of the proposed parameter
chain = np.zeros((niters+1)) #initialize the Chain of samples
chain[0] = MC_coord_start #The chain is started in the middle of support

to_tune_cov = np.zeros((niters)) #array storing the values of proposed parameter that have to be used in order to adapt the std of the proposal on the fly 
index_for_cov = 0 #index to control the adaption

# ---------------------------------------------------------------------------- 
    
# DEFINE SOME FUNCTIONS 

def RMS(vect): # Root Mean Square
    return np.sqrt(np.mean(np.square(vect)))

def RMSE(vect1, vect2): # Root Mean Squared Error
    return np.sqrt(np.mean(np.square(vect1 - vect2)))

def RSSE(vect1, vect2): # Root Sum Squared Error
    return np.sqrt(np.sum(np.square(vect1 - vect2)))

def single_param_like_single_obs(obs,mapping): #Computation of the likelihood for a single obesrvation, assuming each channel independent
    for i1 in range(n_channels):
        rmse = RMSE(obs[i1], mapping[i1])
        rsse = RSSE(obs[i1], mapping[i1])
        like_single_dof = 1./(np.sqrt(2.*np.pi)*rmse) * np.exp(-((rsse**2)/(2.*(rmse**2)))) #Gaussian likelihood
        if i1 == 0:
            like_single_obs = like_single_dof *1e30 #(1e30 it is a normalization constant to avoid numerical issues)
        else:
            like_single_obs = like_single_obs * like_single_dof *1e30 #multiply the likelihood of each channel
    return like_single_obs

prior_coord = lambda theta: st.uniform.pdf(theta) #A priori uniform pdf over the damage coordinate

def target(total_like, prior, theta): #Compute the target quantity to decide weather to accept or reject the current sample
    return total_like*prior(theta)

# BEGIN OF MCMC 

#Apply MCMC for each testing instance (for each instance we have N_obs observations)
i1 = which_inst #which instance to consider            
weights = np.linspace(1,0.2,N_basis) #coefficients used to weight the regression over the POD coefficients

# BEGIN OF THE LF PART

LF_mapping = LF_net.predict_on_batch(X_HF[i1*N_obs:(i1+1)*N_obs,0:2]) #regression over the POD interpolation coefficients

for i2 in range(N_obs):
    for i3 in range(N_basis):
        LF_mapping[i2,i3] = LF_mapping[i2,i3] / weights[i3] #reweight the coefficients 
        LF_mapping[i2,i3]=LF_mean[i3]+LF_mapping[i2,i3]*LF_std[i3] #renormalize the coefficients
            
LF_reconstruct = np.matmul(basis,LF_mapping.T).T #Expand the LF signals projecting the POD interpolation coefficients on the POD basis
    
for i2 in range(N_obs):
    for i3 in range(n_channels):
        LF_reconstruct[i2,i3*N_entries:(i3+1)*N_entries] = (LF_reconstruct[i2,i3*N_entries:(i3+1)*N_entries] - LF_signals_means[i3])/LF_signals_stds[i3] #normalize each channel of the LF signals
        LF_signals[i2,:,i3] = LF_reconstruct[i2,i3*N_entries:(i3+1)*N_entries] #Recast the reconstructed LF signals separating each channel
    
# ---------------------------------------------------------------------------- 

# BEGIN OF THE HF PART  --> this is called iteratively inside the MCMC loop

k=-1
while k<(niters-1):
    k+=1
    if k==0: 
        
        #recast the value of the damage position defined in the interval (0,1) in terms of (x;y) coordinates required by the surrogate model
        if chain[0] <= 0.5:
            damage_x = chain[0]*2.
            damage_y = 0.
        elif chain[0] > 0.5:
            damage_x = 1.
            damage_y = (chain[0] - 0.5)*2.
        
        #Define a structure collecting the input parameters for the surrogate model at the first iteration: load frequency, load amplitude, damage position along x, damage position along y
        X_input_MF_start = np.transpose(np.array([X_HF[i1*N_obs:(i1+1)*N_obs,0], X_HF[i1*N_obs:(i1+1)*N_obs,1], [damage_x,damage_x,damage_x], [damage_y,damage_y,damage_y]]))
        
        #Define the structure that is given as input to the surrogate model at the first iteration
        for i2 in range(N_obs):
            Input_HF_start[i2,:,0:4]=X_input_MF_start[i2,:] #input parameters defined above
            Input_HF_start[i2,:,4] = np.linspace(signal_resolution, signal_resolution * N_entries, N_entries) #time index
            for i3 in range(n_channels):
                Input_HF_start[i2,:,5+i3]=LF_signals[i2,:,i3] #LF approximations 

        Y_start=HF_net_to_pred(Input_HF_start).numpy() #predict the HF signals
         
        for i2 in range(N_obs): #renormalize the HF signals
            for i3 in range(n_channels):
                Y_start[i2,:,i3] = HF_signals_means[i3] + (Y_start[i2,:,i3] * HF_signals_stds[i3])             
        
        for i2 in range(N_obs): #compute the likelihood
            like_single_obs_before = single_param_like_single_obs(Y_HF[i1,i2,:,limit:],np.transpose(Y_start[i2,limit:,:])) #likelihood for the single observation
            if i2 == 0: 
                like_tot_before = like_single_obs_before
            else:
                like_tot_before = like_tot_before * like_single_obs_before #multiply the likelihood of each observation
                
        like_hist[0] = like_tot_before #store the likelihood value
        proposed_params[0] = chain[0] #store the first value of the proposed parameters (always accepted)
        
        target_before = target(like_tot_before, prior_coord, chain[0]) #compute the target quantity
       
    #From here we repeat for each iteration
    
    a, b = (0.0 - chain[k]) / sigma_proposal, (1.0 - chain[k]) / sigma_proposal #Update the parameters of the proposal (truncated Gaussian) on the basis of the current center
    #Metropolis-Hasting
    temp = st.truncnorm(a, b, loc = chain[k], scale = sigma_proposal).rvs() #sample the proposed parameter, given the last accepted sample
    #Metropolis
    #temp = st.norm(loc = chain[k], scale = sigma_proposal).rvs() #sample the proposed parameter, given the last accepted sample
    
    #recast the value of the damage position defined in the interval (0,1) in terms of (x;y) coordinates required by the surrogate model
    if temp <= 0.5:
        damage_x = temp*2.
        damage_y = 0
    elif temp > 0.5:
        damage_x = 1.
        damage_y = (temp - 0.5)*2.
    
    if k == 0:
        #We can do these operations only one time, since the associated quantities are constant during the MCMC loop and equale to those assignede to Input_HF_start
        for i2 in range(N_obs):
            Input_HF[i2,:,0:4]=X_input_MF_start[i2,:]
            Input_HF[i2,:,4] = np.linspace(signal_resolution, signal_resolution * N_entries, N_entries)
            for i3 in range(n_channels):
                Input_HF[i2,:,5+i3]=LF_signals[i2,:,i3]
        
    elif k > 0:
        # After the first iteration we have to update only the sought parameters
        for i2 in range(N_obs):
            Input_HF[i2,:,2]=damage_x
            Input_HF[i2,:,3]=damage_y
    
    Y = HF_net_to_pred(Input_HF).numpy() #predict the HF signals
    
    for i2 in range(N_obs): #renormalize the HF signals
        for i3 in range(n_channels):
            Y[i2,:,i3] = HF_signals_means[i3] + (Y[i2,:,i3] * HF_signals_stds[i3])             

    for i2 in range(N_obs): #compute the likelihood
        like_single_obs_now = single_param_like_single_obs(Y_HF[i1,i2,:,limit:],np.transpose(Y[i2,limit:,:]))
        if i2 == 0:
            like_tot_now = like_single_obs_now
        else:
            like_tot_now = like_tot_now * like_single_obs_now #multiply the likelihood of each observation
            
    like_hist[k+1] = like_tot_now
    proposed_params[k+1] = temp
            
    target_now = target(like_tot_now, prior_coord, temp)
    
    #Only required if Metropolis-Hasting is considered
    q_den = st.truncnorm.pdf(temp, a, b, loc = chain[k], scale = sigma_proposal) #q(theta_k+1|theta_k)
    a, b = (0.0 - temp) / sigma_proposal, (1.0 - temp) / sigma_proposal #Update the parameters of the proposal (truncated Gaussian) on the basis of the current center
    q_num = st.truncnorm.pdf(chain[k], a, b, loc = temp, scale = sigma_proposal) #q(theta_k|theta_k+1) 
    
    # Accept the new sample with probability given by rho
    rho = min(1., (target_now*q_num)/(target_before*q_den)) #Metropolis-Hasting
    #rho = min(1., target_now/target_before)               #Metropolis
    
    #sample from a uniform distribution
    u = np.random.uniform()
    
    #store the accepted samples into the chain and perform the adaptation of the stdv of the proposal
    if k <= burnin_cov: #period in which the stdv is not adapted but you accumulate samples to do that
        if u < rho: #sample accepted
            chain[k+1] = temp #store the sample within the chain
            target_before = target_now #update the target quantity of the previously accepted sample
            to_tune_cov[index_for_cov] = temp #store the accepted sample among these to be used to adapt the stdv
            index_for_cov += 1 #increase the index counting the accepted samples useful to adapt the stdv
        else: #sample rejected
            chain[k+1] = chain[k] #keep the previous sample
    elif k > burnin_cov: #period in which the stdv is adapted 
        if u < rho: # sample accepted
            if k >= burnin: # period useful to perform the Monte Carlo estimation
                naccept += 1 #increase the index counting the accepted samples useful to perform the Monte Carlo estimation
            chain[k+1] = temp #store the sample within the chain
            target_before = target_now #update the target quantity of the previously accepted sample
            to_tune_cov[index_for_cov] = temp #update the target quantity of the previously accepted sample
            index_for_cov += 1 #increase the index counting the accepted samples useful to adapt the stdv       
        else: #sample rejected
            chain[k+1] = chain[k] #keep the previous sample
            #to_tune_cov[index_for_cov] = temp #we are considering only accepted samples to update the stdv
            #index_for_cov += 1                #we are considering only accepted samples to update the stdv
        sigma_proposal = np.std(to_tune_cov[:index_for_cov]) * np.sqrt(sp) #Accelerabile!! (p.197 Andrea)
        #A voler essere rigorosi, si dovrebbe prevedere un periodo lungo cui la stdv si adatta che viene incluso nella fase di burnin.
        #Altrimenti continuare ad adattare la stdv lungo il sampling può far venir a mancare la stazionarietà della catena

np.save(work_path+'chain_'+str(i1+1), chain) #save the chain
        
clean_chain = chain[burnin::thinning] #clean the chain from the burnin period and perform thinning

#For the target, recast the value of the damage position defined in the interval (0,7.4) in terms of (x;y) coordinates
if X_HF[i1*N_obs,3]==0:   
    target = X_HF[i1*N_obs,2]/2.
    coord_x_target = 0.15 + X_HF[i1*N_obs,2] * 3.7
    coord_y_target = 0.15
else:
    target = 0.5 + X_HF[i1*N_obs,3]/2.
    coord_x_target = 3.85
    coord_y_target = 0.15 + X_HF[i1*N_obs,3] * 3.7
    
# computation of the empirical mean, mode and standard deviation 
mu_param  = np.mean(clean_chain)
mode_param = st.mode(np.round(clean_chain * 7.4 * (1/0.125), 1), axis=0, nan_policy='propagate')[0][0]/7.4 * 0.125 #In questo modo il punto in cui cade la moda viene calcolato su un intervallo di 0.25m
std_param = np.sqrt(1/(len(clean_chain)-1) * np.sum(np.square(clean_chain-mu_param)))

acceptance = naccept/(niters-burnin)*100 #acceptance ratio
    
#recast the value of the mean defined in the interval (0,1) in terms of (x;y) coordinates
if mu_param <= 0.5:
    coord_x_finded = 0.15 + mu_param*2*3.7
    coord_y_finded = 0.15
else:
    coord_x_finded = 3.85
    coord_y_finded = 0.15 + (mu_param-0.5)*2*3.7
    
#recast the value of the mode defined in the interval (0,1) in terms of (x;y) coordinates
if mode_param <= 0.5:
    mode_x_finded = 0.15 + mode_param*2*3.7
    mode_y_finded = 0.15
else:
    mode_x_finded = 3.85
    mode_y_finded = 0.15 + (mode_param-0.5)*2*3.7
    
#plot of the likelihood history over the correspondent proposed value
max_index = np.argmax(like_hist)
top = proposed_params[max_index]
fig = plt.figure(figsize = (9,6.2),dpi=100)
plt.scatter(proposed_params,like_hist)
plt.plot([top,top], [0,like_hist[max_index]])
plt.plot([target,target], [0,like_hist[max_index]])

#plot of the chain, empirical mean, mode and credibility intervals
ascissa = np.linspace(1, len(clean_chain), len(clean_chain))
fig, ax = plt.subplots(figsize = (9,7),dpi=100)
plt.plot(ascissa, clean_chain, 'goldenrod', linewidth=1)
plt.plot([1, len(clean_chain)], [target,target], 'lime', linewidth=2)
plt.plot([1, len(clean_chain)], [mu_param,mu_param], 'indigo', linewidth=2)
plt.plot([1, len(clean_chain)], [mode_param,mode_param], color='tomato', linestyle='dashed', linewidth=2)
plt.fill_between([1, len(clean_chain)], [mu_param,mu_param]-0.67449*std_param, [mu_param,mu_param]+0.67449*std_param, color='#0504aa', alpha=0.6)
plt.fill_between([1, len(clean_chain)], [mu_param,mu_param]-1.15035*std_param, [mu_param,mu_param]+1.15035*std_param, color='turquoise', alpha=0.4)
plt.ylim(0, 1.4)
plt.xlim(1, len(clean_chain))
plt.xticks([1, 200, 400, 600, 800, 1000, 1200],['1', '200', '400', '600', '800', '1000', '1200'])
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0],['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
plt.xlabel('Discrete step [-]')
plt.ylabel(r'Sample of $p(\theta_{\Omega}|\mathbf{U}^{\mathtt{EXP}}_{1,2,3}, \mathcal{NN}_{\mathtt{MF}})$ [-]')
ax.yaxis.set_label_coords(-0.09,0.36)
plt.legend(['Markov chain', r'Target value of $\theta_{\Omega}$', 'Posterior mean', 'Posterior mode', '$50\%$ confidence', '$75\%$ confidence'], ncol=2, loc='upper center')
plt.title('Burn-in = %d steps. Thinning = 1:%d. Acceptance = %d%s.' % (burnin, thinning, acceptance, '%'), fontdict={'fontfamily': 'Times New Roman'}, loc='center')
plt.savefig(work_path + 'chain_' + str(i1+1) + '.pdf')
plt.show()

#plot of histogram over the structure
step = 0.025
hist, bin_edges = np.histogram(clean_chain, bins=np.arange(0, 1, step), density=True)
coord_x = np.zeros((len(hist)))
coord_y = np.zeros((len(hist)))
coord_z = np.zeros((len(hist)))
dx = np.zeros((len(hist)))
dy = np.zeros((len(hist)))
dz = hist/hist.max()*3
for i10 in range(len(hist)):
    if i10 < len(hist)//2:
       coord_y[i10] = 0.05
       coord_x[i10] = bin_edges[i10]*7.85+0.05
       dx[i10] = step * 8 - 0.1
       dy[i10] = 0.2
    else:
       coord_y[i10] =bin_edges[i10]*7.85-3.7 + 0.05
       coord_x[i10] = 3.75
       dy[i10] = step * 8 - 0.1
       dx[i10] = 0.2     

fig = plt.figure(figsize = (9,6.2),dpi=100)
ax = plt.axes(projection='3d')
# vertices of a pyramid
v = np.array([[0,0,0], [4,0,0], [4,4,0], [3.7,4,0], [3.7,0.3,0], [0,0.3,0], [0,0,-0.4], [4,0,-0.4], [4,4,-0.4], [3.7,4,-0.4], [3.7,0.3,-0.4], [0,0.3,-0.4], [0,-0.4,0.4], [0,0.8,0.4], [0,-0.4,-0.8], [0,0.8,-0.8], [coord_x_target-0.15,coord_y_target-0.15,0],[coord_x_target-0.15,coord_y_target+0.15,0],[coord_x_target+0.15,coord_y_target-0.15,0],[coord_x_target+0.15,coord_y_target+0.15,0], [coord_x_target-0.15,coord_y_target-0.15,-0.4],[coord_x_target-0.15,coord_y_target+0.15,-0.4],[coord_x_target+0.15,coord_y_target-0.15,-0.4],[coord_x_target+0.15,coord_y_target+0.15,-0.4], [coord_x_finded-0.15,coord_y_finded-0.15,0],[coord_x_finded-0.15,coord_y_finded+0.15,0],[coord_x_finded+0.15,coord_y_finded-0.15,0],[coord_x_finded+0.15,coord_y_finded+0.15,0], [coord_x_finded-0.15,coord_y_finded-0.15,-0.4],[coord_x_finded-0.15,coord_y_finded+0.15,-0.4],[coord_x_finded+0.15,coord_y_finded-0.15,-0.4],[coord_x_finded+0.15,coord_y_finded+0.15,-0.4], [mode_x_finded-0.15,mode_y_finded-0.15,0],[mode_x_finded-0.15,mode_y_finded+0.15,0],[mode_x_finded+0.15,mode_y_finded-0.15,0],[mode_x_finded+0.15,mode_y_finded+0.15,0], [mode_x_finded-0.15,mode_y_finded-0.15,-0.4],[mode_x_finded-0.15,mode_y_finded+0.15,-0.4],[mode_x_finded+0.15,mode_y_finded-0.15,-0.4],[mode_x_finded+0.15,mode_y_finded+0.15,-0.4]])
ax.scatter3D(v[:, 0], v[:, 1], v[:, 2], s=0)
# generate list of sides' polygons of our pyramid
verts   = [ [v[0],v[1],v[7],v[6]], [v[1],v[2],v[8],v[7]], [v[2],v[3],v[9],v[8]], [v[3],v[4],v[10],v[9]], [v[4],v[5],v[11],v[10]],[v[0],v[5],v[11],v[6]],[v[0],v[1],v[2],v[3],v[4],v[5]],[v[6],v[7],v[8],v[9],v[10],v[11]]]
verts_2 = [ [v[12],v[13],v[15],v[14]]]
verts_3 = [ [v[16],v[17],v[19],v[18]], [v[20],v[21],v[23],v[22]], [v[16],v[17],v[21],v[20]], [v[16],v[18],v[22],v[20]], [v[19],v[18],v[22],v[23]], [v[19],v[17],v[21],v[23]]]
verts_4 = [ [v[24],v[25],v[27],v[26]], [v[29],v[31],v[30],v[28]], [v[24],v[25],v[29],v[28]], [v[24],v[28],v[30],v[26]], [v[27],v[26],v[30],v[31]], [v[25],v[27],v[31],v[29]]]
verts_5 = [ [v[32],v[33],v[35],v[34]], [v[37],v[39],v[38],v[36]], [v[32],v[33],v[37],v[36]], [v[32],v[36],v[38],v[34]], [v[35],v[34],v[38],v[39]], [v[33],v[35],v[39],v[37]]]
# plot sides
ax.add_collection3d(Poly3DCollection(verts, facecolors='lightsteelblue', linewidths=0.1, edgecolors='k', alpha=.25,label='_nolegend_'))
ax.add_collection3d(Poly3DCollection(verts_2, facecolors='darkslategrey', linewidths=0.5, edgecolors='k', hatch='\\\\///...'))
ax.add_collection3d(Poly3DCollection(verts_3, facecolors='lime', alpha=.2, linewidths=0.9, edgecolors='lime'))
ax.add_collection3d(Poly3DCollection(verts_4, facecolors='indigo', alpha=.35, linewidths=0.9, edgecolors='indigo'))
ax.add_collection3d(Poly3DCollection(verts_5, facecolors='tomato', alpha=.35, linewidths=0.9, edgecolors='tomato'))
ax.bar3d(coord_x, coord_y, coord_z, dx, dy, dz, color='goldenrod',  edgecolor='goldenrod', alpha=0.5,label='_nolegend_') #zsort='average', *args, **kwargs)
# make the panes transparent
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.set_zticks([])
ax.set_xlim([0,4])
ax.set_ylim([0,4])
ax.set_xlabel('[m]',labelpad=15)
ax.set_ylabel('[m]',labelpad=15)
ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.view_init(elev=40, azim=-45)
ax.grid(False)
fig.savefig(work_path + 'detect_' + str(i1+1) + '.pdf')
plt.show()