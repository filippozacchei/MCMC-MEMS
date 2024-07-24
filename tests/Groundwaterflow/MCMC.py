import numpy as np
from numpy import trace
from numpy.linalg import det, inv

from scipy.linalg import sqrtm

from utils import *

#from importlib import reload; import utils; reload(utils); from utils import *

class Link:
    
    '''
    The Link class contains all the attributes used by the MCMC.
    There are also methods to compute priors, likelihoods and the (unnormalised) posterior.
    '''
    
    def __init__(self, parameters, output, data, prior_location, prior_scale, likelihood_scale, xi = None, qoi = None):
        
        # Set parameters.
        self.parameters = parameters
        self.output = output
        self.data = data
        self.prior_location = prior_location
        self.prior_scale = prior_scale
        self.likelihood_scale = likelihood_scale
        self.xi = xi
        self.qoi = qoi
        
        # Run the Link-methods to compute likelihoods.
        self.prior = self.compute_prior()
        self.likelihood = self.compute_likelihood()
        self.posterior = self.compute_posterior()
    
    def compute_prior(self):
        # Multivariate lognormal distribution.
        return np.log(1/np.sqrt((2*np.pi)**len(self.parameters)*det(np.diag(self.prior_scale)))) \
            - 0.5*np.linalg.multi_dot((self.parameters - self.prior_location, inv(np.diag(self.prior_scale)), self.parameters - self.prior_location))
    
    def compute_likelihood(self):
        # Gaussian log-likelihood.
        return -np.linalg.norm(self.output - self.data)**2/(2*self.likelihood_scale)

    def compute_posterior(self):
        # Compute the posterior.
        return self.prior + self.likelihood



# Chain implements the preconditioned Crank-Nicolson proposal distribution and standard Metropolis-Hastings MCMC.
class MFChain:
    def __init__(self, 
                 model_coarse,
                 model_fine,
                 data,
                 prior_location, 
                 prior_scale, 
                 likelihood_scale, 
                 proposal_scale,
                 beta_pCN,
                 acceptance_delay = 1):
        """ 
        model_coarse:       Coarse level model 
        model_fine:         Fine level model that recieves in input = (parameters, model_coarse(paramaters))
        data:               the data observed 
        prior_location :    mean of the prior which is a normal
        prior_scale :       standard deviation of the Normal prior
        likelihood_scale :  a list of the scales of the likelyhood coefficients 
        proposal_scale:     scaling of the proposal  
        beta_pCN: beta      paramter of the Crank-Nicolson 
        acceptance_delay :  Subsampling rate of the MDA 
        """
        
        # Create separate attributes for the fine and coarse models.
        self.model_c = model_coarse
        self.model_f = model_fine
        
        # Set the modal cutoff for the reduced-order model.
        #self.mkl_cutoff = self.model_f.mkl - self.model_c.mkl
        
        # Internalise the data.
        self.data = data
        
        # Set some parameters relevant to likelihood-computations
        self.prior_location = prior_location
        self.prior_scale = prior_scale
        self.likelihood_scale_c = likelihood_scale[0]
        self.likelihood_scale_f = likelihood_scale[1]
        
        # Set the covariance matrix of the proposals.
        self.SIGMA_c = np.diag(proposal_scale * np.ones(prior_location.shape[0]))
        #self.SIGMA_f = np.diag(proposal_scale[])
        
        # Internalise the pCN beta coefficients.
        self.beta_pCN_c = beta_pCN
        #self.beta_pCN_f = beta_pCN[1]
            
        # Internalise the delay.
        self.acceptance_delay = acceptance_delay
        
        ##### Initialise a first link, based on a guess from the proir. ####
        # Create an initial random vector from the prior.
        self.initial_parameters_c = np.random.normal(self.prior_location, self.prior_scale)
        #self.initial_parameters_f = np.random.normal(self.prior_location, self.prior_scale[self.mkl_cutoff:])
        
        # Create a chain and put the first link in it.
        self.chain_c = [self.new_link_coarse(self.initial_parameters_c)]
        self.state = self.chain_c[0]
        
        # Create a list for acceptance budgeting.
        self.acceptance_c = [1]
        
    def new_link_coarse(self, parameters, xi = None):
        
        # Solve the problem given the input vector.
        output = self.model_c(parameters)
        
        return Link(parameters, output, self.data, 
                    self.prior_location, 
                    self.prior_scale, 
                    self.likelihood_scale_c, xi, qoi=output)
    
    def new_link_fine(self, parameters, xi = None):

        # Just a proxy-method to allow easy creation of links.        
        # Solve the problem given the input vector.
        self.model_f.solve(parameters)
        output = self.model_f.get_data(self.data[:,0:2])
        qoi = self.model_f.get_outflow()
        
        # Create a chain and put the first link in it.
        return Link(parameters, output, self.data[:,2], 
                     self.prior_location, self.prior_scale, 
                     self.likelihood_scale_f, xi, qoi)
        
    # preconditioned Crank Nicolson (pCN) proposal.
    def propose_pCN(self, parameters, beta, SIGMA):
        xi = np.random.multivariate_normal(np.zeros(len(parameters)), SIGMA)
        return np.sqrt(1 - beta**2)*parameters + beta*xi, xi
    
    # This method implements the (adaptive pCN) Delayed Acceptance MCMC algorithm
    # and extends the chain according to the number of iterations
    def run(self, chain_length_fine, burnin_fine = 0, burnin_coarse = 0):
        
        # Run only the coarse modes for a while to allow them to converge.
        for i in range(burnin_coarse):
            
            # Propose some new coarse parameters given the last link
            new_parameters_c, xi_c = self.propose_pCN(self.chain_c[-1].parameters, 
                                                      self.beta_pCN_c,
                                                      self.SIGMA_c)
            
            # Create a link from the parameter proposal.
            new_proposal_c = self.new_link_coarse(new_parameters_c)
            
            # Compute the probability of accepting the new coarse link.
            alpha = min(1, np.exp(new_proposal_c.likelihood - self.chain_c[-1].likelihood))
                
            # If succesful, add the new link to both chains.
            if np.random.random() < alpha:
                self.state = new_proposal_c
                self.chain_c.append(self.state)
                self.acceptance_c.append(1)
            
            # If failed, go back to the last accepted fine link on both chains.
            else:
                self.chain_c.append(self.state)
                self.acceptance_c.append(0) 
        
        # Create a chain and put the first link in it.
        parameters_fine = np.hstack((self.chain_c[-1].parameters, self.model_c(self.chain_c[-1].parameters)))
        output_fine = self.model_f(parameters_fine)
        first_link_fine = Link(self.chain_c[-1].parameters, output_fine, self.data, 
                     self.prior_location, self.prior_scale, 
                     self.likelihood_scale_f, qoi=output_fine)
        self.chain_f = [first_link_fine] # ???? Perche f?
        #self.chain_f = [self.new_link_fine(np.hstack((self.chain_c[-1].parameters, self.initial_parameters_f)))]  # OLD
        # Create a list for acceptance budgeting.
        self.acceptance_f = [1]
        
        # Run the delayed acceptance for a number of iterations.
        while len(self.acceptance_f) < burnin_fine + chain_length_fine:
            
            # Propose some new coarse parameters given the last link
            new_parameters_c, xi_c = self.propose_pCN(self.chain_c[-1].parameters, 
                                                      self.beta_pCN_c,
                                                      self.SIGMA_c)
            
            # Create a link from the parameter proposal.
            new_proposal_c = self.new_link_coarse(new_parameters_c)
            
            # Compute the probability of accepting the new coarse link.
            alpha_1 = min(1, np.exp(new_proposal_c.likelihood - self.chain_c[-1].likelihood))
            
            # Check at the coarse levels
            if np.random.random() < alpha_1:
                
                # If the delay condition has been met, evaluate at the fine level.
                if len(self.acceptance_f) <= burnin_fine or sum(self.acceptance_c)%self.acceptance_delay == 0:
                    
                    # Create a proposal form the fine modes.
                    #new_parameters_f, xi_f = self.propose_pCN(self.chain_f[-1].parameters[self.mkl_cutoff:], 
                    #                                          self.beta_pCN_f,
                    #                                          self.SIGMA_f)

                    # Evaluate at the fine level
                    parameters_fine = np.hstack((new_parameters_c, new_proposal_c.output)) ## vedi qua 
                    output_fine = self.model_f(parameters_fine)
                    new_proposal_f = Link(new_parameters_c, output_fine, self.data, 
                                          self.prior_location, self.prior_scale, 
                                            self.likelihood_scale_f, qoi=output_fine)
                    # Compute the probability of accepting the new fine link.
                    alpha_2 = min(1, np.exp(new_proposal_f.likelihood - self.chain_f[-1].likelihood + \
                                            self.state.likelihood - new_proposal_c.likelihood))
                    #print('New parameter proposed',new_parameters_c, '\nOutput of the model:', output_fine, '\ndata:', self.data, '\nalpha2',alpha_2 )
                    #print(alpha_2)
                    # If succesful, add the new link to both chains.
                    if np.random.random() < alpha_2:

                        #print('ADDED A NUMBER to fine with alpha2' , alpha_2 )
                        self.state = new_proposal_c
                        self.chain_c.append(self.state)
                        self.acceptance_c.append(1)
                        
                        self.chain_f.append(new_proposal_f)
                        self.acceptance_f.append(1)                         
                    
                    # If failed, go back to the last accepted fine link on both chains.
                    else:
                        self.chain_c.append(self.state)
                        self.acceptance_c.append(1) 
                        
                        self.chain_f.append(self.chain_f[-1])
                        self.acceptance_f.append(0)
                
                # If the delay condition was not met.
                else:
                    self.chain_c.append(new_proposal_c)
                    self.acceptance_c.append(1)
                
            # If the coarse link was not accepted.
            else:
                self.chain_c.append(self.chain_c[-1])
                self.acceptance_c.append(0)
