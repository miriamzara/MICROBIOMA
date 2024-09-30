"""""""""""""""""
Created on Aug 14 2024

@author: Miriam Zara
GitHub repo: https://github.com/miriamzara/Microbioma_Project
"""
""""""""""""""""""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import statsmodels.api as sm
import pandas as pd
import os
from datetime import datetime
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity



### ALL ouputs created using routines from the classes below will be saved in a directory named "Regression"
output_dir = "Regression"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


class LV:
    """""
    The standard generalized LV, with a stochastic term.
    Growth rates are randomly sampled from a normal distribution.
    Interaction matrix is randomly sampled as symmetric, D-stable. The resulting equilibrium
    is globally stable. When not feasible, it is saturated (non invasible).
    """""
    def __init__(self, num_species, noise, sparsity):
        assert isinstance(num_species, int) and num_species > 0, "Num species must be a positive integer"
        self.num_species = num_species
        self.initial =  np.random.random_sample(self.num_species) # uniform in interval [0,1)]
        self.growth_rates = np.random.normal(loc=0, scale= 1, size = self.num_species)
        self.noise = noise
        self.sparsity = sparsity
        self.M = self.Lstable_matrix() 
    
    def Lstable_matrix(self):
        """
        Generates a symmetric Lyapunov diagonally stable matrix with controlled sparsity.
        Parameters:
        sparsity (float): Fraction of off-diagonal elements to set to zero (between 0 and 1). 
                        0 means no sparsity (fully dense matrix), and 1 means fully sparse (no off-diagonal elements).
        Returns:
        np.ndarray: A symmetric, diagonally Lyapunov stable matrix.
        """
        M = np.zeros([self.num_species, self.num_species])
        upper_triangle_indexes = np.triu_indices(self.num_species, k=1)
        M[upper_triangle_indexes] = np.random.randn(self.num_species * (self.num_species - 1) // 2)
        mask = np.random.rand(*M[upper_triangle_indexes].shape) > self.sparsity 
        #creates an array of random uniform values in [0,1).
        # mask is an array of booleans True/False. the probability that a random integer in [0, 1) is > sparsity is exactly sparsity!
        # thus, an element of the array mask is True with probability (1-sparsity) and False with probability sparsity
        # sparsity is the fraction of zero elements in the upper triangle.
        M[upper_triangle_indexes] *= mask #multiplication by a boolean value! *True= original value, *False= 0
        M += M.T
        lambda_max = np.max(np.linalg.eigvals(M))
        if lambda_max > 0:
            M = M - (lambda_max + 0.01) * np.eye(self.num_species)
        self.M = M
        return self.M

    def glv_model(self, t, s):
        dsdt = np.zeros(self.num_species)
        for i in range(self.num_species):
            if s[i] <= (0.001):
                s[i] = 0
            dsdt[i] = s[i] * ((self.growth_rates[i] + (self.M @ s)[i])) + np.sqrt(s[i]) * np.random.normal(0, 1) * self.noise
        return dsdt
    
    def simulate(self, day_max, sampling_interval, save_csv = False, save_fig = False):
        t_eval = np.arange(0, day_max, sampling_interval)
        t_span = (0, t_eval[-1])
        sol = solve_ivp(self.glv_model, t_span, self.initial, t_eval=t_eval, method='RK45')
        sol.y = np.clip(sol.y, a_min=0, a_max=None)
        #### plotting and saving ###
        colors = plt.get_cmap('tab10')
        fig, ax = plt.subplots()
        for i in range(0, self.num_species):
            ax.plot(sol.t, sol.y[i, :], color= colors(i), label = f"{i+1}")
        plt.legend()
        if save_fig:
            plt.savefig(f"fisher_setups/{self.num_species}_timeseries.pdf")
        if save_csv:
            data = np.vstack((sol.t, sol.y)).T
            num_species = sol.y.shape[0]
            columns = ["time"] + [f"species_{i+1}" for i in range(num_species)]
            solution_df = pd.DataFrame(data, columns=columns)
            solution_df.to_csv(f"fisher_setups/{self.num_species}_timeseries.csv", index=False)
        return sol.y




class Linearized_stochastic_dynamics:
    """""
    Time series data simulation from a Generalized Stochastic Lokta- Volterra model

    Class Attributes:
        interaction matrix - specific: num species, sparsity, simmetric, intra_scale, inter_scale
        time series - specific: noise, t_span, n_steps

    Class Methods:
    sample_eq_abundances
    check_lin_stability
    glv_model
    sample_interaction_matrix
    """""
    def __init__(self, num_species):
        assert isinstance(num_species, int) and num_species > 0, "Num species must be a positive integer"
        self.num_species = num_species
        self.equilibrium = None
        self.initial = None
        self.M = None

    def sample_equilibrium(self, verbose = False):
        self.equilibrium = np.random.lognormal(1, 0.5, self.num_species)
        if verbose:
            print(f"equilibrium= {self.equilibrium}")
        return self.equilibrium
    
    def sample_initial(self, vmin=0.1, vmax=1, verbose = False):
        if self.equilibrium is None:
            raise ValueError("Equilibrium values have not been initialized. Please, run self.sample_equilibrium() first.")
        self.initial = (vmax - vmin) * np.random.random_sample(self.num_species) + vmin
        if verbose:
            print(f"initial abundances= {self.initial}")
        return self.initial

    def check_exponential_stability(self, M):
        """
        Check if all eigenvalues of the matrix M have non positive real parts.
        Parameters:
        M (numpy.ndarray): A square matrix.
        Returns:
        bool: True if all eigenvalues have negative real parts, False otherwise.
        """
        assert isinstance(M, np.ndarray) and M.ndim == 2, "M must be a 2D numpy array"
        eigenvalues = np.linalg.eigvals(M)
        return np.all(np.real(eigenvalues) <= 0)
 
    def Lstable_matrix(self):
        """
        Returns a Lyapunov-diagonally stable symmetric matrix.
        This property ensures the existence and uniqueness of either a feasible or
        a saturated equilibrium point which is globally stable (all trajectories
        starting in the interior of R^n converge to the equilibrium).
        See [Allesina]
        """
        M = np.zeros([self.num_species, self.num_species])
        upper_triangle_indexes = np.triu_indices(self.num_species, k=1)
        M[upper_triangle_indexes] = np.random.randn(self.num_species * (self.num_species - 1)// 2)
        M += M.T
        lambda_max = np.max(np.linalg.eigvals(M))
        if lambda_max > 0:
            M = M - (lambda_max+ 0.01) * np.eye(self.num_species)
        self.M = M
        return self.M

    def glv_model(self, t, s, s_mean, interaction_matrix, sigma_noise = 0.):
        num_species = interaction_matrix.shape[0]
        dsdt = np.zeros(num_species)
        for i in range(0, num_species):
            if s[i] <= (0.001 * self.equilibrium[i]):
                # extinction is an absorbing state
                # whenever a species abundance is <= 0,
                # its derivative is set to zero to ensure it remains
                # at that value forever. it cannot increase anymore even if there is noise.
                # combining this with the instruction sol.y = np.clip(sol.y, a_min=0, a_max=None)
                # inside method self.simulate(), we ensure that the timeseries only contains
                # non negative values
                dsdt[i] = 0
            else:
                sum = 0
                for j in range(0, num_species):
                    sum += interaction_matrix[i, j] *  (s[j] - s_mean[j])
                dsdt[i] = sum * s[i] * (1 + np.random.normal(0, sigma_noise))
        return dsdt
    
    def simulate(self, day_max, sampling_interval, noise, save_csv = True, save_fig = True):
        """""
        Caution: the regression routine assumes a DISCRETE time lokta volterra. 
        If a continuous time dynamics is used to simulate the data, the accuracy of the fit will strongly depend
        on the chosen sampling interval. A small sampling interval (<< 1 day) should be chosen to have
        high accuracy. 
        In fact, the continuous-time and the discrete-time lokta volterra models are equivalent in the limit
        where sampling_interval -> 0.
        """""
        if self.M is None:
            raise ValueError("Interaction matrix was not initialized. Please, run self.sample_matrix() first")
        if self.initial is None: 
            raise ValueError("Initial conditions were not initialized. Please, run self.sample_initial() first")
        if self.equilibrium is None: 
            raise ValueError("Equilibrium values were not initialized. Please, run self.sample_equilibrium() first")
        assert isinstance(day_max, int) and day_max > 0, "day_max must be a positive integer"
        assert 0 <= noise, "noise must be positive"
        t_eval = np.arange(0, day_max, sampling_interval)
        t_span = (0, t_eval[-1])
        sol = solve_ivp(self.glv_model, t_span, self.initial, args=(self.equilibrium, self.M, noise), t_eval=t_eval, method='RK45')
        sol.y = np.clip(sol.y, a_min=0, a_max=None)
        colors = plt.get_cmap('tab10')
        fig, ax = plt.subplots()
        for i in range(0, self.num_species):
            ax.plot(sol.t, sol.y[i, :], color= colors(i), label = f"{i+1}")
            ax.axhline(self.equilibrium[i], color= colors(i), linestyle = 'dashed')
        plt.legend()
        if save_fig:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_path = os.path.join(output_dir, f"timeseries_{current_time}.pdf")
            plt.savefig(output_path)
            print(f"saved as {output_path}")
        if save_csv:
            data = np.vstack((sol.t, sol.y)).T
            num_species = sol.y.shape[0]
            columns = ["time"] + [f"species_{i+1}" for i in range(num_species)]
            solution_df = pd.DataFrame(data, columns=columns)
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_path = os.path.join(output_dir, f"timeseries_{current_time}.csv")
            solution_df.to_csv(f"fisher_setups/{self.num_species}_timeseries.csv", index=False)
            print(f"saved as {output_path}")
        return sol.y
    

    def simulate_discrete(self, day_max, update_interval = 0.01, noise = 0.05, repeat = 1,save_csv = True, save_fig = True):
        """""
        update_interval (float): optional, default == 1. Expressed in days. It is the timestep between two consecutive points in the
                        discrete stochastic process. The same process is obtained either by scaling update_interval while keeping fixed the interaction matrix,
                        or the opposite thing. By default, the interaction matrix entries are normalized to unity and it is suggested
                        to scale update_interval in this routine without changing the interaction matrix.
        
        repeat (float): optional, default == 1. how many instances of the stochastic process need to be computed. If more than one,
                the routine returns the plot of the mean and the standard deviation of the process. The parameter repeat is 
                allowed to be greater than 1 only when noise != 0.

        Returns

        X (numpy.array): the matrix of the process. If repeat > 1, returns the last instance of the proces.
        """""
        if self.initial is None:
            raise ValueError("Initial values not defined. run self.sample_initial() first")
        if not isinstance(repeat, (float, int)):
            raise TypeError('repeat must be a float.')
        if not isinstance(update_interval, (float, int)):
            raise TypeError('update_interval must be a float or int.')
        if repeat > 1:
            assert noise > 0, "Repetition is asked but the process is deterministic (noise == 0)."

        n_points = int(day_max / update_interval)
        times = [t * update_interval for t in range(n_points)]
        X_list = []
        for time in range(repeat):
            X = np.zeros((self.num_species, n_points))
            for i in range(self.num_species):
                X[i, 0] = self.initial[i]
            for t in np.arange(1, n_points):
                eta = np.random.lognormal(0, noise, size = self.num_species)
                for i in range(self.num_species):
                    if X[i, t - 1] <= 0.01 * self.equilibrium[i]:
                        X[i, t] = 0
                    else:
                        sum = 0
                        for j in range(self.num_species):
                            sum += self.M[i,j] * (X[j, t - 1] - self.equilibrium[j])
                        if np.isinf(update_interval * np.exp(sum)):
                            raise ValueError("Overflow error: please choose a smaller sampling interval")
                        X[i, t] = X[i, t - 1] * eta[i] * np.exp( update_interval * sum)
            X = np.clip(X, a_min = 0, a_max = None)
            ###
            ###
            X_list.append(X)
            colors = plt.get_cmap('tab10')
            fig, ax = plt.subplots(figsize = (8, 6), dpi = 300)

        if repeat == 1:
            for i in range(0, self.num_species):
                ax.plot(times, X[i, :], color= colors(i), label = f"{i+1}")
                ax.axhline(self.equilibrium[i], color= colors(i), linestyle = 'dashed')
            plt.legend()

        if repeat > 1:
            X_mean = np.mean(np.array(X_list), axis=0)
            X_std = np.std(np.array(X_list), axis=0)
            for i in range(0, self.num_species):
                ax.plot(times, X_mean[i], color= colors(i), label = f"{i+1}")
                ax.fill_between(times, y1= X_mean[i]- X_std[i], y2= X_mean[i] + X_std[i], color=colors(i), alpha=0.4)
                ax.axhline(self.equilibrium[i], color= colors(i), linestyle = 'dashed')
            plt.legend()
        if save_fig:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_path = os.path.join(output_dir, f"timeseries_{current_time}.pdf")
            plt.savefig(output_path)
            print(f"saved as {output_path}")
        if save_csv:
            data = np.vstack((times, X)).T
            num_species = X.shape[0]
            columns = ["time"] + [f"species{i+1}" for i in range(num_species)]
            solution_df = pd.DataFrame(data, columns=columns)
            solution_df.to_csv(f"fisher_setups/{self.num_species}_discrete_timeseries.csv", index=False)
            #plt.close(fig) # prevents automatic rendering of the figure in Jupiter notebook
            print(f"saved .csv in fisher_setups/{self.num_species}_discrete_timeseries.csv")
        return X




class Regression:

    """""""""""""""""""""""""""""""""
    Required parameters:
    imput_data: numpy nd array (if one subject) or list of numpy nd arrays (if more than one subject).
                The arrays have with nrows= (number of initial conditions) * (number of time points), ncols= number of species. 
                (If the array has nrows=  number of species, ncols = time points instead, parameter transpose = True
                must be passed on to self.format_data() routine.)
                The number of time points can differ across subjects. All values contained in arrays are abundances.
                The time interval separating two measures is assumed unitary (1 day). 
                All values for abundances must be non negative.
    threshold: (%) specifies how much the fit needs to improve in order to add one more regressor. high values = high sparsity, low values = low sparsity
                of the resulting interaction matrix
    bagging_iterations: number of times the forward regression is performed. if >1, the median of the inferrede matrices is returned
    """""""""""""""""""""""""""""""""

    def __init__(self, input_data, transpose, method, sampling, threshold, bagging_iterations,  verbose = False):
        if not method in ['logarithm', 'derivative']:
            raise ValueError("method must be one of 'logarithm' or 'derivative'")
        if isinstance(input_data, np.ndarray) and np.any(input_data < 0):
            raise ValueError( "Invalid timeseries: contains negative values")
        if not isinstance(transpose, bool):
            raise TypeError( "Expected a boolean value for argument transpose")
        if isinstance(input_data, list):
            for n in range(len(input_data)):
                if np.any(input_data[n] < 0):
                    input_data[n] = np.clip(input_data[n], a_min= 0, a_max= None)
                    #raise ValueError( "Invalid timeseries: contains negative values")
        self.input_data = input_data.copy()
        self.num_species = None
        assert isinstance(verbose, bool), "Verbose must be a boolean value"
        self.verbose = verbose
        assert 0 <= threshold <= 100, "Threshold must be a percentage"
        self.threshold = threshold
        assert isinstance(bagging_iterations, int), "Bagging iterations must be positive integer"
        self.iterations = bagging_iterations
        self.data = None
        self.format_data(transpose= transpose, method = method, sampling = sampling)

    def format_data(self, transpose, method, sampling):
        """
        Arguments:
        Computes the variables needed to perform linear regression. Deletes all imput_data rows containing near zero values (because log is undefined)
        Stores as numpy nd array  self.data, with nrows = timesteps - 1 - deleted rows , ncols = 2 * num_species
        Caution: some covariate columns may contain inf and nan values. This happens when the species is extinct.
        These rows are deleted when regression over the corresponding species is performed.
        """
        if not isinstance(self.input_data, (list, np.ndarray)):
            raise ValueError("Invalid type for input data: must be a numpy ndarray or list of ndarrays")
        
        if isinstance(self.input_data, np.ndarray):
            # wrap inside a list
            self.input_data = [self.input_data]
        # from now on we only work with list
        if transpose == True:
            for i in range(len(self.input_data)):
                self.input_data[i] = np.transpose(self.input_data[i]) #now shape is T x N
        self.num_species = self.input_data[0].shape[1]
        if method == 'logarithm':
            self.data = self.format_data_subroutine(self.input_data[0])
            for n in range(1, len(self.input_data)):
                self.data = np.vstack((self.data, self.format_data_subroutine(self.input_data[n])))
        if method == 'derivative':
            self.data = self.format_data_subroutine_try(self.input_data[0], sampling = sampling)
            for n in range(1, len(self.input_data)):
                self.data = np.vstack((self.data, self.format_data_subroutine_try(self.input_data[n], sampling = sampling)))
        return self.data, self.num_species


    def format_data_subroutine(self, ndarray):
        log_data = np.zeros(shape= (ndarray.shape[0] - 1, ndarray.shape[1]))
        for t in np.arange(0, ndarray.shape[0] - 1):
            for i in range(ndarray.shape[1]):
                log_data[t, i] = np.log(ndarray[t + 1, i]) - np.log(ndarray[t , i])
        medians = np.median(ndarray[:, :self.num_species], axis = 0)
        for i, m in enumerate(medians):
            ndarray[:, i] = ndarray[:, i] - m
        output_data = np.hstack((ndarray[:-1, :], log_data))
        return output_data


    def format_data_subroutine_try(self, ndarray, sampling):
        medians = np.median(ndarray[:, :self.num_species], axis = 0)
        for i, m in enumerate(medians):
            ndarray[:, i] = ndarray[:, i] - m
        ####
        #dX_dt = np.diff(ndarray, axis=0) / sampling #derivatives
        #X = ndarray[:-1, :]
        ####
        dX_dt = np.zeros_like(ndarray[:-2, :])  # Shape will be (T-2, N)
        for i in range(1, len(ndarray) - 1):
            dX_dt[i-1, :] = (ndarray[i+1, :] - ndarray[i-1, :]) / (2 * sampling)

        X = ndarray[:-2, :]
        output_data = np.hstack((X, dX_dt))
        return output_data

    def split_data(self, data):
        """
        Arguments:
        data: numpy.ndarray with nrows= time points, ncolums= number of species (regressors) + 1 (covariate)
        Splits the formatted data into training and test datasets.
        Returns:
            training_data (np.ndarray): The training data set.
            test_data (np.ndarray): The test data set.
        """
        if len(data) < 2:
            raise ValueError("Invalid data provided in split_data()")
        split_point = len(data) // 2 #equivalent to np.floor(len(data) / 2)
        permuted_indices = np.random.permutation(len(data))
        training_data = data[permuted_indices[: split_point], :]
        test_data = data[permuted_indices[split_point :], :]
        return training_data, test_data
    

    def row_fw_regression(self, i, iter = 1):
        """
        i: dependent species index (the row of interaction matrix we want to infer)
        admits values 1, ... N 
        threshold: percentage on the error reduction. Suggested values are  [0- 5]%
        """
        assert i in np.arange(1, self.num_species + 1, dtype=int),  "Invalid value for i"

        covariate_col_idx = self.num_species + (i - 1)
        selected_cols = np.append(np.arange(0, self.num_species), covariate_col_idx)
        species_data = self.data[:, selected_cols]


        inf_mask = np.isinf(species_data)
        nan_mask = np.isnan(species_data)
        rows_with_inf = np.any(inf_mask, axis=1)
        rows_with_nan = np.any(nan_mask, axis=1)
        species_data = species_data[~rows_with_inf]
        species_data = species_data[~rows_with_nan]
        if len(species_data) == 0:
            raise ValueError("Some species is extinct right from the start. Please compute initial conditions again.")

        training_data, test_data = self.split_data(species_data)

        training_design_matrix = training_data[:, :-1]
        training_covariate = training_data[:, -1]

        test_design_matrix = test_data[:, :-1]
        test_covariate = test_data[:, -1]

        possible_regressors = np.arange(0, self.num_species, dtype= int)
        regressors = np.array([], dtype = int)
        if self.verbose:
            print("====================================================================")
            print(f" Inferring interactions for species  {i}, bagging iteration {iter}      ")
            print("============================================================================")
            print("number of regressors,     regressors,          c_ij ,       RMSE/covariate_mean     ")
        error = 1
        while True:
            if error == 0.001 or len(possible_regressors) < 2:
                # when only one possible regressor remains, the design matrix is singular 
                # (because of compositionality constraint)
                # and the least squares problem becomes ill defined
                break
            else:
                temp_percentage_dict = {}
                temp_error_dict = {}
                for j in possible_regressors:
                    temp_regressors = np.append(regressors, j)
                    temp_res = sm.OLS(training_covariate, training_design_matrix[:, temp_regressors]).fit()
                    temp_c = temp_res.params
                    temp_rmse = np.sqrt(np.mean((test_covariate - np.dot(test_design_matrix[:, temp_regressors], temp_c))**2))
                    
                    temp_error = temp_rmse / np.mean(test_covariate) 
                    temp_percentage = ((error - temp_error) / error ) * 100
                    temp_percentage_dict[j] = temp_percentage
                    temp_error_dict[j] = temp_error

                if max(temp_percentage_dict.values()) > self.threshold or len(possible_regressors) == self.num_species:
                    selected_j = int(max(temp_percentage_dict, key= lambda k: temp_percentage_dict[k]))
                    regressors = np.append(regressors, selected_j)
                    index = np.where(possible_regressors == selected_j)[0][0]
                    possible_regressors = np.delete(possible_regressors, index)
                    c = sm.OLS(training_covariate, training_design_matrix[:, regressors]).fit().params.tolist()
                    error = temp_error_dict[selected_j]
                    if self.verbose:
                        print("-------------------------------------------------------------------")
                        for i,reg in enumerate(regressors):
                            print("%-8.0f                   %-8.0f        %8.5f         %8.5f" % (len(regressors), reg, c[i], error))
                else:
                    #print(f"Threshold on model improvement has been met")
                    break
            
        #Final estimate of parameters
        results = sm.OLS(training_covariate, training_design_matrix[:, regressors]).fit()
        parameters = results.params
        std_errors = results.bse
        sorted_indices = np.argsort(regressors)
        sorted_regressors = regressors[sorted_indices]
        sorted_parameters = parameters[sorted_indices]
        sorted_std_errors = std_errors[sorted_indices]
        
        if self.verbose:
            print("=====================================================================")
            formatted_estimates = ', '.join(
                [f"c({i+1}, {j + 1}) = {sorted_parameters[k]:.2f} ± {sorted_std_errors[k]:.2f}" for k,j in enumerate(regressors)]
                )
            print(f"Final estimates: {formatted_estimates}")
            print("                                                        ")
        return sorted_regressors, sorted_parameters
    
    def fw_regression(self, iter = 1):
        M_estimate = np.zeros((self.num_species, self.num_species))
        for i in np.arange(0, self.num_species):
            regressors, parameters = self.row_fw_regression(i+1, iter)
            for j, r in enumerate(regressors):
                M_estimate[i, r] = parameters[j] 
        return M_estimate

    def LIMITS(self):
        matrices = np.zeros((self.iterations, self.num_species, self.num_species))
        for iter in range(self.iterations):
            estimate_matrix = self.fw_regression(iter= iter)
            for j in range(self.num_species):
                for k in range(self.num_species):
                    matrices[iter, j, k] = estimate_matrix[j, k]
        median_matrix = np.median(matrices, axis=0)



        return median_matrix 



class Matrix_Comparison():   

    def compare_matrices(self, M, M_estimate, normalize = True, plot = False, savefig = False):
        if normalize:
            M = M / np.max(np.abs(M))
            M_estimate = M_estimate / np.max(np.abs(M_estimate))
        true_interactions = M.flatten()
        estimated_interactions = M_estimate.flatten()
        results = sm.OLS(true_interactions, estimated_interactions).fit()
        slope = results.params[0]
        R_squared = results.rsquared
        if plot:
            fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (30, 10))
            sns.heatmap(M, ax = ax1, annot= True, cmap='coolwarm', cbar=True, fmt=".2f")
            ax1.set_title("Real Interaction matrix")
            sns.heatmap(M_estimate, ax = ax2, annot=True,  cmap='coolwarm', cbar=True, fmt=".2f")
            ax2.set_title("Inferred Interaction matrix")
            ax3.scatter(true_interactions, estimated_interactions, label = f"slope = {slope:.3f}")
            ax3.plot(true_interactions, slope * true_interactions, color = 'black', linewidth = 1, label = fr"$R^2$ = {R_squared:.3f}")
            ax3.grid()
            ax3.set_xlabel("True Interactions")
            ax3.set_ylabel("Inferred Interactions")
            ax3.legend()
        if savefig:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_path = os.path.join(output_dir, f"matrix_comparison_{current_time}.pdf")
            plt.savefig(output_path)
            print(f"saved as {output_path}")
        return R_squared, slope
    

    def compute_error_rates(self, M, M_estimate, print_results = False):
        """"
        Returns 

        Information about presence/absence of interaction
        - type_I_rate  
            % of type I errors (false positive): a false positive error is made when
              a species pair is identified as interacting when it is not
        - type_II_rate
            % of type II errors (false negatives): a species pair is identified as non- interacting 
            when in reality it is

        Information about interaction sign
        -sign error rate
            % of true interactions inferred with the correct sign
        """

        true_interactions = M.flatten()
        inferred_interactions = M_estimate.flatten()

        ## type I and II error rates ##
        num_true_interactions = np.sum(np.abs(true_interactions) > 0)
        num_inferred_interactions = np.sum(np.abs(inferred_interactions) > 0)
        type_I_errors = 0
        type_II_errors = 0
        sign_errors = 0
        for true, estimated in zip(true_interactions, inferred_interactions):
            if np.abs(true) == 0 and np.abs(estimated) > 0:
                type_I_errors += 1
            elif np.abs(true) > 0 and np.abs(estimated) == 0:
                type_II_errors += 1
            #now handle the case where the presence of interaction is spotted
            # is the sign also correct?
            elif (np.abs(true) * np.abs(estimated)) > 0 and (true * estimated) < 0:
                sign_errors += 1

        type_I_rate = ( type_I_errors / num_inferred_interactions ) * 100
        type_II_rate = (type_II_errors / num_true_interactions ) * 100
        sign_errors_rate = (sign_errors / num_inferred_interactions) * 100
        if print_results:
            print(f"Type I = {type_I_rate} %, Type II = {type_II_rate}, Wrong sign = {sign_errors_rate} %")
        return [type_I_rate, type_II_rate, sign_errors_rate]


    def root_mean_squared_error(self, A, B):
        RMSE = np.sqrt(np.mean((A - B) ** 2))/np.mean(A)
        return RMSE


    def pearson_correlation(self, A, B):
        """
        measures the linear correlation (when A_ij larger than mean,
        is also B_ij larger than mean?)
        ranges in [-1, +1]
        """
        A_flat = A.flatten()
        B_flat = B.flatten()
        correlation, _ = pearsonr(A_flat, B_flat)
        return correlation

    def relative_error_frobenius_norm(self, A, B):
        """
        measures the euclidean distance between the matrices 
        as usually done for vectors
        ranges in [0, inf)
        """
        frob = np.linalg.norm(A - B, 'fro')
        return frob/np.linalg.norm(A, 'fro')

    def cosine_similarity_matrix(self, A, B):
        """
        measures cosine between flattened matrices
        ranges in [-1 (most dissimilar), +1 (most similar)]
        """
        # Flatten the matrices
        A_flat = A.flatten().reshape(1, -1)
        B_flat = B.flatten().reshape(1, -1)
        # Compute cosine similarity
        return cosine_similarity(A_flat, B_flat)[0][0]
    
    def jaccard_index_weighted(self, A, B):
        """
        how much of the total weight is shared between the
        two graphs
        ranges from 0 (no similarity) to 1 (max similarity)
        min_weights, max_weights are scalars
        """
        min_weights = np.minimum(A, B) #f both matrices have 
        #a non-zero weight for a given edge, 
        #the lower of the two weights is taken.
        max_weights = np.maximum(A, B)
        # the total sum of the weights
        return np.sum(min_weights) / np.sum(max_weights)