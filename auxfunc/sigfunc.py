#%%
# Imports
import sys
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import gpytorch
import pandas as pd
from scipy import stats
from datetime import datetime

# Kernel definition
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class lossEvaluation():
    def __init__(self, y, y_hat):
        self.y_hat = y_hat
        self.y = y

    # 0-1 Loss evaluation
    def getLoss(self,norm):
        if self.y.shape != self.y_hat.shape:
            assert('y and y_hat do not have the same shape')
        # compute loss 0/1 loss
        n_instances = np.size(self.y)
        loss = np.zeros(n_instances)
        if norm == 'l01':
            loss[self.y != self.y_hat] = 1
        elif norm == 'labs':
            loss = np.abs(self.y_hat-self.y)
        elif norm == 'l2':
            loss = np.power(self.y_hat-self.y,2)
        elif norm == 'res':
            loss = self.y - self.y_hat
        return loss


class critEvaluation():
    def __init__(self, norm='l01', direction='further'):
        self.norm = norm
        self.direction = direction

    def evaluate(self, y_val, y_hat_val, crit_val, rho_grid=[0.01, 0.05, 0.1, 0.15, 0.2]):
        N_val = y_val.shape[0]
        rho_grid = np.array(rho_grid)
        L_val = lossEvaluation(y_val, y_hat_val).getLoss(self.norm)
        idx_rho = np.floor(N_val*rho_grid).astype('int')
        
        rho_grid = rho_grid[idx_rho!=0]
        idx_rho = idx_rho[idx_rho!=0]
        
        d_mat = np.tile(crit_val, (idx_rho.size,1))

        L_val_mat = np.tile(L_val, (idx_rho.size,1))
        if self.direction == 'closer':
            d_sort = np.sort(crit_val)
            threshold = d_sort[idx_rho]
            
            sumL_val = np.sum(L_val_mat*(d_mat>=threshold.reshape(-1,1)), axis=1)/N_val
            error_val = np.sum(L_val_mat*(d_mat<threshold.reshape(-1,1)), axis=1)/N_val
        if self.direction == 'further':
            d_sort = np.sort(crit_val)[::-1]
            threshold = d_sort[idx_rho]
            
            sumL_val = np.sum(L_val_mat*(d_mat<=threshold.reshape(-1,1)), axis=1)/N_val
            error_val = np.sum(L_val_mat*(d_mat>threshold.reshape(-1,1)), axis=1)/N_val
        # rho_val_check = np.sum(d_sort_mat>threshold, axis=1)/N_val
        # assert(np.all(rho_val_check<=np.array(rho_grid)))

        sortL_val = np.sort(L_val, axis=0)
        sortL_val_mat = np.tile(sortL_val, (idx_rho.size,1))
        cumL_val = np.flip(np.cumsum(sortL_val_mat, axis=1)) / N_val
        # error_val = cumL_val[:,0]-sumL_val
        red_val = 100*error_val/cumL_val[:,0]
        table_val = pd.DataFrame({
            # 'rule':rule_tab,\
            #---------------- 
            'rho_user':rho_grid, \
                'error_val':np.around(error_val,decimals=4),\
                    'L_val':np.around(sumL_val,decimals=4),\
                        '%reduction_val':np.around(red_val,decimals=2),\
            # 'budget':np.around(budget_tab, decimals=2), \
                # 'error_test':np.around(error_test_tab, decimals=4),\
                #     'L_test':np.around(cum_L_test_tab,decimals=4), \
                #         '%reduction_test':np.around(reduction_test_tab,decimals=2),\
            #----------------
            'thresh':threshold
            # 'p_value':pvalue_tab,\
            # 'check':check_tab
        })
        return table_val

    def test(self, y_test, y_hat_test, crit_test, eta):
        # budget_tab = []
        # error_test_tab = []
        # cum_L_test_tab = []
        # reduction_test_tab = []
        # self.gpr_mean_test, self.gpr_var_test = self.gpr.predict(X_test)
        L_test = lossEvaluation(y_test, y_hat_test).getLoss(self.norm)
        # if self.gpr_mean_test.ndim > 1:
        #     self.gpr_mean_test = self.gpr_mean_test.numpy().ravel()
        # if self.gpr_var_test.ndim > 1:
        #     self.gpr_var_test = self.gpr_var_test.numpy().ravel()
        # rule = rule.reshape(-1,1)
        eta = eta.reshape(-1,1)
        N_test = y_test.shape[0]

        d_mat = np.tile(crit_test, (eta.size,1))
        # f_test = self.gpr_mean_test.reshape(1,-1) + rule@np.sqrt(self.gpr_var_test.reshape(1,-1))

        L_test_mat = np.tile(L_test, (eta.size,1))
        sortL_test = np.sort(L_test, axis=0)
        sortL_test_mat = np.tile(sortL_test, (eta.size,1))
        if self.direction == 'closer':
            sumL_test = np.sum(L_test_mat*(d_mat>=eta), axis=1) / N_test
            rho_test = np.sum(d_mat<eta, axis=1)/N_test
        elif self.direction == 'further':
            sumL_test = np.sum(L_test_mat*(d_mat<=eta), axis=1) / N_test
            rho_test = np.sum(d_mat>eta, axis=1)/N_test

        cumL_test = np.flip(np.cumsum(sortL_test_mat, axis=1)) / N_test
        error_test = cumL_test[:,0]-sumL_test
        red_test = error_test/cumL_test[:,0]
        table_test = pd.DataFrame({
            'budget':np.around(rho_test, decimals=2), \
                'error_test':np.around(error_test, decimals=4),\
                    'L_test':np.around(sumL_test,decimals=4), \
                        '%reduction_test':np.around(red_test*100,decimals=2)
        })
        return table_test
    

class gaussianProcess():
    def __init__(self, X_train, L_train, kernel):
        self.X_train = X_train
        self.L_train = L_train
        self.kernel = kernel
        self.model = None
        self.likelihood = None
        
    def fit(self, n_iter, lr):
        # initialize likelihood and model
        X_train_tensor = torch.Tensor(self.X_train)
        L_train_tensor = torch.Tensor(self.L_train)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = ExactGPModel(X_train_tensor, L_train_tensor, self.likelihood, self.kernel)
        if torch.cuda.is_available():
            print('initializing cuda...')
            X_train_tensor = X_train_tensor.cuda()
            L_train_tensor = L_train_tensor.cuda()
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()

        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()
        # Use the adam optimizer
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=lr)
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        print('lr={}'.format(lr)+', n_iterations={}'.format(n_iter))
        ### TensorBoard Writer Setup ###
        date = datetime.now().strftime('%H_%M_%S_%d_%m.log')
        log_name = "lr{}_{}".format(lr,date)
        writer = SummaryWriter(log_dir=f"runs/{log_name}")
        # training_iter = 1000
        for i in range(n_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(X_train_tensor)
            # Calc loss and backprop gradients
            loss = -mll(output, L_train_tensor)
            loss.backward()
            noise = self.model.likelihood.noise.item()
            if i>=(n_iter-10) and i<=n_iter:
                print('Iter %d/%d - Loss: %.3f  noise: %.3f' % (
                    i + 1, n_iter, loss.item(),
                    # self.model.covar_module.base_kernel.lengthscale.item(),
                    noise
                ))
            optimizer.step()
            writer.add_scalar('loss', loss, i)
            writer.add_scalar('noise', noise, i)
            # writer.add_scalar('L', current_loss, global_step=t)
        writer.close()
        # Get into evaluation (predictive posterior) mode
        self.model.eval()
        self.likelihood.eval()
    
    def predict(self, X_pred):
        X_pred_tensor = torch.Tensor(X_pred)
        if torch.cuda.is_available():
            print('evaluating with cuda...')
            X_pred_tensor = X_pred_tensor.cuda()

        with torch.no_grad():
            pred = self.likelihood(self.model(X_pred_tensor))
            gpr_mean_pred, gpr_var_pred = pred.mean, pred.variance
            if torch.cuda.is_available():
                gpr_mean_pred = gpr_mean_pred.cpu().numpy()
                gpr_var_pred = gpr_var_pred.cpu().numpy()
            else:
                gpr_mean_pred = gpr_mean_pred.numpy()
                gpr_var_pred = gpr_var_pred.numpy()

        return gpr_mean_pred, gpr_var_pred


class signaling():
    def __init__(self, norm='l01'):
        # Hyperparameters definition
        # self.rule_grid = rule_grid
        # self.rho_grid = rho_grid
        # Sets definition (train, test, val)
        # self.X_train = X[idx[0]]
        # self.y_train = y[idx[0]]
        # self.X_test = X[idx[1]]
        # self.y_test = y[idx[1]]
        # self.X_val = X[idx[2]]
        # self.y_val = y[idx[2]]
        # self.y_hat = y_hat # predictions
        # self.idx = idx
        # self.kernel = kernel
        self.norm = norm
        self.gpr = None


    def fit(self, X_train, y_train, y_hat_train, kernel, n_iter=50, lr=0.1):
        # Fit signaling function based on training set
        L_train = lossEvaluation(y_train, y_hat_train).getLoss(self.norm)
        # self.L_test = lossEvaluation(self.y_test, self.y_hat[self.idx[1]]).getLoss(self.norm)
        # self.L_val = lossEvaluation(self.y_val, self.y_hat[self.idx[2]]).getLoss(self.norm)
        
        if kernel == 'exponential':
            k = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5,ard_num_dims=X_train.shape[1]))
        elif kernel == 'RBF':
            k = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=X_train.shape[1]))
        elif kernel == 'eRBF':
        # RBFKernel(active_dims=torch.tensor([1])) + RBFKernel(active_dims=torch.tensor([2]))
            dims = list(range(0,X_train.shape[1]))
            print(dims)
            k_x = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=len(dims[:-1]), active_dims=dims[:-1]))
            k_y = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=len(dims[-1]), active_dims=dims[-1]))
            # k_x = gpflow.kernels.Exponential(
            #         lengthscales=[0.1]*(np.size(self.X_train,1)-1), \
            #             active_dims=dims[:-1])
            # k_y = gpflow.kernels.SquaredExponential(
                    # lengthscales=0.1,\
                        # active_dims=[dims[-1]])
            k = k_x + k_y
        # elif self.kernel == 'exCombined':
        #     dims = list(range(0,self.X_train.shape[1]))
        #     k_x = gpflow.kernels.Exponential(
        #             lengthscales=[0.1]*(np.size(self.X_train,1)-1), \
        #                 active_dims=dims[:-1])
        #     k_y = gpflow.kernels.Exponential(
        #             lengthscales=0.1,\
        #                 active_dims=[dims[-1]])
        #     k = k_x + k_y
            # k = gpflow.kernels.SquaredExponential(
                    # lengthscales=[0.1]*np.size(self.X_train,1))
        # elif self.kernel == 'RBFcombined':
        #     dims = list(range(0,self.X_train.shape[1]))
        #     k_x = gpflow.kernels.SquaredExponential(
        #             lengthscales=[0.1]*(np.size(self.X_train,1)-1), \
        #                 active_dims=dims[:-1])
        #     k_y = gpflow.kernels.SquaredExponential(
        #             lengthscales=0.1,\
        #                 active_dims=[dims[-1]])
        #     k = k_x + k_y
        self.gpr = gaussianProcess(X_train, L_train, k)
        self.gpr.fit(n_iter=n_iter,lr=lr)
        # self.gpr_mean_test, self.gpr_var_test = self.gpr.predict(self.X_test)
        # self.gpr_mean_val, self.gpr_var_val = self.gpr.predict(self.X_val)

    def evaluate(self, X_val, y_val, y_hat_val, rule_grid=[0, 1, 2, 3], rho_grid=[0.01, 0.05, 0.1, 0.15, 0.2]):
        # Compute losses and predictions with GP
        # ======================================
        # self.L_test = lossEvaluation(y_test, y_hat_test).getLoss(self.norm)
        self.L_val = lossEvaluation(y_val, y_hat_val).getLoss(self.norm)
        # self.gpr_mean_test, self.gpr_var_test = self.gpr.predict(self.X_test)
        self.gpr_mean_val, self.gpr_var_val = self.gpr.predict(X_val)
        # Plot rows
        #==========
        # ecorrected_val_plot = []
        # Lremaining_val_plot = []
        # ecorrected_test_plot = []
        # Lremaining_test_plot = []
        # rule_plot = []
        # rho_plot = []
        # budget_plot = []
        #++++++++++
        # Table rows
        #==========
        rule_tab = []
        rho_tab = []
        error_val_tab = []
        cum_L_val_tab = []
        reduction_val_tab = []
        # budget_tab = []
        # error_test_tab = []
        # cum_L_test_tab = []
        # reduction_test_tab = []
        eta_tab = []
        pvalue_tab = []
        check_tab = []

        # if self.gpr_mean_test.ndim > 1:
        #     self.gpr_mean_test = self.gpr_mean_test.numpy().ravel()
        # if self.gpr_var_test.ndim > 1:
        #     self.gpr_var_test = self.gpr_var_test.numpy().ravel()
        if self.gpr_mean_val.ndim > 1:
            self.gpr_mean_val = self.gpr_mean_val.numpy().ravel()
        if self.gpr_var_val.ndim > 1:
            self.gpr_var_val = self.gpr_var_val.numpy().ravel()
        # N_test = self.gpr_mean_test.shape[0]
        N_val = self.gpr_mean_val.shape[0]
        rule_grid = np.array(rule_grid).reshape(-1,1)
        for rho in rho_grid:
            # enough samples for user's budget?
            idx_rho = int(np.floor(rho*N_val))
            if idx_rho==0:
                continue # Not enough
            
            f_val = self.gpr_mean_val.reshape(1,-1) + rule_grid@np.sqrt(self.gpr_var_val.reshape(1,-1))
            f_val_sorted = np.flip(np.sort(f_val, axis=1), axis=1)
            η_grid = f_val_sorted[:,idx_rho].reshape(-1,1)
            L_val_mat = np.tile(self.L_val, (f_val.shape[0],1))
            sumL_val = np.sum(L_val_mat*(f_val<=η_grid), axis=1)/N_val
            error_val = np.sum(L_val_mat*(f_val>η_grid), axis=1)/N_val
            rho_val_check = np.sum(f_val>η_grid, axis=1)/N_val
            assert(np.all(rho_val_check<=rho))

            sortL_val = np.sort(self.L_val, axis=0)
            sortL_val_mat = np.tile(sortL_val, (f_val.shape[0],1))
            cumL_val = np.flip(np.cumsum(sortL_val_mat, axis=1)) / N_val
            # error_val = cumL_val[:,0]-sumL_val
            red_val = error_val/cumL_val[:,0]
 
            # Pick best rule here, and evaluate statistics
            best_idx = np.argmin(sumL_val)
            # now using only rule 
            rule = rule_grid[best_idx][0] 
            η = η_grid[best_idx][0]
            check = rho_val_check[best_idx]
            
            # Validation table
            # ------------------------
            best_f_val = f_val[best_idx,:]
            rho_val = rho
            best_error_val = error_val[best_idx]
            best_sumL_val = sumL_val[best_idx]
            best_red_val = red_val[best_idx]

            # f_val = self.gpr_mean_val + rule*np.sqrt(self.gpr_var_val)
            # sortL_val = np.sort(self.L_val, axis=0)
            # sumL_val = np.sum(self.L_val[f_val<=η]) / N_val
            
            # cumL_val = np.flip(np.cumsum(sortL_val)) / N_val
            # # error_val = cumL_val[0]-sumL_val
            # if cumL_val[0] != 0:
            #     red_val = (cumL_val[0]-sumL_val)/cumL_val[0]
            # else:
            #     red_val = 0
            # Test table
            # ------------------------
            # f_test = self.gpr_mean_test + rule*np.sqrt(self.gpr_var_test)
            # sortL_test = np.sort(self.L_test, axis=0)
            # sumL_test = np.sum(self.L_test[f_test<=η]) / N_test
            # rho_test = np.sum(f_test>η)/N_test
            # cumL_test = np.flip(np.cumsum(sortL_test)) / N_test
            # error_test = cumL_test[0]-sumL_test
            # if cumL_test[0] != 0:
            #     red_test = (cumL_test[0]-sumL_test)/cumL_test[0]
            # else:
            #     red_test = 0
            # Mann-Whitney test
            # -----------------
            if len(self.L_val[best_f_val>η])>0 and np.sum(self.L_val[best_f_val<=η]>0):
                _, p_value = stats.mannwhitneyu(self.L_val[best_f_val<=η], self.L_val[best_f_val>η], alternative='less', use_continuity=True)
            else:
                p_value = -1

            # Store table
            # -----------
            rule_tab.append(rule)
            # -----------------
            rho_tab.append(rho_val)
            error_val_tab.append(best_error_val)
            cum_L_val_tab.append(best_sumL_val)
            reduction_val_tab.append(best_red_val*100)
            # -----------------
            # budget_tab.append(rho_test)
            # error_test_tab.append(error_test)
            # cum_L_test_tab.append(sumL_test)
            # reduction_test_tab.append(red_test*100)
            # -----------------
            eta_tab.append(η)
            # -----------------
            pvalue_tab.append(p_value)
            check_tab.append(check)
        
        table_val = pd.DataFrame({
            'rule':rule_tab,\
            #---------------- 
            'rho_user':rho_tab, \
                'error_val':np.around(error_val_tab,decimals=4),\
                    'L_val':np.around(cum_L_val_tab,decimals=4),\
                        '%reduction_val':np.around(reduction_val_tab,decimals=2),\
            # 'budget':np.around(budget_tab, decimals=2), \
                # 'error_test':np.around(error_test_tab, decimals=4),\
                #     'L_test':np.around(cum_L_test_tab,decimals=4), \
                #         '%reduction_test':np.around(reduction_test_tab,decimals=2),\
            #----------------
            'eta':eta_tab,\
            'p_value':pvalue_tab,\
            'check':check_tab
        })
        # plot_variables = pd.DataFrame({
        #     'rule':rule_plot,\
        #     'rho':rho_plot,\
            # 'ec_val':ecorrected_val_plot,\
            # 'Lr_val':Lremaining_val_plot,\
            # 'budget':budget_plot,\
            # 'ec_test':ecorrected_test_plot,\
            # 'Lr_test':Lremaining_test_plot
        # })
        return table_val
    
    def test(self, X_test, y_test, y_hat_test, rule, eta):
        # budget_tab = []
        # error_test_tab = []
        # cum_L_test_tab = []
        # reduction_test_tab = []
        self.gpr_mean_test, self.gpr_var_test = self.gpr.predict(X_test)
        self.L_test = lossEvaluation(y_test, y_hat_test).getLoss(self.norm)
        if self.gpr_mean_test.ndim > 1:
            self.gpr_mean_test = self.gpr_mean_test.numpy().ravel()
        if self.gpr_var_test.ndim > 1:
            self.gpr_var_test = self.gpr_var_test.numpy().ravel()
        rule = rule.reshape(-1,1)
        eta = eta.reshape(-1,1)
        N_test = self.gpr_mean_test.shape[0]

        f_test = self.gpr_mean_test.reshape(1,-1) + rule@np.sqrt(self.gpr_var_test.reshape(1,-1))
        L_test_mat = np.tile(self.L_test, (f_test.shape[0],1))

        sortL_test = np.sort(self.L_test, axis=0)
        sortL_test_mat = np.tile(sortL_test, (f_test.shape[0],1))
        sumL_test = np.sum(L_test_mat*(f_test<=eta), axis=1) / N_test
        rho_test = np.sum(f_test>eta, axis=1)/N_test
        cumL_test = np.flip(np.cumsum(sortL_test_mat, axis=1)) / N_test
        error_test = cumL_test[:,0]-sumL_test
        red_test = (cumL_test[:,0]-sumL_test)/cumL_test[:,0]
        # if cumL_test[0] != 0:
        #     red_test = (cumL_test[:,0]-sumL_test)/cumL_test[:,0]
        # else:
        #     red_test = 0
        # budget_tab.append(rho_test)
        # error_test_tab.append(error_test)
        # cum_L_test_tab.append(sumL_test)
        # reduction_test_tab.append(red_test*100)
        table_test = pd.DataFrame({
            'budget':np.around(rho_test, decimals=2), \
                'error_test':np.around(error_test, decimals=4),\
                    'L_test':np.around(sumL_test,decimals=4), \
                        '%reduction_test':np.around(red_test*100,decimals=2)
        })
        return table_test


class patching():
    def __init__(self, norm='res'):
        self.norm = norm
        self.gpr = None
    
    def fit(self, X_train, y_train, y_hat_train, kernel, n_iter=50, lr=0.1):
        # Fit signaling function based on training set
        L_train = lossEvaluation(y_train, y_hat_train).getLoss(self.norm)
        
        if kernel == 'exponential':
            k = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5,ard_num_dims=X_train.shape[1]))
        elif kernel == 'RBF':
            k = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=X_train.shape[1]))
        elif kernel == 'eRBF':
            dims = list(range(0,X_train.shape[1]))
            # print(dims)
            k_x = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=len(dims[:-1]), active_dims=dims[:-1]))
            k_y = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=len([dims[-1]]), active_dims=[dims[-1]]))
            k = k_x + k_y
        elif kernel == 'RBF+RBF':
            dims = list(range(0,X_train.shape[1]))
            # print(dims)
            k_x = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=len(dims[:-1]), active_dims=dims[:-1]))
            k_y = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=len([dims[-1]]), active_dims=[dims[-1]]))
            k = k_x + k_y
        
        self.gpr = gaussianProcess(X_train, L_train, k)
        self.gpr.fit(n_iter=n_iter,lr=lr)
    
    def evaluate(self, X_val, y_val, y_hat_val, rule_grid=[0, 1, 2, 3], rho_grid=[0.01, 0.05, 0.1, 0.15, 0.2]):
        # Compute losses and predictions with GP
        # ======================================
        self.L_val = lossEvaluation(y_val, y_hat_val).getLoss(self.norm)
        self.gpr_mean_val, self.gpr_var_val = self.gpr.predict(X_val)
        # Table rows
        #==========
        rule_tab = []
        rho_tab = []
        error_val_tab = []
        cum_L_val_tab = []
        reduction_val_tab = []
        eta_tab = []
        pvalue_tab = []
        check_tab = []

        if self.gpr_mean_val.ndim > 1:
            self.gpr_mean_val = self.gpr_mean_val.numpy().ravel()
        if self.gpr_var_val.ndim > 1:
            self.gpr_var_val = self.gpr_var_val.numpy().ravel()

        N_val = self.gpr_mean_val.shape[0]
        rule_grid = np.array(rule_grid).reshape(-1,1)
        for rho in rho_grid:
            # enough samples for user's budget?
            idx_rho = int(np.floor(rho*N_val))
            if idx_rho==0:
                continue # Not enough
            
            f_val = self.gpr_mean_val.reshape(1,-1) + rule_grid@np.sqrt(self.gpr_var_val.reshape(1,-1))
            f_val_sorted = np.flip(np.sort(f_val, axis=1), axis=1)
            η_grid = f_val_sorted[:,idx_rho].reshape(-1,1)
            L_val_mat = np.tile(self.L_val, (f_val.shape[0],1))
            sumL_val = np.sum(L_val_mat*(f_val<=η_grid), axis=1)/N_val
            error_val = np.sum(L_val_mat*(f_val>η_grid), axis=1)/N_val
            rho_val_check = np.sum(f_val>η_grid, axis=1)/N_val
            assert(np.all(rho_val_check<=rho))

            sortL_val = np.sort(self.L_val, axis=0)
            sortL_val_mat = np.tile(sortL_val, (f_val.shape[0],1))
            cumL_val = np.flip(np.cumsum(sortL_val_mat, axis=1)) / N_val
            # error_val = cumL_val[:,0]-sumL_val
            red_val = error_val/cumL_val[:,0]
 
            # Pick best rule here, and evaluate statistics
            best_idx = np.argmin(sumL_val)
            # now using only rule 
            rule = rule_grid[best_idx][0] 
            η = η_grid[best_idx][0]
            check = rho_val_check[best_idx]
            
            # Validation table
            # ------------------------
            best_f_val = f_val[best_idx,:]
            rho_val = rho
            best_error_val = error_val[best_idx]
            best_sumL_val = sumL_val[best_idx]
            best_red_val = red_val[best_idx]

            # Mann-Whitney test
            # -----------------
            if len(self.L_val[best_f_val>η])>0 and np.sum(self.L_val[best_f_val<=η]>0):
                _, p_value = stats.mannwhitneyu(self.L_val[best_f_val<=η], self.L_val[best_f_val>η], alternative='less', use_continuity=True)
            else:
                p_value = -1

            # Store table
            # -----------
            rule_tab.append(rule)
            # -----------------
            rho_tab.append(rho_val)
            error_val_tab.append(best_error_val)
            cum_L_val_tab.append(best_sumL_val)
            reduction_val_tab.append(best_red_val*100)
            # -----------------
            eta_tab.append(η)
            # -----------------
            pvalue_tab.append(p_value)
            check_tab.append(check)
        
        table_val = pd.DataFrame({
            'rule':rule_tab,\
            #---------------- 
            'rho_user':rho_tab, \
                'error_val':np.around(error_val_tab,decimals=4),\
                    'L_val':np.around(cum_L_val_tab,decimals=4),\
                        '%reduction_val':np.around(reduction_val_tab,decimals=2),\
            #----------------
            'eta':eta_tab,\
            'p_value':pvalue_tab,\
            'check':check_tab
        })
        return table_val
    
    def test(self, X_test, y_test, y_hat_test, rule, eta):
        self.gpr_mean_test, self.gpr_var_test = self.gpr.predict(X_test)
        self.L_test = lossEvaluation(y_test, y_hat_test).getLoss(self.norm)
        if self.gpr_mean_test.ndim > 1:
            self.gpr_mean_test = self.gpr_mean_test.numpy().ravel()
        if self.gpr_var_test.ndim > 1:
            self.gpr_var_test = self.gpr_var_test.numpy().ravel()
        rule = rule.reshape(-1,1)
        eta = eta.reshape(-1,1)
        N_test = self.gpr_mean_test.shape[0]

        f_test = self.gpr_mean_test.reshape(1,-1) + rule@np.sqrt(self.gpr_var_test.reshape(1,-1))
        L_test_mat = np.tile(self.L_test, (f_test.shape[0],1))

        sortL_test = np.sort(self.L_test, axis=0)
        sortL_test_mat = np.tile(sortL_test, (f_test.shape[0],1))
        sumL_test = np.sum(L_test_mat*(f_test<=eta), axis=1) / N_test
        rho_test = np.sum(f_test>eta, axis=1)/N_test
        cumL_test = np.flip(np.cumsum(sortL_test_mat, axis=1)) / N_test
        error_test = cumL_test[:,0]-sumL_test
        red_test = (cumL_test[:,0]-sumL_test)/cumL_test[:,0]

        table_test = pd.DataFrame({
            'budget':np.around(rho_test, decimals=2), \
                'error_test':np.around(error_test, decimals=4),\
                    'L_test':np.around(sumL_test,decimals=4), \
                        '%reduction_test':np.around(red_test*100,decimals=2)
        })
        return table_test