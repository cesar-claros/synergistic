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
import urllib.request

ts_code  = 'https://raw.githubusercontent.com/google/TrustScore/master/trustscore.py'
ts_req     = urllib.request.urlopen(ts_code)
read_req = ts_req.read()
exec(read_req)

# Trust score adaptation for python3 (xrange)
class trust_score(TrustScore):
    def __init__(self,k=10, alpha=0., filtering="none", min_dist=1e-12):
        super().__init__(k,alpha,filtering,min_dist)
    def fit(self, X, y):
      """Initialize trust score precomputations with training data.
      WARNING: assumes that the labels are 0-indexed (i.e.
      0, 1,..., n_labels-1).
      Args:
      X: an array of sample points.
      y: corresponding labels.
      """
      self.n_labels = np.max(y) + 1
      self.kdtrees = [None] * self.n_labels
      if self.filtering == "uncertainty":
        X_filtered, y_filtered = self.filter_by_uncertainty(X, y)
      for label in range(self.n_labels):
        if self.filtering == "none":
          X_to_use = X[np.where(y == label)[0]]
          self.kdtrees[label] = KDTree(X_to_use)
        elif self.filtering == "density":
          X_to_use = self.filter_by_density(X[np.where(y == label)[0]])
          self.kdtrees[label] = KDTree(X_to_use)
        elif self.filtering == "uncertainty":
          X_to_use = X_filtered[np.where(y_filtered == label)[0]]
          self.kdtrees[label] = KDTree(X_to_use)

        if len(X_to_use) == 0:
          print("Filtered too much or missing examples from a label! Please lower alpha or check data.")

    def get_score(self, X, y_pred):
      """Compute the trust scores.
      Given a set of points, determines the distance to each class.
      Args:
      X: an array of sample points.
      y_pred: The predicted labels for these points.
      Returns:
      The trust score, which is ratio of distance to closest class that was not
      the predicted class to the distance to the predicted class.
      """
      d = np.tile(None, (X.shape[0], self.n_labels))
      for label_idx in range(self.n_labels):
        d[:, label_idx] = self.kdtrees[label_idx].query(X, k=2)[0][:, -1]

      sorted_d = np.sort(d, axis=1)
      d_to_pred = d[range(d.shape[0]), y_pred]
      d_to_closest_not_pred = np.where(sorted_d[:, 0] != d_to_pred,
                                      sorted_d[:, 0], sorted_d[:, 1])
      return d_to_closest_not_pred / (d_to_pred + self.min_dist)

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
        self.norm = norm
        self.gpr = None

    def fit(self, X_train, y_train, y_hat_train, kernel, n_iter=50, lr=0.1, ex_dim=1):
        # Fit signaling function based on training set
        L_train = lossEvaluation(y_train, y_hat_train).getLoss(self.norm)
        dims = list(range(0,X_train.shape[1]))
        active_dims_x = np.array(dims[:-ex_dim])
        active_dims_y = np.array(dims[-ex_dim:])

        if kernel == 'exponential':
            k = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5,ard_num_dims=X_train.shape[1]))
        elif kernel == 'RBF':
            k = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=X_train.shape[1]))
        elif kernel == 'e+RBF':
            # dims = list(range(0,X_train.shape[1]))
            # print(dims)
            k_x = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=len(active_dims_x), active_dims=active_dims_x))
            k_y = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=len(active_dims_y), active_dims=active_dims_y))
            k = k_x + k_y
        elif kernel == 'e*RBF':
            # dims = list(range(0,X_train.shape[1]))
            # print(dims)
            k_x = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=len(active_dims_x), active_dims=active_dims_x))
            k_y = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=len(active_dims_y), active_dims=active_dims_y))
            k = k_x * k_y
        elif kernel == 'e+e':
            # dims = list(range(0,X_train.shape[1]))
            # print(dims)
            k_x = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=len(active_dims_x), active_dims=active_dims_x))
            k_y = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=len(active_dims_y), active_dims=active_dims_y))
            k = k_x + k_y
        elif kernel == 'e*e':
            # dims = list(range(0,X_train.shape[1]))
            # print(dims)
            k_x = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=len(active_dims_x), active_dims=active_dims_x))
            k_y = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=len(active_dims_y), active_dims=active_dims_y))
            k = k_x * k_y
        elif kernel == 'RBF+RBF':
            # dims = list(range(0,X_train.shape[1]))
            # print(dims)
            k_x = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=len(active_dims_x), active_dims=active_dims_x))
            k_y = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=len(active_dims_y), active_dims=active_dims_y))
            k = k_x + k_y
        elif kernel == 'RBF*RBF':
            # dims = list(range(0,X_train.shape[1]))
            # print(dims)
            k_x = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=len(active_dims_x), active_dims=active_dims_x))
            k_y = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=len(active_dims_y), active_dims=active_dims_y))
            k = k_x * k_y
        
        self.gpr = gaussianProcess(X_train, L_train, k)
        self.gpr.fit(n_iter=n_iter,lr=lr)

    def evaluate(self, X_val, y_val, y_hat_val, rule_grid=[0, 1, 2, 3], rho_grid=[0.01, 0.05, 0.1, 0.15, 0.2]):
        # Compute losses and predictions with GP
        # ======================================
        self.L_val = lossEvaluation(y_val, y_hat_val).getLoss(self.norm)
        self.gpr_mean_val, self.gpr_var_val = self.gpr.predict(X_val)
        #++++++++++
        # Table rows
        #==========
        rule_tab = []
        rho_tab = []
        # error_val_tab = []
        # cum_L_val_tab = []
        corrected_val_tab = []
        queries_val_tab = []
        loss_red_val_tab = []
        eta_tab = []
        pvalue_tab = []
        total_wrong_val_tab = []
        # check_tab = []

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
            # sumL_val = np.sum(L_val_mat*(f_val<=η_grid), axis=1)/N_val
            corrected_val = np.sum(L_val_mat*(f_val>η_grid), axis=1)
            queries_val = np.sum(f_val>η_grid, axis=1)
            total_wrong_val = np.sum(self.L_val)
            assert(np.all(queries_val/N_val<=rho))

            # sortL_val = np.sort(self.L_val, axis=0)
            # sortL_val_mat = np.tile(sortL_val, (f_val.shape[0],1))
            loss_red_val = corrected_val/total_wrong_val
 
            # Pick best rule here, and evaluate statistics
            best_idx = np.argmax(corrected_val)
            # now using only rule 
            rule = rule_grid[best_idx][0] 
            η = η_grid[best_idx][0]
            best_queries_val = queries_val[best_idx]
            
            # Validation table
            # ------------------------
            best_f_val = f_val[best_idx,:]
            rho_val = rho
            best_corrected_val = corrected_val[best_idx]
            # best_sumL_val = sumL_val[best_idx]
            best_loss_red_val = loss_red_val[best_idx]

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
            corrected_val_tab.append(best_corrected_val)
            # cum_L_val_tab.append(best_sumL_val)
            loss_red_val_tab.append(best_loss_red_val*100)
            # -----------------
            eta_tab.append(η)
            # -----------------
            pvalue_tab.append(p_value)
            queries_val_tab.append(best_queries_val)
            total_wrong_val_tab.append(total_wrong_val)
        
        table_val = pd.DataFrame({
            'rule':rule_tab,\
            'rho_user':rho_tab, \
            #----------------
            'corrected_val':np.around(corrected_val_tab,decimals=2),\
            'queries_val':queries_val_tab,\
            'total_wrong_val':total_wrong_val_tab,\
            'loss_query_val':np.around(np.array(corrected_val_tab)/np.array(queries_val_tab),decimals=2),\
            #---------------- 
            'rho_hat_val':np.around(np.array(queries_val_tab)/N_val,decimals=2),\
            '%loss_red_val':np.around(loss_red_val_tab,decimals=2),\
            #----------------
            'eta':eta_tab,\
            'p_value':pvalue_tab
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

        queries_test = np.sum(f_test>eta, axis=1) 
        rho_hat_test = queries_test/N_test
        total_wrong_test = np.sum(L_test_mat, axis=1)
        corrected_test = np.sum(L_test_mat*(f_test>eta), axis=1)
        loss_red_test = corrected_test/total_wrong_test

        table_test = pd.DataFrame({
            'corrected_test':np.around(corrected_test, decimals=2),\
            'queries_test':np.around(queries_test, decimals=2),\
            'total_wrong_test':total_wrong_test,\
            'loss_query_test':np.around(np.array(corrected_test)/np.array(queries_test),decimals=2),\
            'rho_hat_test':np.around(rho_hat_test, decimals=2), \
            '%loss_red_test':np.around(loss_red_test*100,decimals=2)
        })
        return table_test

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
        
        crit_mat = np.tile(crit_val, (idx_rho.size,1))

        L_val_mat = np.tile(L_val, (idx_rho.size,1))
        if self.direction == 'closer':
            crit_sort = np.sort(crit_val)
            threshold = crit_sort[idx_rho]
            corrected_val = np.sum(L_val_mat*(crit_mat<threshold.reshape(-1,1)), axis=1)
            queries_val = np.sum(crit_mat<threshold.reshape(-1,1), axis=1)

        elif self.direction == 'further':
            crit_sort = np.sort(crit_val)[::-1]
            threshold = crit_sort[idx_rho]
            corrected_val = np.sum(L_val_mat*(crit_mat>threshold.reshape(-1,1)), axis=1)
            queries_val = np.sum(crit_mat>threshold.reshape(-1,1), axis=1)

        
        total_wrong_val = np.sum(L_val_mat, axis=1)
        loss_red_val = 100*corrected_val/total_wrong_val

        table_val = pd.DataFrame({
            #---------------- 
            'rho_user':rho_grid, \
            'corrected_val':np.around(corrected_val,decimals=2),\
            'queries_val':np.around(queries_val,decimals=2),\
            'total_wrong_val':total_wrong_val,\
            'loss_query_val':np.around(np.array(corrected_val)/np.array(queries_val),decimals=2),\
            'rho_hat_val':queries_val/N_val,\
            '%loss_red_val':np.around(loss_red_val,decimals=2),\
            #----------------
            'thresh':threshold
        })
        return table_val

    def test(self, y_test, y_hat_test, crit_test, eta):
        L_test = lossEvaluation(y_test, y_hat_test).getLoss(self.norm)
        eta = eta.reshape(-1,1)
        N_test = y_test.shape[0]

        crit_mat = np.tile(crit_test, (eta.size,1))
        L_test_mat = np.tile(L_test, (eta.size,1))
        if self.direction == 'closer':
            corrected_test = np.sum(L_test_mat*(crit_mat<eta), axis=1)
            queries_test = np.sum(crit_mat<eta, axis=1)
        elif self.direction == 'further':
            corrected_test = np.sum(L_test_mat*(crit_mat>eta), axis=1)
            queries_test = np.sum(crit_mat>eta, axis=1)

        total_wrong_test = np.sum(L_test_mat, axis=1)
        rho_hat_test = queries_test/N_test
        loss_red_test = corrected_test/total_wrong_test
        table_test = pd.DataFrame({
            'corrected_test':np.around(corrected_test, decimals=2),\
            'queries_test':np.around(queries_test, decimals=2),\
            'total_wrong_test':total_wrong_test,\
            'loss_query_test':np.around(np.array(corrected_test)/np.array(queries_test),decimals=2),\
            'rho_hat_test':np.around(rho_hat_test, decimals=2), \
            '%loss_red_test':np.around(loss_red_test*100,decimals=2)
        })
        return table_test

class patching():
    def __init__(self, norm='res'):
        self.norm = norm
        self.gpr = None
    
    def fit(self, X_train, y_train, y_hat_train, kernel, n_iter=50, lr=0.1, ex_dim=1):
        # Fit signaling function based on training set
        L_train = lossEvaluation(y_train, y_hat_train).getLoss(self.norm)
        dims = list(range(0,X_train.shape[1]))
        active_dims_x = np.array(dims[:-ex_dim])
        active_dims_y = np.array(dims[-ex_dim:])

        if kernel == 'exponential':
            k = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5,ard_num_dims=X_train.shape[1]))
        elif kernel == 'RBF':
            k = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=X_train.shape[1]))
        elif kernel == 'e+RBF':
            # dims = list(range(0,X_train.shape[1]))
            # print(dims)
            k_x = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=len(active_dims_x), active_dims=active_dims_x))
            k_y = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=len(active_dims_y), active_dims=active_dims_y))
            k = k_x + k_y
        elif kernel == 'e*RBF':
            # dims = list(range(0,X_train.shape[1]))
            # print(dims)
            k_x = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=len(active_dims_x), active_dims=active_dims_x))
            k_y = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=len(active_dims_y), active_dims=active_dims_y))
            k = k_x * k_y
        elif kernel == 'e+e':
            # dims = list(range(0,X_train.shape[1]))
            # print(dims)
            k_x = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=len(active_dims_x), active_dims=active_dims_x))
            k_y = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=len(active_dims_y), active_dims=active_dims_y))
            k = k_x + k_y
        elif kernel == 'e*e':
            # dims = list(range(0,X_train.shape[1]))
            # print(dims)
            k_x = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=len(active_dims_x), active_dims=active_dims_x))
            k_y = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=len(active_dims_y), active_dims=active_dims_y))
            k = k_x * k_y
        elif kernel == 'RBF+RBF':
            # dims = list(range(0,X_train.shape[1]))
            # print(dims)
            k_x = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=len(active_dims_x), active_dims=active_dims_x))
            k_y = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=len(active_dims_y), active_dims=active_dims_y))
            k = k_x + k_y
        elif kernel == 'RBF*RBF':
            # dims = list(range(0,X_train.shape[1]))
            # print(dims)
            k_x = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=len(active_dims_x), active_dims=active_dims_x))
            k_y = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=len(active_dims_y), active_dims=active_dims_y))
            k = k_x * k_y
        
        self.gpr = gaussianProcess(X_train, L_train, k)
        self.gpr.fit(n_iter=n_iter,lr=lr)
    
    def apply(self, X_val, y_val, y_val_pred_soft, y_val_pred_th, 
                    X_test, y_test, y_test_pred_soft, y_test_pred_th):
        if y_val_pred_soft.shape[1]>1:
            y_val_pred_soft = y_val_pred_soft[:,1]
            y_test_pred_soft = y_test_pred_soft[:,1]
        else:
            y_val_pred_soft = y_val_pred_soft.squeeze()
            y_test_pred_soft = y_test_pred_soft.squeeze()
        # Predicted residual
        self.r_mean_val, self.r_var_val = self.gpr.predict(X_val)
        self.r_mean_test, self.r_var_test = self.gpr.predict(X_test)
        N_val = self.r_mean_val.shape[0] 
        N_test = self.r_mean_test.shape[0] 
        # Calibrated soft predictions
        cal_y_val_pred_soft = y_val_pred_soft + self.r_mean_val
        cal_y_test_pred_soft = y_test_pred_soft + self.r_mean_test
        # Search best threshold (TODO: USE 0.5 AS THRESHOLD)
        # thSet = np.sort(cal_y_val_pred_soft)[::-1]
        # ths = np.array([(thSet[i]+thSet[i+1])/2 for i in range(len(thSet)-1)])
        # best_idx = np.argmax(np.array([np.sum(y_val==(cal_y_val_pred_soft>th)) for th in ths]))
        # Calibrated thresholded predictions
        new_thr = 0.5 #ths[best_idx]
        cal_y_val_pred_th = np.array([cal_y_val_pred_soft > new_thr]).squeeze()
        cal_y_test_pred_th = np.array([cal_y_test_pred_soft > new_thr]).squeeze()
        # Compute losses with calibrated predictions
        cal_L_val = lossEvaluation(y_val, cal_y_val_pred_th).getLoss('l01')
        cal_L_test = lossEvaluation(y_test, cal_y_test_pred_th).getLoss('l01')
        # Compute error with calibrated predictions
        cal_error_val = np.sum(cal_L_val)/N_val
        cal_error_test = np.sum(cal_L_test)/N_test
        # Compute loss without calibrated predictions
        L_val = lossEvaluation(y_val, y_val_pred_th).getLoss('l01')
        L_test = lossEvaluation(y_test, y_test_pred_th).getLoss('l01')
        # Compute error without calibrated predictions
        error_val = np.sum(L_val)/N_val
        error_test = np.sum(L_test)/N_test
       
        # Compute improvement
        reduction_val = 100*(error_val-cal_error_val)/error_val
        reduction_test = 100*(error_test-cal_error_test)/error_test
        # Tabular results for patch
        table_patch = pd.DataFrame({
            '%reduction_val':np.around([reduction_val], decimals=2),\
                '%reduction_test':np.around([reduction_test], decimals=2)
        })
        return table_patch 
        
    def evaluate(self, y_val, y_val_pred_soft, y_val_pred_th, 
                    rho_grid=[0.01, 0.05, 0.1, 0.15, 0.2]):
        if y_val_pred_soft.shape[1]>1:
            y_val_pred_soft = y_val_pred_soft[:,1]
        else:
            y_val_pred_soft = y_val_pred_soft.squeeze()
        rho_grid = np.array(rho_grid)
        crit_val = self.r_var_val
        N_val = y_val.shape[0]
        # Calibrated soft predictions
        cal_y_val_pred_soft = y_val_pred_soft + self.r_mean_val
        # Calibrated thresholded predictions
        new_thr = 0.5 #ths[best_idx]
        cal_y_val_pred_th = np.array([cal_y_val_pred_soft > new_thr]).squeeze()
        # Compute losses with calibrated predictions
        cal_L_val = lossEvaluation(y_val, cal_y_val_pred_th).getLoss('l01')
        # Compute losses without calibrated predictions
        L_val = lossEvaluation(y_val, y_val_pred_th).getLoss('l01')
        # Grid of thresolds
        idx_rho = np.floor(N_val*rho_grid).astype('int')
        rho_grid = rho_grid[idx_rho!=0]
        idx_rho = idx_rho[idx_rho!=0]

        # Losses in Matrix representation 
        crit_mat = np.tile(crit_val, (idx_rho.size,1))
        L_val_mat = np.tile(L_val, (idx_rho.size,1))
        cal_L_val_mat = np.tile(cal_L_val, (idx_rho.size,1))
        # Compute thresolds based on the budgets
        crit_sort = np.sort(crit_val)[::-1]
        threshold = crit_sort[idx_rho]
        # Query the expert for GP_var>\eta
        error_val = np.sum(L_val_mat*(crit_mat>threshold.reshape(-1,1)), axis=1)/N_val
        # New loss for variance only
        sumL_val = np.sum(L_val_mat*(crit_mat<=threshold.reshape(-1,1)), axis=1)/N_val
        # New loss for calibrated predictions variance+patch
        cal_sumL_val = np.sum(cal_L_val_mat*(crit_mat<=threshold.reshape(-1,1)), axis=1)/N_val
        # Losses corrected by patch only
        patch_L_val = sumL_val-cal_sumL_val 

        sort_L_val = np.sort(L_val, axis=0)
        sort_L_val_mat = np.tile(sort_L_val, (idx_rho.size,1))
        cumL_val = np.flip(np.cumsum(sort_L_val_mat, axis=1)) / N_val

        red_val = 100*error_val/cumL_val[:,0]
        patch_red_val = 100*patch_L_val/cumL_val[:,0]
        table_val = pd.DataFrame({
            #---------------- 
            'rho_user':rho_grid, \
                'error_val':np.around(error_val,decimals=2),\
                    'patch_val':np.around(patch_L_val,decimals=2),\
                        'L_val':np.around(cal_sumL_val,decimals=2),\
                            '%patch_red_val':np.around(patch_red_val,decimals=2),\
                                '%T_reduction_val':np.around(red_val+patch_red_val,decimals=2),\
            #----------------
            'thresh':threshold
        })
        return table_val

    def test(self, y_test, y_test_pred_soft, y_test_pred_th, eta):
        if y_test_pred_soft.shape[1]>1:
            y_test_pred_soft = y_test_pred_soft[:,1]
        else:
            y_test_pred_soft = y_test_pred_soft.squeeze()
        crit_test = self.r_var_test
        eta = eta.reshape(-1,1)
        N_test = y_test.shape[0]
        # Calibrated soft predictions
        cal_y_test_pred_soft = y_test_pred_soft + self.r_mean_test
        # Calibrated thresholded predictions
        new_thr = 0.5 #ths[best_idx]
        cal_y_test_pred_th = np.array([cal_y_test_pred_soft > new_thr]).squeeze()
        # Compute losses with calibrated predictions
        cal_L_test = lossEvaluation(y_test, cal_y_test_pred_th).getLoss('l01')
        # Compute losses without calibrated predictions
        L_test = lossEvaluation(y_test, y_test_pred_th).getLoss('l01')

        crit_mat = np.tile(crit_test, (eta.size,1))
        L_test_mat = np.tile(L_test, (eta.size,1))
        cal_L_test_mat = np.tile(cal_L_test, (eta.size,1))

        # Query the expert for GP_var>\eta
        error_test = np.sum(L_test_mat*(crit_mat>eta), axis=1)/N_test
        # New loss for variance only
        sumL_test = np.sum(L_test_mat*(crit_mat<=eta), axis=1)/N_test
        # New loss for calibrated predictions variance+patch
        cal_sumL_test = np.sum(cal_L_test_mat*(crit_mat<=eta), axis=1)/N_test
        # Losses corrected by patch only
        patch_L_test = sumL_test-cal_sumL_test 

        rho_test = np.sum(crit_mat>eta, axis=1)/N_test
        sortL_test = np.sort(L_test, axis=0)
        sortL_test_mat = np.tile(sortL_test, (eta.size,1))
        cumL_test = np.flip(np.cumsum(sortL_test_mat, axis=1)) / N_test

        patch_red_test = 100*patch_L_test/cumL_test[:,0]
        red_test = 100*error_test/cumL_test[:,0]


        table_test = pd.DataFrame({
            'budget':np.around(rho_test, decimals=2), \
                'error_test':np.around(error_test, decimals=2),\
                    'patch_test':np.around(patch_L_test,decimals=2),\
                        'L_test':np.around(sumL_test,decimals=2),\
                            '%patch_red_test':np.around(patch_red_test,decimals=2),\
                                '%T_reduction_test':np.around(red_test+patch_red_test,decimals=2)
        })
        return table_test