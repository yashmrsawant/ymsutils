
import numpy as np
def sinc_interpolation(sig_mat1, dim, tr1, tr2):
    
    dim_shape = sig_mat1.shape
    nt = dim_shape[dim]
    
    T = tr1 * nt
    dt2 = tr2
    ts1 = np.linspace(0, T, nt)
    ts2 = np.arange(0, T, dt2)
    w_0 = 1/tr1
        
    dim_shape2 = list(dim_shape)
    dim_shape2[dim] = ts2.shape[0]
    sig_mat2 = np.zeros((dim_shape2))
    
    for i in range(ts2.shape[0]):
        
        ts = np.sinc(w_0 * (ts2[i] - ts1))
        
        if dim == 0:
            ts = ts.reshape((nt, 1, 1))
            sig_mat2[i, :, :] = np.sum(sig_mat1 * ts, axis = dim)
        elif dim == 1:
            ts = ts.reshape((1, nt, 1))
            sig_mat2[:, i, :] = np.sum(sig_mat1 * ts, axis = dim)
        elif dim == 2:
            ts = ts.reshape((1, 1, nt))
            sig_mat2[:, :, i] = np.sum(sig_mat1 * ts, axis = dim)
    return [{'ts' : ts1, 'signal' : sig_mat1}, {'ts' : ts2, 'signal' : sig_mat2}]
  
  
  
# Test code
from scipy.ndimage import gaussian_filter1d

sigma = 5

ntrials = 100
nt = 500
p = 5
sig_mat1 = np.zeros((ntrials, nt, p))
for trial_i in range(ntrials):
    x = np.random.randn(nt, p)
    for p_i in range(p):
        x[:, p_i] = gaussian_filter1d(x[:, p_i], sigma)
    sig_mat1[trial_i, :, :] = x
    
tr1 = 1/60
tr2 = 1/15.5
dim = 1

[dict1, dict2] = sinc_interpolation(sig_mat1, dim, tr1, tr2)

from matplotlib import pyplot as plt


ix = np.random.randint(100)
p_i = np.random.randint(5)
plt.plot(dict1['ts'], dict1['signal'][ix, :, p_i])
plt.plot(dict2['ts'], dict2['signal'][ix, :, p_i], linestyle = 'dashed')
plt.show()

  
