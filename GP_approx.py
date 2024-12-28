import numpy as np 
import matplotlib.pyplot as plt 
import time 
from time import time 



noise = 1e-3#observation noise of data
jitter = 1e-6 #jitter for numerical stability

########## for real data
# data = np.genfromtxt("co2_monthly.csv",delimiter = ",")
# x = data[:,0].reshape(-1,1)
# y = data[:,1].reshape(-1,1)

# x = x[::8]
# y = y[::8]


# print("datapoints",len(x))
################################


x = np.linspace(-2,2,40).reshape(-1,1)
x_test = np.linspace(min(x),max(x),300).reshape(-1,1) #test locations

y = np.cos(5*x)+x**2+np.random.randn(len(x)).reshape(-1,1)*noise

# x = np.sort(100 * np.random.rand(10, 1), axis=0) not equally spaced

# initial batch that is used (must be even number for code)
init_index = 16
half_ind = int(init_index/2)


#rearraning for interpolation mode
# x = np.append(np.array([x[:half_ind],x[-half_ind:]]),x[half_ind:-half_ind]).reshape(-1,1)
# y = np.append(np.array([y[:half_ind],y[-half_ind:]]),y[half_ind:-half_ind]).reshape(-1,1)

avg_spacing = (max(x)-min(x))/len(x) # guess for l

x_test = np.linspace(min(x),max(x),300).reshape(-1,1) #test locations
y_original = np.cos(5*x_test)+x_test**2 #original noiseless function

#cov. kernelfunction
def SE_kernel(X1, X2, sigma,l):
    sqdist = np.sum(X1**2, 1).reshape(-1,1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma**2 * np.exp(-0.5 / l**2 * sqdist)



#creates covariance block-matrix
def make_covariance_arr(sigma, l, X,X_test):
    
    global K_x_x
    K_x_x = SE_kernel(X,X,sigma,l)
    K_x_x = K_x_x + np.eye(len(K_x_x))*jitter

    global K_x1_x1
    K_x1_x1 = SE_kernel(X_test,X_test, sigma,l)

    global K_x_x1
    K_x_x1 = SE_kernel(X,X_test,sigma,l)
    
    global K_x1_x
    K_x1_x = np.transpose(K_x_x1)
    cov_arr = np.block([[K_x_x,K_x_x1],
                       [K_x1_x,K_x1_x1]])    
    
    return cov_arr,K_x_x,K_x_x1,K_x1_x,K_x1_x1



#partial derivatives of kernel fct. for gradient / hessian computation

def dsig_kernel(x1,x2, sigma, l):
        return  2*sigma*np.exp(-1*(x1-x2)**2/(2*l**2))


def dl_kernel(x1,x2,sigma, l): 
    return ((x1-x2)**2*SE_kernel(x1.reshape(1,1),x2.reshape(1,1),sigma,l))/l**3


def dsig_mat(x,sigma, l): 
    dsig_mat = np.zeros((len(x), len(x)))
    for i in range(len(x)): 
          for j in range(len(x)): 
               dsig_mat[i,j] = dsig_kernel(x[i],x[j], sigma, l)
    return dsig_mat


def dl_mat(x, sigma,l):
    dl_mat = np.zeros((len(x), len(x)))
    for i in range(len(x)): 
        for j in range(len(x)): 
            dl_mat[i,j] = dl_kernel(x[i],x[j], sigma, l)
    return dl_mat 

#2nd derivative wrt. sigma 
def d_sig2_mat(x,sigma, l):
    d_sig2_mat = np.zeros((len(x),len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            d_sig2_mat[i,j] =  2*np.exp(-1*(x[i]-x[j])**2/(2*l**2))
    return d_sig2_mat

#2nd derivative wrt. l
def d_l2_mat(x,sigma,l):
     d_l2_mat = np.zeros((len(x), len(x)))
     for i in range(len(x)): 
          for j in range(len(x)):
               d_l2_mat[i,j] = -1*((x[j]-x[i])**2*sigma**2*(3*l**2-x[j]**2+2*x[i]*x[j]-x[i]**2)*np.exp(-(x[i]-x[j])**2/(2*l**2))/(l**6))
     return d_l2_mat

#2nd derivative wrt. l,sigma
def d_lsig_mat(x, sigma, l):
    d_lsig_mat = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        for j in range(len(x)): 
             d_lsig_mat[i,j] = (2*(x[i]-x[j])**2*sigma*np.exp(-(x[i]-x[j])**2/(2*l**2))/(l**3))
    return d_lsig_mat


#3rd derivative wrt. l
def d_l3_mat(x,sigma,l):
    dl_3_arr = np.zeros((len(x),len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            dl_3_arr[i,j] = 1/l**9*((x[j]-x[i])**2*sigma**2*(12*l**4+(-9*x[j]**2+18*x[i]*x[j]-9*x[i]**2)*l**2 +x[j]**4-4*x[i]*x[j]**3+6*x[i]**2*x[j]**2-4*x[i]**3*x[j]+x[i]**4)*np.exp(-(x[i]-x[j])/(2*l**2)))
    return dl_3_arr


def d_l2sig_mat(x,sigma,l):
    d_l2sig_arr = dl_mat(x,sigma,l)*2/sigma
    return d_l2sig_arr

def d_sig2l_mat(x,sigma,l):
    d_sig2l_arr = np.zeros((len(x),len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            d_sig2l_arr[i,j] = 1/l**3*((x[i]-x[j])**2*np.exp(-(x[i]-x[j])**2/(2*l**2)))


######

# 2nd derivatives of inverse cov matrix
def d_sig2_inv_mat(x,theta,K_inv):
    term = -K_inv@d_sig2_mat(x,theta[0],theta[1])@K_inv+2*K_inv@dsig_mat(x,theta[0],theta[1])@K_inv@dsig_mat(x,theta[0],theta[1])@K_inv
    return term

def d_l2_inv_mat(x,theta,K_inv):
    term = 2*K_inv@dl_mat(x,theta[0],theta[1])@K_inv@dl_mat(x,theta[0],theta[1])@K_inv-K_inv@d_l2_mat(x,theta[0],theta[1])@K_inv
    return term

def d_sig_l_inv_mat(x,theta,K_inv):
    term = K_inv@dl_mat(x,theta[0],theta[1])@K_inv@dsig_mat(x,theta[0],theta[1])@K_inv + K_inv@dsig_mat(x,theta[0],theta[1])@K_inv@dl_mat(x,theta[0],theta[1])@K_inv - K_inv@d_lsig_mat(x,theta[0],theta[1])@K_inv
    return term

##########


# 3rd derivatives of inverse cov martix


def d_sig3_inv(x,theta,K_inv):
    inv_dsig = -K_inv@dsig_mat(x,theta[0],theta[1])@K_inv
    term = -(inv_dsig@d_sig2_mat(x,theta[0],theta[1])@K_inv+K_inv@d_sig2_mat(x,theta[0],theta[1])@inv_dsig)+ 2*(inv_dsig@dsig_mat(x,theta[0],theta[1])@K_inv@dsig_mat(x,theta[0],theta[1])@K_inv+K_inv@d_sig2_mat(x,theta[0],theta[1])@K_inv@dsig_mat(x,theta[0],theta[1]@K_inv) \
                                                                                                                +K_inv@dsig_mat(x,theta[0],theta[1])@inv_dsig@dsig_mat(x,theta[0],theta[1])@K_inv+K_inv@dsig_mat(x,theta[0],theta[1])@K_inv@d_sig2_mat(x,theta[0],theta[1])@K_inv\
                                                                                                                +K_inv@dsig_mat(x,theta[0],theta[1])@K_inv@dsig_mat(x,theta[0],theta[1])@inv_dsig
                                                                                                                )
    return term

def d_l3_inv(x,theta,K_inv):
    inv_dl = -K_inv@dl_mat(x,theta[0],theta[1])@K_inv
    term = -(inv_dl@d_l2_mat(x,theta[0],theta[1])@K_inv+K_inv@d_l3_mat(x,theta[0],theta[1])@K_inv+K_inv@d_l2_mat(x,theta[0],theta[1])@inv_dl)+ 2*(inv_dl@dl_mat(x,theta[0],theta[1])@K_inv@dl_mat(x,theta[0],theta[1])@K_inv+K_inv@d_l2_mat(x,theta[0],theta[1])@K_inv@dl_mat(x,theta[0],theta[1]@K_inv) \
                                                                                                                +K_inv@dl_mat(x,theta[0],theta[1])@inv_dl@dl_mat(x,theta[0],theta[1])@K_inv+K_inv@dl_mat(x,theta[0],theta[1])@K_inv@d_l2_mat(x,theta[0],theta[1])@K_inv\
                                                                                                                +K_inv@dl_mat(x,theta[0],theta[1])@K_inv@dl_mat(x,theta[0],theta[1])@inv_dl
                                                                                                                )               
    return term

def d_l2sig_inv(x,theta,K_inv):
    inv_dsig = -K_inv@dsig_mat(x,theta[0],theta[1])@K_inv
    inv_dl = -K_inv@dl_mat(x,theta[0],theta[1])@K_inv

    term =  -(inv_dl@d_sig2_mat(x,theta[0],theta[1])@K_inv+K_inv@d_sig2l_mat(x,theta[0],theta[1])@K_inv+K_inv@d_sig2_mat(x,theta[0],theta[1])@inv_dl)+ 2*(inv_dl@dsig_mat(x,theta[0],theta[1])@K_inv@dsig_mat(x,theta[0],theta[1])@K_inv+K_inv@d_lsig_mat(x,theta[0],theta[1])@K_inv@dsig_mat(x,theta[0],theta[1]@K_inv) \
                                                                                                                +K_inv@dsig_mat(x,theta[0],theta[1])@inv_dl@dsig_mat(x,theta[0],theta[1])@K_inv+K_inv@dsig_mat(x,theta[0],theta[1])@K_inv@d_lsig_mat(x,theta[0],theta[1])@K_inv\
                                                                                                                +K_inv@dsig_mat(x,theta[0],theta[1])@K_inv@dsig_mat(x,theta[0],theta[1])@inv_dl
                                                                                                                )
    return term 


def d_sig2l_inv(x,theta,K_inv):
    inv_dsig = -K_inv@dsig_mat(x,theta[0],theta[1])@K_inv
    inv_dl = -K_inv@dl_mat(x,theta[0],theta[1])@K_inv

    term =  -(inv_dsig@d_l2_mat(x,theta[0],theta[1])@K_inv+K_inv@d_l2sig_mat(x,theta[0],theta[1])@K_inv+K_inv@d_l2_mat(x,theta[0],theta[1])@inv_dsig)+ 2*(inv_dsig@dl_mat(x,theta[0],theta[1])@K_inv@dl_mat(x,theta[0],theta[1])@K_inv+K_inv@d_lsig_mat(x,theta[0],theta[1])@K_inv@dl_mat(x,theta[0],theta[1]@K_inv) \
                                                                                                                +K_inv@dl_mat(x,theta[0],theta[1])@inv_dsig@dl_mat(x,theta[0],theta[1])@K_inv+K_inv@dl_mat(x,theta[0],theta[1])@K_inv@d_lsig_mat(x,theta[0],theta[1])@K_inv\
                                                                                                                +K_inv@dl_mat(x,theta[0],theta[1])@K_inv@dl_mat(x,theta[0],theta[1])@inv_dsig
                                                                                                                )
    return term 



##############



#initialization of gradient of log marginal likelihood
def LML_gradient(sigma,l,x,y, Kinv = None, init_index = None):     
    K_x_x_dsig = np.zeros((len(x), len(x)))
    K_x_x_dl = np.zeros((len(x), len(x)))

    Kxx = SE_kernel(x,x,sigma,l)
    for i in range(len(x)):
        for j in range(len(x)):
            K_x_x_dsig[i,j] = dsig_kernel(x[i],x[j],sigma,l)
            K_x_x_dl[i,j] = dl_kernel(x[i],x[j],sigma, l)
    Kxx = Kxx + np.eye(len(Kxx))*jitter

    if Kinv is None: #for initial matrix inversion, (not online)
         Kinv = np.linalg.inv(Kxx)
    alpha = Kinv@y
    
    dsig = 1/2*np.trace((alpha@np.transpose(alpha)-Kinv)@K_x_x_dsig)
    dl = 1/2*np.trace((alpha@np.transpose(alpha)-Kinv)@K_x_x_dl)
    grad = np.zeros(2)
    grad[0]= dsig
    grad[1] = dl
    
    return grad


# gradient based maximization, fixed stepsize and threshold, initial parameter learning
def max_gradient(step_size,x,y): 
    threshold = 1e-4
    params = np.array([5,avg_spacing[0]])
    # params = np.array([3,3])
    print("guess:", params)
    grad = LML_gradient(params[0],params[1],x,y)
    theta_diff = threshold +1 
    while np.abs(theta_diff) > threshold:
        grad = LML_gradient(params[0],params[1],x,y)
        # print("grad = ", grad)
        params1 = params + step_size*grad
        diff = params-params1 
        theta_diff = np.sqrt(np.abs(np.dot(diff,diff)))
        params = params1
    
    return params
def LML(x,y,sigma,l,K_inv = None):
    if K_inv is None:
        K = SE_kernel(x,x,sigma,l)
        K = K+np.eye(len(K))*jitter
        K_inv = np.linalg.inv(K) #just for checking purpose
    
    return -0.5*y.T@K_inv@y -0.5*np.log(1/np.linalg.det(K_inv))-len(y)/2*np.log(2*np.pi)


# inference and parameter learning with standard GP, for initial part of dataset 
def standard_GP(x,y,init_index):
    #initially used part of dataset
    x_init = x[0:init_index]
    y_init = y[0:init_index]
    x_test_init = np.linspace(min(x_init), max(x_init),100)
    
    #initial parameters, and covariance block matrix
    init_theta = max_gradient(1e-4,x_init,y_init)
    cov2,Kxx22,Kxx2,Kx2x,Kx2x2 = make_covariance_arr(init_theta[0],init_theta[1],x_init,x_test_init)
    # (optional) inference using initial parameters and data points
    Kinv = np.linalg.inv(Kxx22)
    post_mu = Kx2x@Kinv@y_init
    post_cov = Kx2x2-Kx2x@Kinv@Kxx2

    return init_theta, post_mu, post_cov, Kxx22,Kinv


#computation of hessian matrix of LML for approximative procedure 
def LML_Hessian(K_inv,x,y,theta):
    Hessian = np.zeros((2,2))
    dsig_matrix = dsig_mat(x,theta[0],theta[1])
    dl_matrix = dl_mat(x,theta[0], theta[1])
    inv_dsig = -K_inv@dsig_mat(x,theta[0],theta[1])@K_inv
    inv_dl = -K_inv@dl_mat(x,theta[0], theta[1])@K_inv
    t_y = np.transpose(y)

    d_l2_arr = d_l2_mat(x, theta[0],theta[1])
    d_sig2_arr = d_sig2_mat(x,theta[0], theta[1])
    d_lsig_arr = d_lsig_mat(x, theta[0], theta[1])
    
    #hessian matrix calculated by hand
    Hessian[0,0] = 1/2*t_y@(inv_dsig@dsig_matrix@K_inv+K_inv@d_sig2_arr@K_inv+K_inv@dsig_matrix@inv_dsig)@y -1/2*np.trace(inv_dsig@dsig_matrix+K_inv@d_sig2_arr)
    Hessian[1,1] = 1/2*t_y@(inv_dl@dl_matrix@K_inv+K_inv@d_l2_arr@K_inv+K_inv@dl_matrix@inv_dl)@y - 1/2*np.trace(inv_dl@dl_matrix+K_inv@d_l2_arr)
    Hessian[1,0] = 1/2*t_y@(inv_dsig@dl_matrix@K_inv+K_inv@d_lsig_arr@K_inv+K_inv@dl_matrix@inv_dsig)@y -1/2*np.trace(inv_dsig@dl_matrix+K_inv@d_lsig_arr)
    Hessian[0,1] = 1/2*t_y@(inv_dl@dsig_matrix@K_inv+K_inv@d_lsig_arr@K_inv+K_inv@dsig_matrix@inv_dl)@y -1/2*np.trace(inv_dl@dsig_matrix+K_inv@d_lsig_arr)
    return Hessian


    


# start of online algorithm
def Online_GP_approx(x,y,init_index,key):
    #calculate initial theta and initial inverse cov. matrix from initial subset 
    global init_theta,x1,y1,K_inv_init
    init_theta,_,_,init_K,K_inv_init = standard_GP(x,y,init_index)
    print("standard parameters = ", init_theta)
    
    global K_inv
    K_inv = K_inv_init
    theta = init_theta
    
    # define variables for convergence of optimization alg.
    global threshold
    threshold1 = 1e-4       # initialize numerical threshold for parameter change
    max_iters = 1 # max. iteration number 

    for i in range(init_index+1, len(x)+1): # data consideration for remaining subset 
        # consideration of new data point 
        x1 = x[:i]
        y1 = y[:i]
        xl = x1[-1].reshape(1,1) #newest element
    
        iters = 0
        theta0  = theta 
        norm_quot = 2
        theta_diff = threshold + 1
        while  iters < max_iters and np.abs(theta_diff) > threshold1: # (loop if online shift was done iteratively, default = 1 single computation)
             # iterative loop for local optimization of LML
            
            ## dimension update of covariance matrix using eq. 9 f.
            k_vec = SE_kernel(x1[:-1],xl,theta[0],theta[1]) 
            k_scal = SE_kernel(xl,xl,theta[0],theta[1])
            q = K_inv@k_vec
            gams = k_scal +jitter - k_vec.T@q
            add_term = 1/gams*((np.append(q,-1).reshape(-1,1))\
                               @(np.append(q,-1).reshape(-1,1).T))            

            K_inv_new = np.vstack([K_inv,np.zeros((1,K_inv.shape[1]))])
            K_inv_new = np.hstack([K_inv_new,np.zeros((K_inv_new.shape[0],1))])
            

            #K_t+1 with old \theta for later approx. of $K^{-1}$
            K_inv_new = K_inv_new + add_term
            if iters == 0:
                K_inv_init = K_inv_new
                norm  = np.linalg.norm(add_term,'fro').reshape(1,1)
            else: 
                norm= np.append(norm,np.linalg.norm(add_term,'fro'))



            #estimate learning rate \alpha 
            shift = -H_inv@grad
            if iters == 0:
                alpha = (-shift.T@grad + shift.T@H@theta)/(shift.T@H@shift)


            
            ##parameter update 

            #get Hessian and gradient of LML evaluated at new parameters, with new add_term
            H = LML_Hessian(K_inv_new,x1,y1,theta)
            H = H+np.eye(len(H))*jitter # jitter for num. stabilty 
            H_inv = np.linalg.inv(H) #2x2 matrix, fast
            grad = LML_gradient(theta[0],theta[1],x1,y1,K_inv_new)



            #compute loop parameter shift for new iteration of optimization loop
            loop_shift = H_inv@grad
            theta1 = theta - loop_shift


            #calculate change of parameters for convergence criterion
            theta_diff = (np.abs((theta1-theta)@(theta1-theta))) 
            #update theta for new iteration
            theta = theta1
            iters = iters + 1
            
            if iters >1:
                # print("norm= ", norm[-1])
                norm_quot = norm[-2]/norm[-1]
                # print("quotient=",norm_quot)
           
     


        ##inv. cov. update
        print("Iterations",iters)
        param_shift = theta-theta0 #shift for computation of appr. inverse cov. matrix

        if key == "direct": # pertubate inverse matrix 
            # second order pertubation of inverse covariance matrix 
            # first order term eq. 18: 
            lin_term =  param_shift[0]*(-K_inv_init)@dsig_mat(x1,theta0[0],theta0[1])\
                        @K_inv_init+param_shift[1]*(-K_inv_init)\
                        @dl_mat(x1,theta0[0],theta0[1])@K_inv_init  
            
            # second order term eq. 19
            quad_term = 1/2*(param_shift[0]**2*d_sig2_inv_mat(x1,theta0,K_inv_init)+\
                        2*param_shift[0]*param_shift[1]*\
                        d_sig_l_inv_mat(x1,theta0,K_inv_init)\
                        +param_shift[1]**2*d_l2_inv_mat(x1,theta0,K_inv_init)) 
        
            K_inv_app = K_inv_init + lin_term + quad_term #eq. 21 


        else: #pertubate cov matrix prior to inversion
            lin_term  = param_shift[0]*dsig_mat(x1,theta0[0],theta0[1])\
                        +param_shift[1]*dl_mat(x1,theta0[0],theta0[1])
            quad_term = 1/2*(param_shift[0]**2*d_sig2_mat(x1,theta0[0],theta0[1])\
                        +2*param_shift[0]*param_shift[1]*d_lsig_mat(x1,theta0[0],theta0[1])\
                        +param_shift[1]**2*d_l2_mat(x1,theta0[0],theta0[1]))
        

            K_inv_app = K_inv_init -K_inv_init@(lin_term+quad_term)@K_inv_init



        K_inv = K_inv_app #update matrix for consideration of new data point, for loop repeats

    #inference with gained parameters and respective inverse covariance matrix eqs. 2,3 
    global Kxx22_final
    _,Kxx22_final,Kxx2_final,Kx2x_final,Kx2x2_final\
             = make_covariance_arr(theta[0],theta[1],x,x_test)
    
    post_mu = Kx2x_final@K_inv@y
    post_cov = Kx2x2_final-Kx2x_final@K_inv@Kxx2_final
    print("final parameters",theta)
    return post_mu, post_cov, theta,K_inv

start = time()
mu_test,cov_test,theta,_ = Online_GP_approx(x,y,init_index,"indirect")
end = time()
print("duration = ",end-start, "s")
thetaplot = np.round(theta,5)



# plot of mean, 2xstd, original function and noisy data
plt.plot(x_test,mu_test,"k", label = "posterior mean")   
plt.plot(x,y,"rx",markersize = 0.5, label = "observed data")
# plt.plot(x_test,y_original, "m",alpha = 0.3,label = "original function")
plt.fill_between(
    x_test.ravel(),
    mu_test.ravel() - 1.96*np.diag(cov_test),
    mu_test.ravel() + 1.96*np.diag(cov_test)

)
# plt.title(f"K_inv online $\sigma$ ={thetaplot[0]}, l = {thetaplot[1]}")
plt.xlabel(f"$x$, input")
plt.ylabel(f"$y$, output")
plt.show()



# just for reference "offline" GP from other python file only inference
def ref_standard_GP(x,y,init_index):
    global x_init
    x_init = x[0:init_index]
    y_init = y[0:init_index]
    print("init_index", init_index)
    # init_theta = max_gradient(1e-9,x,y)
    global test_theta
    test_theta = np.array([1.41,1.16])
    print("theta reference",init_theta)
    cov2,Kxx22,Kxx2,Kx2x,Kx2x2 = make_covariance_arr(test_theta[0],test_theta[1],x_init,x_test)
    #inference mit den initialen Matrizen und Parametern
    Kinv = np.linalg.inv(Kxx22)
    post_mu = Kx2x@Kinv@y_init
    post_cov = Kx2x2-Kx2x@Kinv@Kxx2
    return init_theta, post_mu, post_cov, Kxx22


start = time()
ref_theta,ref_mu,ref_cov,Kxx22 = ref_standard_GP(x,y,len(x))
thetaref_plot = np.round(test_theta,2)
end = time()
print("duration", end-start)



plt.plot(x_test,ref_mu,"k", label = "posterior mean")   
plt.plot(x,y,"rx", markersize = 0.5, label = "observed data")
# plt.plot(x_test,y_original, "m",alpha = 0.3,label = "original function")
plt.fill_between(
    x_test.ravel(),
    ref_mu.ravel() - 1.96*np.diag(ref_cov),
    ref_mu.ravel() + 1.96*np.diag(ref_cov)

)
plt.xlabel("input, x")
plt.ylabel("output,y")
plt.show()




###################

def ML(x,y,sigma,l,K_inv = None):
    if K_inv is None:
       K = SE_kernel(x,x,sigma,l)
       K = K+np.eye(len(K))*jitter
       K_inv = np.linalg.inv(K) #just for checking purpose 
    #    print("library inversion",K_inv)
    else: 
        K_inv = K_inv
    
        # K = np.linalg.inv(K_inv+np.eye(len(K_inv))*jitter)
        #matrix norm without determinant of K 
        # print("library inversion",np.linalg.inv(SE_kernel(x,x,sigma,l)+np.eye(len(x))*jitter))
    ML = 1/np.sqrt(1/(np.linalg.det(K_inv))*2*np.pi**(len(x)))*np.exp((-0.5*y.T@K_inv@y)) 

    return ML

