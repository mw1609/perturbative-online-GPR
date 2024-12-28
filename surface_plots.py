import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import scipy as scp
import scipy.integrate as integrate
import math
import warnings
import winsound
import time




##code for surface plots, perturbative update (plotting marginal likelihood (p(y|\theta)))
warnings.filterwarnings("error", category=RuntimeWarning)




np.random.seed(0)
noise = 1e-3
jitter = 1e-6
# data = np.genfromtxt("spektrum.csv",delimiter = ",")
# x = data[:,0].reshape(-1,1)
# y = data[:,1].reshape(-1,1)




# Step 1: Generate the grid of x and y values
x = np.linspace(0,2,40).reshape(-1,1)
#rearraning for interpolation mode
x_test = np.linspace(min(x),max(x),300).reshape(-1,1) #test locations

y = 2*x+np.random.randn(len(x)).reshape(-1,1)*noise

def SE_kernel(X1, X2, sigma,l):
    sqdist = np.sum(X1**2, 1).reshape(-1,1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma**2 * np.exp(-0.5 / l**2 * sqdist)

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



# #3rd derivatives :( 
def d_l3_mat(x,sigma,l):
    d_l3_arr = np.zeros((len(x),len(x)))
    for i in range(len(x)):
        for j in range(len(x)): 
            d_l3_arr[i,j] =  -((x[j] - x[i])**2 * sigma**2 * 
           (12 * l**4 + 9 * (x[j] - x[i])**2 * l**2 + x[j]**4 - 4 * x[i] * x[j]**3 + 
            6 * x[i]**2 * x[j]**2 - 4 * x[i]**3 * x[j] + x[i]**4) * 
           np.exp((x[j] - x[i])**2 / (2 * l**2))) / l**9
    return d_l3_arr



def d_l2sig_mat(x,sigma,l):
    d_l2sig_mat = 2/sigma*d_l2_mat(x,sigma,l)
    return d_l2sig_mat


def d_sig2l_mat(x,sigma,l):
    d_sig2l_arr = np.zeros((len(x),len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            d_sig2l_arr[i,j] = (2*(x[j]-x[i])**2*np.exp(-1*(x[i]-x[j])**2/(2*l**2)))/l**3
    return d_sig2l_arr








#derivatives of inverse cov matrix
def d_sig2_inv_mat(x,theta,K_inv):
    term = -K_inv@d_sig2_mat(x,theta[0],theta[1])@K_inv+2*K_inv@dsig_mat(x,theta[0],theta[1])@K_inv@dsig_mat(x,theta[0],theta[1])@K_inv
    return term

def d_l2_inv_mat(x,theta,K_inv):
    term = 2*K_inv@dl_mat(x,theta[0],theta[1])@K_inv@dl_mat(x,theta[0],theta[1])@K_inv-K_inv@d_l2_mat(x,theta[0],theta[1])@K_inv
    return term

def d_sig_l_inv_mat(x,theta,K_inv):
    term = K_inv@dl_mat(x,theta[0],theta[1])@K_inv@dsig_mat(x,theta[0],theta[1])@K_inv + K_inv@dsig_mat(x,theta[0],theta[1])@K_inv@dl_mat(x,theta[0],theta[1])@K_inv - K_inv@d_lsig_mat(x,theta[0],theta[1])@K_inv
    return term




#####################



#############

def LML(x,y,sigma,l,K_inv = None):
    if K_inv is None:
        K = SE_kernel(x,x,sigma,l)
        K = K+np.eye(len(K))*jitter
        K_inv = np.linalg.inv(K) #just for checking purpose
    
    return -0.5*y.T@K_inv@y -0.5*np.log(1/np.linalg.det(K_inv))-len(y)/2*np.log(2*np.pi)



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


def standard_GP(x,y,init_index):
    #initially used part of dataset
    x_init = x[0:init_index]
    y_init = y[0:init_index]
    x_test_init = np.linspace(min(x_init), max(x_init),100)
    
    #initial parameters, and covariance block matrix
    init_theta = max_gradient(4e-2 ,x_init,y_init)
    cov2,Kxx22,Kxx2,Kx2x,Kx2x2 = make_covariance_arr(init_theta[0],init_theta[1],x_init,x_test_init)
    # (optional) inference using initial parameters and data points
    Kinv = np.linalg.inv(Kxx22)
    post_mu = Kx2x@Kinv@y_init
    post_cov = Kx2x2-Kx2x@Kinv@Kxx2

    return init_theta, post_mu, post_cov, Kxx22,Kinv



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
    beta = Kinv@y
    
    dsig = 1/2*np.trace((beta@np.transpose(beta)-Kinv)@K_x_x_dsig)
    dl = 1/2*np.trace((beta@np.transpose(beta)-Kinv)@K_x_x_dl)
    grad = np.zeros(2)
    grad[0]= dsig
    grad[1] = dl
    
    return grad


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






def max_gradient(step_size,x,y): 
    global threshold
    threshold = 5e-4

    params = np.array([5,5])

    global init_guess
    init_guess = params
    # params = np.array([3,3])
    print("guess:", params)
    grad = LML_gradient(params[0],params[1],x,y)
    global theta_diff
    theta_diff = threshold +1 
    while np.abs(theta_diff) > threshold:
        grad = LML_gradient(params[0],params[1],x,y)
        # print("grad = ", grad)
        params1 = params + step_size*grad
        theta_diff = np.sum(np.abs(params)-np.abs(params1))
        params = params1
    
    return params




def posterior(x,y,sigma,l,K_inv = None):
    if K_inv is None:
        term= ML(y,x,sigma,l)*1/(sigma*l)
    else: 
        term = ML(y,x,sigma,l,K_inv)*1/(sigma*l)
    return term

#####################





def Online_GP_approx(x,y,init_index,key,theta = None, K_inv = None):
    #getting initial parameters and initial inverse cov. matrix from the standard GP
    if theta == None and K_inv == None:
        global init_theta,x1,y1,K_inv_init
        init_theta,_,_,init_K,K_inv_init = standard_GP(x,y,init_index)
        print("standard parameters = ", init_theta)
    K = init_K
    K_inv = K_inv_init
    theta = init_theta
    K_inv_0 = K_inv_init


  
    #variables for convergence
    threshold1 = 1e-1
    global theta_diff1
    theta_diff1 = threshold1+1
    max_iters = 1e2

    # d = local_gradient(H,grad,theta) #initial local gradient
    for i in range(init_index+1, init_index+2): # just for one data update in plotting 
        # consideration of new data point 
        x1 = x[:i]
        y1 = y[:i]
        xl = x1[-1].reshape(1,1) #newest element
    
        global iters
        iters = 0
        theta_diff1 = 1
        global add_term
        add_term = np.ones(K_inv.shape)
        norm_quot = 2

        if i == 2:
            global K_inv_online3
            K_inv_online3 = K_inv

        theta0 = theta 
        print("theta0",theta0)

        while  iters < 1 and np.abs(theta_diff1) > threshold1: # (loop if online shift was done iteratively, default = 1 single computation)
            # iterative loop for parameterupdate
            k_vec = SE_kernel(x1[:-1],xl,theta[0],theta[1]) 
            k_scal = SE_kernel(xl,xl,theta[0],theta[1])

            #new inverse covariance matrix
            q = K_inv@k_vec
            gams = k_scal +jitter - k_vec.T@q

            add_term = 1/gams*((np.append(q,-1).reshape(-1,1))@(np.append(q,-1).reshape(-1,1).T))

            if iters ==0:
                norm  = np.linalg.norm(add_term,'fro').reshape(1,1)
            else: 
                norm= np.append(norm,np.linalg.norm(add_term,'fro'))
            

            K_inv_new = np.vstack([K_inv,np.zeros((1,K_inv.shape[1]))])
            K_inv_new = np.hstack([K_inv_new,np.zeros((K_inv_new.shape[0],1))])
            

            #K_t+1 with old \theta for later approx. of K^-1
            K_inv_new = K_inv_new + add_term
            if iters == 0:
                K_inv_0 = K_inv_new

            #parameter update 

            #get Hessian, evaluated at new parameters, with new add_term (new optimum)
            global H
            H = LML_Hessian(K_inv_new,x1,y1,theta)
            H = H+np.eye(len(H))*jitter 
            H_inv = np.linalg.inv(H) #2x2 matrix, fast
            grad = LML_gradient(theta[0],theta[1],x1,y1,K_inv_new)



            #estimate learning rate \alpha 
            shift = -H_inv@grad
            if iters == 0:
                alpha = (-shift.T@grad + shift.T@H@theta)/(shift.T@H@shift)



            global loop_shift
            loop_shift = alpha*H_inv@grad
            theta1 = theta - loop_shift
    
            theta_diff1 = (np.abs((theta1-theta)@(theta1-theta))) #for convergence
            
            theta = theta1
            iters = iters + 1

        
     
        #inv. cov. updateth
        global param_shift
        param_shift = theta-theta0 #shift for computation of appr. inverse cov. matrix
        #pertubation terms normal pertubation
        if key == "direct":
            print("direct")
            #method 1 (direct inversion)
            lin_term =  param_shift[0]*(-K_inv_0)@dsig_mat(x1,theta0[0],theta0[1])@K_inv_0+param_shift[1]*(-K_inv_0)@dl_mat(x1,theta0[0],theta0[1])@K_inv_0
            quad_term = 1/2*(param_shift[0]**2*d_sig2_inv_mat(x1,theta0,K_inv_0)+2*param_shift[0]*param_shift[1]*d_sig_l_inv_mat(x1,theta0,K_inv_0)+param_shift[1]**2*d_l2_inv_mat(x1,theta0,K_inv_0))
            # cube_term = 1/6*d_sig3_inv(x,theta,K_inv)@param_shift[0]**3 + 1/2*d_sig2l_inv(x,theta,K_inv)*param_shift[0]**2*param_shift[1]+1/2*d_l2sig_inv(x,theta,K_inv)*param_shift[0]*param_shift[1]**2+1/6*d_l3_inv(x,theta,K_inv)*param_shift[1]**3 
            K_inv_app = K_inv_0 + lin_term +quad_term

        else:
            print("indirect")
            K = SE_kernel(x1,x1,theta0[0],theta0[1])
            lin_term  = param_shift[0]*dsig_mat(x1,theta0[0],theta0[1])+param_shift[1]*dl_mat(x1,theta0[0],theta0[1])
            quad_term = 1/2*(param_shift[0]**2*d_sig2_mat(x1,theta0[0],theta0[1])+2*param_shift[0]*param_shift[1]*d_lsig_mat(x1,theta0[0],theta0[1])+param_shift[1]**2*d_l2_mat(x1,theta0[0],theta0[1]))
        

            # K_inv_app = K_inv_0 -K_inv_0@(lin_term+quad_term)@K_inv_0 
            K_inv_app = K_inv_0 -K_inv_0@(lin_term+quad_term)@K_inv_0 + K_inv_0@(lin_term+quad_term)@K_inv_0@(lin_term+quad_term)@K_inv_0


        K_inv = K_inv_app #martrix update 

    #inference with gained parameters and respective inverse covariance matrix 
    global Kxx22_final
    # cov_final,Kxx22_final,Kxx2_final,Kx2x_final,Kx2x2_final = make_covariance_arr(theta[0],theta[1],x,x_test)
    # post_mu = Kx2x_final@K_inv@y
    # post_cov = Kx2x2_final-Kx2x_final@K_inv@Kxx2_final

    print("final parameters",theta)
    return theta,K_inv




#function that shifts through grid 
def Online_update_new(x1,init_theta,sigma, l,K_inv_init,key):
    theta1 = np.array([sigma,l])
    param_shift = theta1-init_theta #shift for computation of appr. inverse cov. matrix
    theta0 = init_theta
    #pertubation terms

    #version 1( direct)
    if key == "direct":

        lin_term =  param_shift[0]*(-K_inv_init)@dsig_mat(x1,theta0[0],theta0[1])@K_inv_init+param_shift[1]*(-K_inv_init)@dl_mat(x1,theta0[0],theta0[1])@K_inv_init
        quad_term = 1/2*(param_shift[0]**2*d_sig2_inv_mat(x1,theta0,K_inv_init)+2*param_shift[0]*param_shift[1]*d_sig_l_inv_mat(x1,theta0,K_inv_init)+param_shift[1]**2*d_l2_inv_mat(x1,theta0,K_inv_init))
        # cube_term = 1/6*d_sig3_inv(x1,theta0,K_inv_init)*param_shift[0]**3 + 1/2*d_sig2l_inv(x1,theta0,K_inv_init)*param_shift[0]**2*param_shift[1]+1/2*d_l2sig_inv(x1,theta0,K_inv_init)*param_shift[0]*param_shift[1]**2+1/6*d_l3_inv(x1,theta0,K_inv_init)*param_shift[1]**3 

        K_inv_app = K_inv_init + lin_term + quad_term

    else:    
        K = SE_kernel(x1,x1,theta0[0],theta0[1])
        lin_term  = param_shift[0]*dsig_mat(x1,theta0[0],theta0[1])+param_shift[1]*dl_mat(x1,theta0[0],theta0[1])
        quad_term = 1/2*(param_shift[0]**2*d_sig2_mat(x1,theta0[0],theta0[1])+2*param_shift[0]*param_shift[1]*d_lsig_mat(x1,theta0[0],theta0[1])+param_shift[1]**2*d_l2_mat(x1,theta0[0],theta0[1]))
        
        K_inv_app = K_inv_init - K_inv_init@(lin_term+quad_term)@K_inv_init
        K_inv_app = K_inv_init -K_inv_init@(lin_term+quad_term)@K_inv_init + K_inv_init@(lin_term+quad_term)@K_inv_init@(lin_term+quad_term)@K_inv_init

        



    print("shift",param_shift)
    return K_inv_app, theta1





###################################### end of definition of functions 

key = "direct"
# theta,K_inv = Online_GP_approx(x,y,2,key)
theta,_,_,_,K_inv = standard_GP(x,y,10)



# # #set K_inv_0  = K_inv_init, theta0  = init_theta, to plot surface before update 
# # #set K_inv_0  = K_inv, theta0  = theta, to plot surface after online update
K_inv_0  = K_inv
theta0  = theta


index = len(K_inv_0)
print("index: ",index)
steps = 300
# # bound = 1
# # x_offset = 0
# # y_offset = 0
# # #for plotting of bigger domain, set equal to x_min,x_max etc. for same domain
x_min_man, x_max_man = 1.7,20
y_min_man, y_max_man = 1.7,20

# x_min, x_max = 3,12
# y_min,y_max = 3,10
x_min,x_max = x_min_man,x_max_man
y_min,y_max = y_min_man,y_max_man


SIG,L = np.meshgrid(np.linspace(x_min,x_max,steps),np.linspace(y_min,y_max,steps))
Z = np.zeros((len(SIG),len(SIG)))


#inverse covariance matrix is calculated manually just once in order to perturbatively update (updates are still calculated via perturbation)
#just so that the array can be accessed easily
K_inv_0 = np.linalg.inv(SE_kernel(x[:index],x[:index],SIG[0,0],L[0,0])+np.eye(index)*jitter)
theta0 = np.array([SIG[0,0],L[0,0]])



# loop thorugh grid evaluating ML, w.r.t. perturbatively updated matrix
# try:
#     for i in range(len(SIG)):
#         if i % 2 == 0:
#             for j in range(len(SIG)):
#                     if i == 0 and j == 0:
#                         j = 1
#                 #initial inversion
#                     K_inv1,theta1 = Online_update_new(x[:index],theta0,SIG[i,j],L[i,j],K_inv_0,key)
#                     # print(1/np.linalg.det(K_inv1))
#                     print("element:",SIG[i,j],L[i,j])
#                     Z[i,j] = ML(x[:index],y[:index],SIG[i,j],L[i,j],K_inv1)
#                     K_inv_0 = K_inv1
#                     theta0 = theta1
#         else:
#             for j in range(len(SIG) - 1, -1, -1):
#                 K_inv1,theta1 = Online_update_new(x[:index],theta0,SIG[i,j],L[i,j],K_inv_0,key)
#                 print("element:",SIG[i,j],L[i,j])
#                 Z[i,j] = ML(x[:index],y[:index],SIG[i,j],L[i,j],K_inv1)
#                 K_inv_0 = K_inv1
#                 theta0 = theta1
# except: 
#     print("RUNTIME ERROR, INCREASE STEPS OR DECREASE DOMAIN :(")
#     winsound.Beep(400,1000)

def plot_surface(SIG,L,Z,theta0,K_inv_0,key):
    try:
        for i in range(len(SIG)):
            if i % 2 == 0:
                for j in range(len(SIG)):
                        if i == 0 and j == 0:
                            j = 1
                    #initial inversion
                        K_inv1,theta1 = Online_update_new(x[:index],theta0,SIG[i,j],L[i,j],K_inv_0,key)
                        # print(1/np.linalg.det(K_inv1))
                        print("element:",SIG[i,j],L[i,j])
                        Z[i,j] = ML(x[:index],y[:index],SIG[i,j],L[i,j],K_inv1)
                        K_inv_0 = K_inv1
                        theta0 = theta1
            else:
                for j in range(len(SIG) - 1, -1, -1):
                    K_inv1,theta1 = Online_update_new(x[:index],theta0,SIG[i,j],L[i,j],K_inv_0,key)
                    print("element:",SIG[i,j],L[i,j])
                    Z[i,j] = ML(x[:index],y[:index],SIG[i,j],L[i,j],K_inv1)
                    K_inv_0 = K_inv1
                    theta0 = theta1
    except: 
        print("RUNTIME ERROR, INCREASE STEPS OR DECREASE DOMAIN :(")
        winsound.Beep(400,1000)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    cont = ax.tricontour(SIG.ravel(),L.ravel(),Z.ravel(),cmap='plasma')
    ax.set_xlabel(f'$\sigma$ ')
    ax.plot(init_guess[0],init_guess[1],"ro")
    # ax.plot(init_theta[0],init_theta[1],"rx")
    ax.plot(theta[0],theta[1],"rx")
    ax.set_ylabel('l')
    print("approximated")
    print(key)
    plt.show()
        # #######
    winsound.Beep(800,1000)


plot_surface(SIG,L,Z,theta0,K_inv_0,"direct")



key = "indirect"
#same for different perturbation
# theta,K_inv = Online_GP_approx(x,y,10,key)
# theta,_,_,_,K_inv = standard_GP(x,y,10)


K_inv_0 = np.linalg.inv(SE_kernel(x[:index],x[:index],SIG[0,0],L[0,0])+np.eye(index)*jitter)
theta0 = np.array([SIG[0,0],L[0,0]])


# # #set K_inv_0  = K_inv_init, theta0  = init_theta, to plot surface before update 
# # #set K_inv_0  = K_inv, theta0  = theta, to plot surface after online update
Z = np.zeros((len(SIG),len(SIG)))
plot_surface(SIG,L,Z,theta0,K_inv_0,"indirect")





# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)

# cont = ax.tricontour(SIG.ravel(),L.ravel(),Z.ravel(),cmap='plasma')
# ax.set_xlabel(f'$\sigma$ ')
# ax.plot(init_guess[0],init_guess[1],"ro")
# ax.plot(init_theta[0],init_theta[1],"rx")
# ax.plot(theta[0],theta[1],"rx")
# ax.set_ylabel('l')
# print("approximated")
# plt.show()
# # #######
# winsound.Beep(800,1000)

# # ##manually+
steps = 200
sig_arr = np.linspace(x_min_man, x_max_man,steps)
l_arr = np.linspace(y_min_man, y_max_man, steps)
SIG_arr, L_arr = np.meshgrid(sig_arr,l_arr)



fig = plt.figure()

ax = fig.add_subplot(1,1,1)
z = np.zeros(SIG_arr.shape)
for i in range(len(z)):
        for j in range(len(z)):
            K_inv1 = np.linalg.inv(SE_kernel(x[:index],x[:index],SIG_arr[i,j],L_arr[i,j])+np.eye(index)*jitter)
            # print(1/np.linalg.det(K_inv1))
            z[i,j] = ML(x[:index],y[:index],SIG_arr[i,j],L_arr[i,j])
            if math.isnan(z[i,j]):
                z[i,j] = 0
cont = ax.tricontour(SIG_arr.ravel(),L_arr.ravel(),z.ravel(),cmap='plasma')
ax.set_xlabel(f'$\sigma$ ')
ax.set_ylabel('l')

#initial guess
ax.plot(init_guess[0],init_guess[1],"ro")
#initially optimized
# ax.plot(init_theta[0],init_theta[1],"rx")
#online optimized final
ax.plot(theta[0],theta[1],"rx")

print("manual")
plt.show()


