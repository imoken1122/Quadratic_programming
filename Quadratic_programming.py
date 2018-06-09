import numpy as np
    

class Quadratic_programming:
    def __init__(self,Q,A,c,b):
        self.Q = Q
        self.A = A
        self.c = c
        self.b = b
        self.EPS = 1e-10
        self.flag =0
        self.n,self.m =A.shape
    def f(self,x):
        # demension  (1,2) * (2,2) - (1,2)*(2,1) = (1,1)
        return 0.5 * np.dot(np.dot(x.T,self.Q),x) - np.dot(self.c.T,x)
        
    def g(self,x):
        return np.dot(self.A,x) - self.b
    
    def gradient_f(self,x):
        return np.dot(self.Q,x) - self.c
    
    def parameter(self,W_index):
        
        if len(W_index) != 0:
            A_ = self.A[W_index,:]
            b_ = self.b[W_index]
            k,l = A_.shape
            M = np.dot(np.linalg.inv(np.vstack([np.hstack([A_,np.zeros([k,k])]),np.hstack([self.Q,A_.T])])),np.vstack([b_,self.c]))
            x_opt = M[0:self.n].reshape(-1,1)
            lambda_opt = np.zeros(self.m)
            lambda_opt[W_index] = M[self.n:len(M)]
        else:
            x_opt = np.dot(np.linalg.inv(self.Q),self.c)
            lambda_opt = np.zeros(self.m)
        return x_opt, lambda_opt
    
    def KKT_g(self,x):
        return (sum(self.g(x) < self.EPS) == len(self.g(x)))


    def optimaizer(self,x0):
        
        if not self.KKT_g(x0):
            for i in range(1000):
                x = np.random.normal(0,1,(self.n,1))
                if self.KKT_g(x):
                    x0=x
                    W = np.where(np.abs(self.g(x0))<self.EPS)[0]
                    self.flag = 1
                    break
            if not self.flag:
                print("No initial value")
        
        
        for i in range(1,1000):
            x_opt,lambda_opt = self.parameter(W)

            if self.KKT_g(x_opt): #step4
                if sum(lambda_opt >= 0.0) == len(lambda_opt):
                    self.flag =1
                    
                else:
                    #min_index = np.where(lambda_opt == np.min(lambda_opt))
                    min_index  =np.argmin(lambda_opt)
                    W = np.sort(np.delete(W,min_index)) 
                    self.flag = 0
        
            else: #step3
                a= np.dot(self.A,(x_opt-x0))  #分母が0にならないように
                a[np.where(a == 0.0)]=self.EPS
                t =-self.g(x0)/a 
                if len(t[np.where(t > 0.0)]) == 0:
                    t = 0.0
                else:
                    t = np.min(t[np.where(t >0.0)])
                x0 +=  t*(x_opt-x0)
                W = np.where(np.abs(self.g(x0))<self.EPS)[0]
                self.flag = 0
                
            if self.flag:
                print("optimal solution x:",x_opt.reshape(1,-1))
                print("min f(x):",self.f(x_opt))
                break
        if not self.flag:
            print("No optimal solution x")
                
            
            
def main():
    n = 2
    m = 2
    x0 = np.random.normal(0,1,(n,1))
    Q = np.random.rand(n,n)
    A = np.random.rand(m,n)
    c = np.random.rand(n,1)
    b = np.random.rand(m,1)
    model = Quadratic_programming(Q,A,c,b)
    model.optimaizer(x0)
        
if __name__ == "__main__":
    main()