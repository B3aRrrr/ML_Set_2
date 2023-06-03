"""
   Module for workin with lines
"""
import numpy as np 
import scipy
   
class Line():
    def __init__(self,x:np.ndarray,y:np.ndarray,degree:int=1) -> None:
        self.x:np.ndarray = x
        self.y:np.ndarray = y
               
        self.poly_degree :int=degree
        self.params  =None
        
    def get_params(self):
        return self.params 
    
    def get_poly_degree(self):
        return self.poly_degree
    
    def estimate_params(self) -> None:
        points_num = self.x.shape[0]
        if points_num < 2:
            raise ValueError(f"Not enough points. Must be at least 2,but got {points_num}")
        elif 2 <= points_num <= 5:
            _x = np.sort(self.x)
            _y = self.y[_x.argsort()]
            A  = np.fromfunction(lambda i, j: _x[i,] ** j, (points_num, self.poly_degree+1), dtype=int)
            self.params = np.linalg.lstsq(A, _y, rcond=None)[0]
        else: 
            indexes = list(range(len(self.x)))
            ind_sample = np.random.choice(indexes,2)
            
            _x = np.sort(self.x[ind_sample])
            _y = self.y[ind_sample] 
            _y=_y[_x.argsort()]
            A  = np.fromfunction(lambda i, j: _x[i,] ** j, (2, self.poly_degree+1), dtype=int)
            self.params = np.linalg.lstsq(A, _y, rcond=None)[0]
            
    
    def eval_val(self,x:np.ndarray) -> np.ndarray:
        return  np.fromfunction(lambda i, j: x[i,] ** j, (x.shape[0], self.poly_degree+1), dtype=int) @ self.params

    def devide_points(self, x:np.ndarray,y:np.ndarray,eps:float):
            x_in = []
            y_in = []
            x_out = []  
            y_out = []
            for i in range(x.shape[0]):
                X = x[i,]
                Y = y[i,]
                # print(f"self.eval_val(np.array([X]))= {self.eval_val(np.array([X]))}")
                if abs(Y - self.eval_val(np.array([X]))) < eps:
                    x_in.append(X)
                    y_in.append(Y)
                else:
                    x_out.append(X)
                    y_out.append(Y)
            return (x_in,y_in,x_out,y_out)
        
       
               