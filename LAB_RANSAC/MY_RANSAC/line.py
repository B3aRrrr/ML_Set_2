"""
   Module for workin with lines
"""
import numpy as np 
   
class Line():
    def __init__(self,x:np.ndarray,y:np.ndarray,degree:int=1) -> None:
        self.x:np.ndarray = x
        self.y:np.ndarray = y
               
        self.poly_degree :int=degree
        self.poly1d=None
        
    def get_poly1d(self):
        return self.poly1d
    def get_poly_degree(self):
        return self.poly_degree
    
    def estimate_params(self) -> None:
        points_num = len(self.x)
        if points_num < 2:
            raise ValueError(f"Not enough points. Must be at least 2,but got {points_num}")
        else:
            self.poly1d = np.poly1d(np.polyfit(self.x,self.y,self.poly_degree))
    
    def eval_val(self,x:np.ndarray) -> np.ndarray:
                return self.poly1d(x)

    def devide_points(self, x:np.ndarray,y:np.ndarray,eps:float):
            x_in = []
            y_in = []
            x_out = []  
            y_out = []
            for i in range(x.shape[0]):
                X = x[i,]
                Y = y[i,]
                if abs(Y - self.eval_val(np.array([X]))) < eps:
                    x_in.append(X)
                    y_in.append(Y)
                else:
                    x_out.append(X)
                    y_out.append(Y)
            return (x_in,y_in,x_out,y_out)
        
       
               