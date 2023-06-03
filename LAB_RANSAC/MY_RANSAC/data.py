"""
Data generation module.
"""
import numpy as np

class Point_Generator():
    def __init__(self,num_points:int,percent_outliers:float) -> None:
        self.num_points = num_points
        self.percent_outliers = percent_outliers
        
        self.outliers_num = self.num_points * self.percent_outliers 
        self.inliers_num = self.num_points - self.outliers_num    
    def generate_case(self,
                      k:float=1.,
                      b:float=0.,
                      eps:float=0.1,
                      _case_:str="linear"
                      ) -> np.ndarray:
        """_summary_

        Args:
            k (float, optional): _description_. Defaults to 1..
            b (float, optional): _description_. Defaults to 0..
            eps (float, optional): _description_. Defaults to 0.1.

        Returns:
           datay: points (x,y)
        """                    
        if k is None:
            k = np.random.uniform(-1,1)
        if b is None:
            b = np.random.uniform(0,5)
        
        match _case_:
            case "linear":        
                
                x = np.linspace(0,10,int(self.inliers_num))
                y = k * x  + b  + np.random.normal(scale=eps)
                inliers = np.vstack((x,y)).T
                
                x = np.linspace(0,10,int(self.outliers_num)) 
                y = np.random.uniform(y.min(),y.max(),int(self.outliers_num))
                outliers = np.vstack((x,y)).T   
            case "sin":
                x = np.linspace(0,10,int(self.inliers_num))
                y = k * np.sin(x  + b  + np.random.normal(scale=eps))
                inliers = np.vstack((x,y)).T
                
                x = np.linspace(0,10,int(self.outliers_num)) 
                y = np.random.uniform(y.min(),y.max(),int(self.outliers_num))
                outliers = np.vstack((x,y)).T 
            case "cos":
                x = np.linspace(0,10,int(self.inliers_num))
                y = k * np.cos(x  + b  + np.random.normal(scale=eps))
                inliers = np.vstack((x,y)).T
                
                x = np.linspace(0,10,int(self.outliers_num)) 
                y = np.random.uniform(y.min(),y.max(),int(self.outliers_num))
                outliers = np.vstack((x,y)).T 
            case "complex_1":
                x = np.linspace(0,10,int(self.inliers_num))
                y = 1/np.exp(-b) * (np.cos(x  + k   + np.random.normal(scale=eps)) + np.cos(x  + k  + np.random.normal(scale=eps)))
                inliers = np.vstack((x,y)).T
                
                x = np.linspace(0,10,int(self.outliers_num)) 
                y = np.random.uniform(y.min(),y.max(),int(self.outliers_num))
                outliers = np.vstack((x,y)).T     
                
        data = np.concatenate([inliers,outliers])
        np.random.shuffle(data)                    
                 
        
        return data.T[0], data.T[1]
        