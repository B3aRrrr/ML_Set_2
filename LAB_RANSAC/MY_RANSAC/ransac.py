"""
RANSAC for 2d lines
Algorythm:

I Hypotesys generation Stage
1. Sample 2d points (1. 2 ponts; 2. 5 points)
2. Model estimation (1. analytics; 2. MSE estimation)

II Hypotesys evaluation Stage

3. Inlier counting (%inlinear > threshold) 
    if True -> best params
    if False -> 1.
4. # iter > num_iter?

"""
import numpy as np
import matplotlib.pyplot as plt
from MY_RANSAC import line

class RANSAC:
    def __init__(self):
        self.iter_num: int = 100
        
        self.num_key_points:int=2
        
        self.inlin_thrsh: float = 0.8
        self.epsilon: float = 0.1 # порог принадлежности точки зависимости
        
        
        self.best_params =None
        
        self.x_in: list = [] # список принадлежащих 
        self.x_out: list = [] # не принадлежащих
        self.y_in: list = [] # список принадлежащих 
        self.y_out: list = [] # не принадлежащих
        
        self.poly_degree :int=1 # степень полинома (по умолчанию линейный полином)
        
        self.score: int = 0
        
        self.x:np.ndarray=None
        self.y:np.ndarray=None
        
    def set_case(self,case_params)->None:
        '''
        case_params
        {
            #main
                x:np.ndarray,
                y:np.ndarray,
            #optional
                inlin_thrsh:int
                num_key_points:int
                poly_degree:int,
                epsilon: float,
                score:int 
                iter_num:int                 
        }
        '''
        if not ("x" in case_params.keys() and "y" in case_params.keys()):
            raise ValueError(f"[RANSAC] No x or y values in params")
        if "inlin_thrsh" in case_params.keys():
            self.inlin_thrsh = case_params["inlin_thrsh"]
        if "num_key_points" in case_params.keys():
            self.num_key_points = case_params["num_key_points"]
        if "poly_degree" in case_params.keys():
            self.poly_degree = case_params["poly_degree"]
        if "epsilon" in case_params.keys():
            self.epsilon = case_params["epsilon"]
        if "score" in case_params.keys():
            self.score = case_params["score"]
            
        self.x = case_params['x']
        self.y = case_params['y']
        
    def clear_case(self) -> None:
        pass
    def draw (self):
        plt.plot(self.x_in,self.y_in,color='red')
        plt.scatter(self.x_in,self.y_in,label='inlier',color='green')
        plt.scatter(self.x_out,self.y_out,label='outlier',color='blue')
        plt.grid()
        plt.legend()
        plt.show()
        
    def fit(self):
        for i in range(self.iter_num):
            indexes = list(range(len(self.x)))
            ind_sample = np.random.choice(indexes,self.num_key_points+1) 
            Line = line.Line(
                x=self.x[ind_sample],
                y=self.y[ind_sample],
                degree=self.poly_degree
                )
            Line.estimate_params()
            x_in,y_in,x_out,y_out = Line.devide_points(self.x,self.y,self.epsilon)
            loc_score = len(x_in) / len(self.x)
            if loc_score > self.score:
                self.best_params = Line.get_params()
                
                self.score=loc_score
                self.x_in = x_in
                self.y_in = y_in
                
                self.x_out = x_out
                self.y_out = y_out
                