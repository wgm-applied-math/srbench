import torch
import os, sys
from symbolicregression.model import SymbolicTransformerRegressor
import requests
import numpy as np

from sklearn import feature_selection 
from sklearn.base import BaseEstimator, RegressorMixin

e2et_model=None
# model_path = os.path.join( # model in same folder as experiment/methods/e2et
#     os.path.dirname(os.path.abspath(__file__)),
#     "model.pt" 
# )
model_path = os.path.join( # model in bind folder
    "/srbench_pretrained/",
    "model1.pt" 
)
try:
    if not os.path.isfile(model_path): 
        url = "https://dl.fbaipublicfiles.com/symbolicregression/model1.pt"
        r = requests.get(url, allow_redirects=True)
        open(model_path, 'wb').write(r.content)
    if not torch.cuda.is_available():
        e2et_model = torch.load(model_path, map_location=torch.device('cpu'))
    else:
        e2et_model = torch.load(model_path)
        e2et_model = e2et_model.cuda()
    print(e2et_model.device)
    print("Model successfully loaded!")

except Exception as e:
    print("ERROR: model not loaded! path was: {}".format(model_path))
    print(e)    


max_input_points  = [50, 100, 250]
n_trees_to_refine = [50, 100, 250]
max_number_bags   = [1, 5, 10]

hyper_params = []
for points, trees in zip(max_input_points, n_trees_to_refine):
    for bags in max_number_bags:
        hyper_params.append({
                'max_input_points'  : [points],
                'n_trees_to_refine' : [trees],
                'max_number_bags'   : [bags]
                })
        

class E2ERegressor(BaseEstimator, RegressorMixin):
    def __init__(self, random_state=42, max_input_points=200,
                 n_trees_to_refine=100, max_number_bags=10):
        
        self.random_state = random_state
        self.max_input_points = max_input_points
        self.n_trees_to_refine = n_trees_to_refine
        self.max_number_bags = max_number_bags

    def fit(self, X, y):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        torch.cuda.manual_seed(self.random_state)

        self.est = SymbolicTransformerRegressor(
                        model=e2et_model,
                        max_input_points=self.max_input_points,
                        n_trees_to_refine=self.n_trees_to_refine,
                        max_number_bags=self.max_number_bags,
                        rescale=True
                        )
        
        self.est.fit(X, y, verbose=True)
    
        return self

    def predict(self, X):
        return self.est.predict(X)

est = E2ERegressor()

def model(est, X=None):
        
    replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}

    # model_str = est.retrieve_tree(dataset_idx=0).infix()
    model_str = est.est.retrieve_tree(with_infos=True)["relabed_predicted_tree"].infix()

    for op,replace_op in replace_ops.items():
        model_str = model_str.replace(op,replace_op)
        
    return model_str

def my_pre_train_fn(est, X, y):
    """In this example we adjust FEAT generations based on the size of X 
       versus relative to FEAT's batch size setting. 
    """
    return

# define eval_kwargs.
eval_kwargs = dict(
                   test_params = {
                    'max_input_points':100,
                    'n_trees_to_refine':10
                    },
                   use_dataframe=False
                  )
