import pickle
from src.datapoint import *

dp = pickle.load(open('./dataset/home/goal1-milk-fridge/world_home0/0.datapoint', 'rb'))
g = dp.getGraph()

print(json.dumps(g, indent=2))