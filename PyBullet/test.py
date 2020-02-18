import pickle
from src.datapoint import *

dp = pickle.load(open('test.datapoint', 'rb'))
g = dp.getGraph()

print(json.dumps(g, indent=2))