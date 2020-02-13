from CONSTANTS import *

import torch
import torch.nn as nn

class GraphEncoder(nn.Module):
	def __init__(self):
		self.object_embeddings = nn.Embedding(NUMOBJECTS, embedding_dim)
		