import numpy as np


class Utils:
	def set_weights(self,w):
		w_idx = 0
		for layer in self.layers:
			w_idx = layer.set_weights(w,w_idx)