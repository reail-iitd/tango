from copy import deepcopy

class Datapoint:
	def __init__(self):
		# Robot position list
		self.position = []
		# Sticky objects
		self.sticky = []
		# Fixed objects
		self.fixed = []
		# Has cleaner
		self.cleaner = []
		# Action
		self.actions = []
		# Constraints
		self.constraints = []
		# Symbolic actions
		self.symbolicActions = []

	def addPoint(self, pos, sticky, fixed, cleaner, action, cons):
		self.position.append(pos)
		self.sticky.append(sticky)
		self.fixed.append(fixed)
		self.cleaner.append(cleaner)
		self.actions.append(action)
		self.constraints.append(cons)

	def addSymbolicAction(self, HLaction):
		self.symbolicActions.append(HLaction)

