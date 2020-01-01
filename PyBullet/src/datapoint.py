from copy import deepcopy

class Datapoint:
	def __init__(self):
		# Robot position list
		self.position = []
		# Metrics of all objects
		self.metrics = []
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

	def addPoint(self, pos, sticky, fixed, cleaner, action, cons, metric):
		self.position.append(deepcopy(pos))
		self.sticky.append(deepcopy(sticky))
		self.fixed.append(deepcopy(fixed))
		self.cleaner.append(deepcopy(cleaner))
		self.actions.append(deepcopy(action))
		self.constraints.append(deepcopy(cons))
		self.metrics.append(deepcopy(metric))

	def addSymbolicAction(self, HLaction):
		self.symbolicActions.append(HLaction['actions'])

	def toString(self, delimiter='\n', metrics=False):
		string = 'Symbolic actions:\n'
		for action in self.symbolicActions:
			string = string + "\n".join(map(str, action)) + '\n'
		string += 'States:\n'
		for i in range(len(self.position)):
			string = string + 'State ' + str(i) + ' ----------- ' + delimiter + \
				'Robot position - ' + str(self.position[i]) + delimiter + \
				'Sticky - ' + str(self.sticky[i]) + delimiter + \
				'Fixed - ' + str(self.fixed[i]) + delimiter + \
				'Cleaner? - ' + str(self.cleaner[i]) + delimiter + \
				'Action - ' + str(self.actions[i]) + delimiter + \
				'Constraints - ' + str(self.constraints[i]) + delimiter
			if metrics:
				string = string + 'All metric - ' + str(self.metrics) + delimiter
		return string

