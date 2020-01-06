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
		# Light state
		self.lighton = []
		# Dirt Cleaned
		self.dirtClean = []

	def addPoint(self, pos, sticky, fixed, cleaner, action, cons, metric, light, dirtClean):
		self.position.append(deepcopy(pos))
		self.sticky.append(deepcopy(sticky))
		self.fixed.append(deepcopy(fixed))
		self.cleaner.append(deepcopy(cleaner))
		self.actions.append(deepcopy(action))
		self.constraints.append(deepcopy(cons))
		self.metrics.append(deepcopy(metric))
		self.lighton.append(deepcopy(light))
		self.dirtClean.append(deepcopy(dirtClean))

	def addSymbolicAction(self, HLaction):
		self.symbolicActions.append(HLaction)

	def toString(self, delimiter='\n', metrics=False):
		string = 'Symbolic actions:\n'
		for action in self.symbolicActions:
			if str(action[0]) == 'E' or str(action[0]) == 'U':
				string = string + action + '\n'
				continue
			string = string + "\n".join(map(str, action)) + '\n'
		string += 'States:\n'
		for i in range(len(self.position)):
			string = string + 'State ' + str(i) + ' ----------- ' + delimiter + \
				'Robot position - ' + str(self.position[i]) + delimiter + \
				'Sticky - ' + str(self.sticky[i]) + delimiter + \
				'Fixed - ' + str(self.fixed[i]) + delimiter + \
				'Cleaner? - ' + str(self.cleaner[i]) + delimiter + \
				'Dirt-Cleaned? - ' + str(self.dirtClean[i]) + delimiter + \
				'Light On? - ' + str(self.lighton[i]) + delimiter + \
				'Action - ' + str(self.actions[i]) + delimiter + \
				'Constraints - ' + str(self.constraints[i]) + delimiter
			if metrics:
				string = string + 'All metric - ' + str(self.metrics) + delimiter
		return string

