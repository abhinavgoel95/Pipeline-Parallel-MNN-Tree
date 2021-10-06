import pdb
import configuration
import torch

class hierarchy_structure:
	def __init__(self, config):
		self.next = {
			'root': ['SG1', 'SG2', 'SG3'],
			'SG1': [None, None],
			'SG2': [None, None],
			'SG3': ['SG4', 'SG5', None, None],
			'SG4': [None, None],
			'SG5': [None, None],
		}
		
		self.inputs = {
			'root': (1, 3, 32, 32),
			'SG1':  (1, 32, 16, 16),
			'SG2':  (1, 32, 16, 16),
            'SG3':  (1, 32, 16, 16),
            'SG4':  (1, 64, 16, 16),
            'SG5':  (1, 64, 16, 16),
		}

		self.location = {}
		for index, assignment in enumerate(config.assignments[config.N]):
			for node in assignment:
				self.location[node] = index+1
	
	def getNext(self, node, child):
		next_child = self.next[node][child]
		if next_child == None:
			return None, None
		next_location = self.location[next_child]
		return next_child, next_location
		



if __name__ == '__main__':
	## Testing
	a = configuration.configuration(2,1, ['1','2'], ['10','20'])
	b = hierarchy_structure(a)
	print(b.getNext('SG1', 1))