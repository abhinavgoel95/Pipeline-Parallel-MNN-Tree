import pdb
import configuration

class hierarchy_structure:
	def __init__(self, config):
		self.next = {
			'root': [None, 'SG1', None, 'SG2', None, None],
			'SG1': [None, None],
			'SG2': [None, None, None, None],
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