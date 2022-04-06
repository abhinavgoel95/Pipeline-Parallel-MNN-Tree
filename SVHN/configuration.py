import pdb

class configuration:
	def __init__(self, N, index, ip_list):
		self.assignments = {
			2: [['root'], ['SG1', 'SG2']],
			3: [['root'], ['SG1'], ['SG2']],
			4: [['root'], ['root'], ['SG1'], ['SG2']],
		}

		self.nodes = ['root', 'SG1', 'SG2']

		self.paths = {
            'root': ['root'], 
            'SG1': ['root', 'SG1'], 
            'SG2': ['root', 'SG2'], 
    	}

		assert N == len(ip_list[:-1])
		assert len(self.paths.keys()) == len(self.nodes)

		self.address = dict()
		for i, assignment in enumerate(self.assignments[N]):
			for DNN in assignment:
				self.address[DNN] = (i+1, ip_list[i])

		self.N = N
		self.index = index

	def getAssignment(self):
		DNNs = self.assignments[self.N][self.index-1]
		device_parents = set()
		device_children = set()

		for DNN in DNNs:
			if DNN != 'root':
				device_parents.add(self.paths[DNN][-2])

		for DNN in DNNs:
			for path in self.paths:
				if path != 'root':
					if self.paths[path][-2] == DNN:
						device_children.add(path)

		recv_address = []
		send_address = []

		for parentDNN in device_parents:
			if parentDNN not in DNNs:
				recv_address.append(self.address[parentDNN])

		for childDNN in device_children:
			if childDNN not in DNNs:
				send_address.append(self.address[childDNN])

		return recv_address, send_address, DNNs

if __name__ == '__main__':
	## Testing
	print("1")
	a = configuration(3,1,['1.1.1.1','2.2.2.2','3.3.3.3'])
	print(a.getAssignment())

	print("2")
	a = configuration(2,2,['1.1.1.1','2.2.2.2'])
	print(a.getAssignment())


