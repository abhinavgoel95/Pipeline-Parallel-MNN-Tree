class communication:
	def __init__(self, N):
		self.links = {}
		port = 8080
		for i in range(1,N+1):
			for j in range(i+1, N+1):
				self.links[(i,j)] = port
				port+=1
		

	def getPort(self, index, child):
		left = min(index, child)
		right = max(index, child)
		return self.links[(left, right)]


if __name__ == '__main__':
	## Testing
	a = communication(3)
	print(a.links)
	a = communication(2)
	print(a.links)