class communication:
	def __init__(self, N):
		self.links = {}
		port = 8080
		for j in range(1, N+1):
			self.links[(1,j)] = port
			port+=1
		

	def getPort(self, child):
		return self.links[(1, child)]


if __name__ == '__main__':
	## Testing
	a = communication(3)
	print(a.links)
	a = communication(2)
	print(a.links)