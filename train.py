import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns

class	Train:
	weights = [[],[],[]]
	cost = []
	numTests = 0

	def start(self):
		random.seed();

		for i in range(0, 620):
			self.weights[0].append(random.randrange(0, 1000000) / 1000000.0)

		for i in range(0, 420):
			self.weights[1].append(random.randrange(0, 1000000) / 1000000.0)

		for i in range(0, 42):
			self.weights[2].append(random.randrange(0, 1000000) / 1000000.0)
		
		for i in range(0, 2):
			self.cost.append(0)

	def	resetCost(self, data, mal):
		self.cost[0] = 0;
		self.cost[1] = 0;
		self.numTests = 0

	def getResult(self, input):
		step2 = []
		step3 = []
		out = []

		for j in range(0, 20):
			step2.append(0)
			for i in range(0, 30):
				step2[j] += self.weights[0][j + i * 20 + 20] * input[i]
			step2[j] += self.weights[0][j]
			step2[j] = sigmoid(step2[j])

		for j in range(0, 20):
			step3.append(0)
			for i in range(0, 20):
				step3[j] += self.weights[1][j + i * 20 + 20] * step2[i]
			step3[j] += self.weights[1][j]
			step3[j] = sigmoid(step3[j])

		for j in range(0, 2):
			out.append(0)
			for i in range(0, 20):
				out[j] += self.weights[2][j + i * 2 + 2] * step3[i]
			out[j] += self.weights[2][j]
			out[j] = sigmoid(out[j])

		return out


	def	addCost(self, data, mal):
		res = Train.getResult(self, data)
		print(res)
		self.numTests = self.numTests + 1
		self.cost[0] += ((res[0] - (0 if mal == 0 else 1)) ** 2)
		self.cost[1] += ((res[1] - (1 if mal == 0 else 0)) ** 2)

def sigmoid(x):
  	return 1 / (1 + math.exp(-x))

def softmax(list):
	total = sum(list)
	for i in range(len(list)):
		list[i] /= total
	return list



df = pd.read_csv('data.csv', names = ["id", "result", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", 
	"15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30"])

max = []
min = []

for i in range (0, 30):
	max.append(0)
	min.append(2147483647)

for index, row in df.iterrows():
	list = [row["1"], row["2"], row["3"], row["4"], row["5"], row["6"], row["7"], row["8"], row["9"], row["10"], row["11"], row["12"], row["13"], row["14"], row["15"], row["16"], row["17"], row["18"], row["19"], row["20"], row["21"], row["22"], row["23"], row["24"], row["25"], row["26"], row["27"], row["28"], row["29"], row["30"]] 
	for i in range (0, 30):
		if max[i] < list[i]:
			max[i] = list[i];
		if min[i] > list[i]:
			min[i] = list[i];

f = open("processing.txt", "w")
start = []
scale = []
for i in range (0, 30):
	start.append(min[i])
	scale.append(max[i] - min[i])
	f.write(str(start[i]) + " " + str(scale[i]) + "\n")

tr = Train()

Train.start(tr)

for index, row in df.iterrows():
	list = [row["1"], row["2"], row["3"], row["4"], row["5"], row["6"], row["7"], row["8"], row["9"], row["10"], row["11"], row["12"], row["13"], row["14"], row["15"], row["16"], row["17"], row["18"], row["19"], row["20"], row["21"], row["22"], row["23"], row["24"], row["25"], row["26"], row["27"], row["28"], row["29"], row["30"]] 
	for i in range (0, 30):
		list[i] -= start[i];
		list[i] /= scale[i];

	Train.addCost(tr, list, 1 if row["result"] == 'M' else 0)
	print(tr.cost[0] / tr.numTests, tr.cost[1] / tr.numTests)
	# out = softmax(out)
	# print(out)
# plt.show()

