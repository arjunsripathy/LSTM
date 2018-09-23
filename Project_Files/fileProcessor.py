import numpy as np

datafile = open("hp1f.txt","r")
data = []
for line in datafile:
	lt = line.strip()
	data.append(lt)

print(len(data))

i=0
while i < len(data):
	l = data[i]
	if(l==""):
		data = np.delete(data,i)
		i-=1
	i+=1
	
print(len(data))

page = 1
i=0
while i<len(data):
	l = data[i]
	if(l==str(page)):
		page +=1
		data = np.delete(data,i)
		i-=1
	i+=1

print(len(data))

cData = []
thisLine = ""

cutOff = 72-10
i=0
while i <len(data)-1:
	l = data[i]
	thisLine += l
	lC = l[len(l)-1]
	nl = data[i+1]
	if(lC == "\"" or len(l)<cutOff):
		thisLine+="\n"
		cData.append(thisLine)
		thisLine = ""
	else:
		thisLine +=" "
	i+=1

print(len(cData))


newFile = open("new.txt","w")
for i in range(len(cData)):
	newFile.write(cData[i])
