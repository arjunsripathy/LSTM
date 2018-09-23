import numpy as np
import tensorflow as tf

QUICK_RUN = False

USE_OLD = False
MODE = 0

SAVE_EVERY_EPOCH = True

vocab = np.load("vocab.npy")
numV = len(vocab)

def encode(c):
	encoding = np.zeros([len(c),numV])
	for i in range(len(c)):
		index = np.where(vocab==bytes(c[i],encoding='ascii'))[0][0]
		e = np.zeros(numV)
		e[index]=1
		encoding[i]=e
	return encoding

if(MODE == 0):
	print("Opening File...")
	datafile = open("hp1f.txt","r")
	data = []
	print("Parsing Data...")
	if(QUICK_RUN):
		counter = 0
		counterMax = 20
		for line in datafile:
			counter+=1
			if(counter>counterMax): break
			data = np.concatenate((data,list(line)))
	else:
		for line in datafile:
			data = np.concatenate((data,list(line)))
	print("Encoding Data...")
	data = encode(data)

	numRawWords = len(data)


'''
vocab = []

def saveNewVocab():
	for i in range(len(data)):
		c = data[i]
		if(not c in vocab):
			vocab.append(c)

	np.save("vocab",vocab)
'''
numUnits = 60
numH = 100
MB_SIZE=10

SEQUENCE_LENGTH = 100

if(MODE == 0):
	numSequences = int(((numRawWords/MB_SIZE)-1)/(SEQUENCE_LENGTH))
	if(QUICK_RUN):
		numSequences = 5

cS = tf.placeholder(tf.float32,[MB_SIZE,numUnits])
hS = tf.placeholder(tf.float32,[MB_SIZE,numUnits])
state = (cS,hS)

Wlh = tf.Variable(tf.truncated_normal([numUnits,numH]))
Whs = tf.Variable(tf.truncated_normal([numH,numV]))
lstm = tf.contrib.rnn.LSTMCell(numUnits)

dB = tf.placeholder(tf.float32,[MB_SIZE,SEQUENCE_LENGTH+1,numV])

sLogits = None

for i in range(SEQUENCE_LENGTH):
	inp = dB[:,i]
	target = dB[:,i+1]

	ret = lstm(inp, state)
	output,state = ret

	hidden = tf.nn.relu(tf.matmul(output,Wlh))
	softMaxInput = tf.matmul(hidden,Whs)
	softMaxInput = tf.reshape(softMaxInput,[MB_SIZE,1,numV])
	if(i>0):
		sLogits = tf.concat([sLogits,softMaxInput],axis=1)
	else:
		sLogits = softMaxInput

(finalCS,finalHS) = state

dBLabels = dB[:,1:]

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=dBLabels,logits=sLogits))

if(MODE ==0):
	batchedData = []
	batchSize = int(numRawWords/MB_SIZE)
	for i in range(MB_SIZE):
		batch = data[i*batchSize:(i+1)*batchSize]
		batchedData.append(batch)
	batchedData = np.array(batchedData)

	LEARNING_RATE = 3e-3
	EPOCHS = 5

	print("Creating Training Algorithm...")
	train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

sess = tf.Session()

saver = tf.train.Saver()

if(USE_OLD):
	"Restoring New Variables"
	saver.restore(sess,"/tmp/tfWB.ckpt")
else:
	print("Initialzing New Variables")
	sess.run(tf.global_variables_initializer())

if(MODE ==0):

	iS = np.zeros([MB_SIZE,numUnits])

	print("Calculating Initial Loss ...")
	initialLoss = 0
	currentCS = iS
	currentHS = iS
	for i in range(numSequences):
		dataBatch = batchedData[:,i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH+1]
		l,currentCS,currentHS = sess.run([loss,finalCS,finalHS],feed_dict={dB:dataBatch,cS:currentCS,hS:currentHS})
		initialLoss+=l
	initialLoss/=numSequences
	print("INITIAL LOSS: %f"%initialLoss)

	for i in range(EPOCHS):
		print("EPOCH: %i ..."%(i+1))
		epochLoss = 0
		currentCS = iS
		currentHS = iS
		for i in range(numSequences):
			dataBatch = batchedData[:,i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH+1]
			train_step.run(feed_dict={dB:dataBatch,cS:currentCS,hS:currentHS},session=sess)
			l,currentCS,currentHS = sess.run([loss,finalCS,finalHS],feed_dict={dB:dataBatch,cS:currentCS,hS:currentHS})
			epochLoss+=l
		epochLoss/=numSequences
		print("EPOCH LOSS: %f"%epochLoss)
		print("________________________")
		if(SAVE_EVERY_EPOCH):
			print("SAVING NETWORK...")
			saver.save(sess, "/tmp/tfWB.ckpt")

	if(not SAVE_EVERY_EPOCH):
		print("SAVING NETWORK...")
		saver.save(sess, "/tmp/tfWB.ckpt")

if(MODE==1):
	first = " "

	previousLetter = tf.placeholder(tf.float32,[1,numV])
	cGenState = tf.placeholder(tf.float32,[1,numUnits])
	hGenState = tf.placeholder(tf.float32,[1,numUnits])
	genState = (cGenState,hGenState)
	output, newGenState = lstm(previousLetter,genState)
	hidden = tf.nn.relu(tf.matmul(output,Wlh))
	SMI = tf.matmul(hidden,Whs)

	generationLength = 200

	generation = ""

	prev = encode(first)
	iS = np.zeros([1,numUnits])
	currentCS = iS
	currentHS = iS

	print("Generating...")
	for i in range(generationLength):
		sotftMaxLogits,(currentCS,currentHS) = sess.run([SMI,newGenState],feed_dict={previousLetter:prev,cGenState:currentCS,hGenState:currentHS})
		logits = np.reshape(sotftMaxLogits,numV)
		unNormalizedProbs = np.exp(logits)
		unNormalizedProbs = unNormalizedProbs*unNormalizedProbs
		pDist = unNormalizedProbs/np.sum(unNormalizedProbs)
		ind = np.random.choice(range(numV),p=pDist)
		genC = vocab[ind].decode('ascii')
		generation+=genC
		prev = encode(genC)

	print(generation)





















