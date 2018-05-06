''' 
python version of distance_morph.c
will also read in similarity dataset and compute Spearman coefficient
'''

import sys
import numpy as np

max_size = 2000         # max length of strings
N = 40                  # number of closest words that will be shown
max_w = 50              # max length of vocabulary entries

if len(sys.argv) < 3:
	print "Usage: python column_normalize.py <input_file> <output_file>"
	print "where input_file contains word projections in the the text format,"
	print "and output_file is the destination for column-normalized projections,"
	exit(0)

input_file = sys.argv[1]
output_file = sys.argv[2]
f_i = open(input_file, "rb")
f_o = open(output_file, "w")

line_index = 0
words = 0
size = 0
vocab = []
bestw = []
bestd = []
bi = np.zeros(100, dtype=int)
M = None
means = None
variances = None
vec = None
b = 0
for line in f_i:

	fields = line.split()

	# Grab the numbers from the header of the file
	if line_index == 0:
		f_o.write(line)
		header_nums = [int(x) for x in fields]
		words = header_nums[0]
		size = header_nums[1]
		M = np.zeros((words, size))
		means = np.zeros(size, dtype=float)
		variances = np.zeros(size, dtype=float)
		line_index += 1	
		continue

	vocab.append(fields[0])

	# read in vector
	float_fields = [float(x) for x in fields[1:]]

	M[b] = float_fields

	# add row to means
	means += M[b]

	line_index += 1
	b += 1

means /= float(words)
#print means
for b in xrange(words):
	variances += np.square(M[b] - means)
variances /= float(words)
std_devs = np.sqrt(variances)
#print std_devs
for b in xrange(words):
	#if b == 0:
		#print M[b]
		#print means
		#print M[b] - means
		#print std_devs
	M[b] = np.divide((M[b] - means), std_devs)
	#if b == 0:
		#print M[b]
	f_o.write(vocab[b] + " " + " ".join([str(x) for x in M[b].tolist()]) + "\n")

f_i.close()
f_o.close()
