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
	print "Usage: python distance_morph.py <FILE> <word or word+lemma or lemma>"
	print "where FILE contains word projections in the BINARY FORMAT,"
	print "word compares just the word embedding,"
	print "and word+lemma compares the word embedding concatenated with a weighted average of its lemma vectors"

	print "Note: just use word+lemma parameter to read whole vector of lemma file"
	exit(0)
file_name = sys.argv[1]
word_lemma_opt = sys.argv[2]
f = open(file_name, "rb")

line_index = 0
words = 0
size = 0
word_size = 0
vocab = []
bestw = []
bestd = []
bi = np.zeros(100, dtype=int)
M = None
vec = None
b = 0
for line in f:

	fields = line.split()

	# Grab the numbers from the header of the file
	if line_index == 0:
		header_nums = [int(x) for x in fields]
		words = header_nums[0]
		size = header_nums[1]
		M = np.zeros((words, size))
		if word_lemma_opt != "word+lemma":
			if len(header_nums) < 3:
				sys.exit("Passed a file with not enough header arguments")
			word_size = header_nums[2]
		line_index += 1	
		continue

	vocab.append(fields[0])

	# read in vector
	float_fields = [float(x) for x in fields[1:]]
	
	# if we only want to compare word, stop at word_size, fill the rest with zero
	if word_lemma_opt == "word":
		for a in xrange(size):
			if a >= word_size:
				float_fields[a] = 0.0

	# if we only want to compare lemma, fill with zero until word_size
	if word_lemma_opt == "lemma":
		for a in xrange(size):
			if a < word_size:
				float_fields[a] = 0.0

	M[b] = float_fields

	# normalize
	Z = np.linalg.norm(M[b])
	M[b] /= Z

	line_index += 1
	b += 1

f.close()

while True:
	for a in xrange(N):
		bestd.append(0)
		bestw.append("")
	st1 = raw_input("Enter word or sentence (EXIT to break): ")
	if st1 == "EXIT":
		break
	st = st1.split()
	cn = len(st)
	for a in xrange(cn):
		reached_end = True
		for b in xrange(words):
			if vocab[b] == st[a]:
				reached_end = False
				break
		if reached_end:
			b = -1
		bi[a] = b
		print "\nWord: ", st[a], " Position in vocabulary: ", bi[a]
		if b == -1:
			print "Out of dictionary word!"
			break
	if b == -1:
		continue
	print "\n                                              Word       Cosine distance\n------------------------------------------------------------------------"
	vec = np.zeros(size)
	for b in xrange(cn):
		if bi[b] == -1:
			continue
		vec += M[bi[b]]
	Z = np.linalg.norm(vec)
	vec /= Z
	for a in xrange(N):
		bestd[a] = -1
		bestw[a] = ""
	for c in xrange(words):
		a = 0
		for b in xrange(cn):
			if bi[b] == c: 
				a = 1
		if a == 1:
			continue
		dist = np.dot(vec,M[c])
		for a in xrange(N):
			if dist > bestd[a]:
				for d in xrange(N - 1, a, -1):
					bestd[d] = bestd[d - 1]
					bestw[d] = bestw[d - 1]
				bestd[a] = dist
				bestw[a] = vocab[c]
				break
	for a in xrange(N):
		print "%50s\t\t%f" % (bestw[a], bestd[a])