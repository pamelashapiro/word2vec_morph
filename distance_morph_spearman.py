''' 
python version of distance_morph.c
will also read in similarity dataset and compute Spearman coefficient
'''

import sys
import numpy as np
import csv
from scipy.stats import spearmanr

def read_in_MC_dataset(filename):

	similarity_scores = {}
	word_pairs = []

	with open(filename, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=';', quotechar='"')
		row_index = 0
		for row in reader:
			if row_index != 0:
				word_pairs.append((string1, string2))
				similarity_scores[(string1, string2)] = float(row[-1])
			row_index += 1

	return similarity_scores, word_pairs

def main():

	if len(sys.argv) < 4:
		print "Usage: python distance_morph_spearman.py <FILE> <word or word+lemma> <comparison file>"
		print "where FILE contains word projections in the BINARY FORMAT,"
		print "word compares just the word embedding,"
		print "and word+lemma compares the word embedding concatenated with a weighted average of its lemma vectors"
		print "and comparison file is an MC-style word similarity file to calculate spearman correlation with"
		print "Note: just use word+lemma parameter to read whole vector of lemma file"
		return
	file_name = sys.argv[1]
	word_lemma_opt = sys.argv[2]
	mc_file = sys.argv[3]

	similarity_scores, word_pairs = read_in_MC_dataset(mc_file)
	f = open(file_name, "rb")

	num_oovs = 0
	line_index = 0
	words = 0
	size = 0
	word_size = 0
	vocab = []
	bi1 = np.zeros(100, dtype=int)
	bi2 = np.zeros(100, dtype=int)
	M = None
	vec = None
	b = 0
	for line in f:

		fields = line.split()

		# Grab the numbers from the header of the file
		if line_index == 0:
			header_nums = [int(x) for x in fields]
			words = header_nums[0] + len(word_pairs)
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
		if Z >= 0.0000001:
			M[b] /= Z

		line_index += 1
		b += 1

	f.close()

	similarity_scores_est = {}

	for (word1, word2) in word_pairs:
		st1 = word1
		st = st1.split()
		cn = len(st)
		for a in xrange(cn):
			b = -1
			try:
				b = vocab.index(st[a])
			except ValueError:
				b = -1
			bi1[a] = b
			if b == -1:
				print st[a], " is not in the dictionary"
				num_oovs += 1
				break
		#if b == -1:
		#	continue

		vec1 = np.zeros(size)
		for b in xrange(cn):
			if bi1[b] == -1:
				continue
			vec1 += M[bi1[b]]
		Z = np.linalg.norm(vec1)
		if Z != 0.0:
			vec1 /= Z

		st2 = word2
		st = st2.split()
		cn = len(st)
		for a in xrange(cn):
			b = -1
			try:
				b = vocab.index(st[a])
			except ValueError:
				b = -1
			bi2[a] = b
			if b == -1:
				print st[a], " is not in the dictionary"
				num_oovs += 1
				break
		#if b == -1:
		#	continue


		vec2 = np.zeros(size)
		for b in xrange(cn):
			if bi2[b] == -1:
				continue
			vec2 += M[bi2[b]]
		Z = np.linalg.norm(vec2)
		if Z != 0.0:
			vec2 /= Z

		dist = np.dot(vec1,vec2)
		similarity_scores_est[(word1, word2)] = dist

	scores = []
	scores_est = []
	for (word1, word2) in similarity_scores_est:
		scores.append(similarity_scores[(word1, word2)])
		scores_est.append(similarity_scores_est[(word1, word2)])
	
	rho, pval = spearmanr(scores, scores_est)
	print "Spearman coefficient: ", rho
	print "number of oov: ", num_oovs
	print "total number of pairs: ", len(similarity_scores.keys())

main()
