''' 
python version of distance_morph.c
will also read in analogies dataset and compute accuracy
'''

import sys
import numpy as np

def read_in_analogies(filename):

	unique_words = set()

	analogies = []
	with open(filename, 'r') as f:
		for line in f:
			tatweel = u"\u0640"
			normalized = line.replace(tatweel.encode('utf-8'),"")
			fields = normalized.strip().split()
			unique_words.update(set(fields))
			analogy = []
			for i in xrange(0, len(fields), 2):
				pair = (fields[i], fields[i+1])
				analogy.append(pair)
			analogies.append(analogy)

	print len(unique_words), " unique words"

	return analogies

def main():

	if len(sys.argv) < 5:
		print "Usage: python distance_morph_analogies.py <FILE> <word or word+lemma> <analogies file> <N>"
		print "where FILE contains word projections in the BINARY FORMAT,"
		print "word compares just the word embedding,"
		print "and word+lemma compares the word embedding concatenated with a weighted average of its lemma vectors"
		print "and analogies file contains white-space-separated pairs of words in a relationship"
                print " and N is the number of similar words you want to consider"
		print "Note: just use word+lemma parameter to read whole vector of lemma file"
		return
	file_name = sys.argv[1]
	word_lemma_opt = sys.argv[2]
	analogies_file = sys.argv[3]

	analogies = read_in_analogies(analogies_file)
	f = open(file_name, 'r')
	print "Read in analogies"

	N = int(sys.argv[4])
	num_oovs = [0, 0, 0, 0]
	line_index = 0
	words = 0
	size = 0
	word_size = 0
	vocab = []
	bestw = []
	bestd = []
	b1 = 0
	b2 = 0
	b3 = 0
	b4 = 0
	b = 0
	M = None
	vec = None
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
		if Z < 0.0000001:
			print M[b]
			exit(0)
		M[b] /= Z

		line_index += 1
		b += 1

	f.close()

	num_correct = 0
	analogy_index = 0

	for analogy in analogies:

		print "On analogy #", analogy_index
		analogy_index += 1

		avg_diff = np.zeros(size)

		for (word1, word2) in analogy[:-1]:
			print "word1: ", word1, "word2: ", word2

			try:
				b1 = vocab.index(word1)
			except ValueError:
				b1 = -1
			if b1 == -1:
				print word1, " is not in the dictionary"
				num_oovs[0] += 1
				continue

			vec1 = M[b1]
			Z = np.linalg.norm(vec1)
			if Z != 0.0:
				vec1 /= Z

			try:
				b2 = vocab.index(word2)
			except ValueError:
				b2 = -1
			if b2 == -1:
				print word2, " is not in the dictionary"
				num_oovs[1] += 1
				continue

			vec2 = M[b2]
			Z = np.linalg.norm(vec2)
			if Z != 0.0:
				vec2 /= Z

			avg_diff += vec2 - vec1

		if b1 == -1 or b2 == -1:
			continue

		avg_diff /= len(analogy)
		Z = np.linalg.norm(avg_diff)
		if Z != 0.0:
			avg_diff /= Z

		try:
			b3 = vocab.index(analogy[-1][0])
		except ValueError:
			b3 = -1
		if b3 == -1:
			print analogy[-1][0], " is not in the dictionary"
			num_oovs[2] += 1
			continue

		vec3 = M[b3]
		Z = np.linalg.norm(vec3)
		if Z != 0.0:
			vec3 /= Z

		for a in xrange(N):
			bestd.append(0)
			bestw.append("")

		vec = avg_diff + vec3
		Z = np.linalg.norm(vec)
		vec /= Z
		for a in xrange(N):
			bestd[a] = -1
			bestw[a] = ""
		for c in xrange(words):
			a = 0
			if b1 == c or b2 == c or b3 == c: 
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
			print "my guess: ", bestw[a]
			print "actual: ", analogy[-1][1]

			try:
				b4 = vocab.index(analogy[-1][1])
			except ValueError:
				b4 = -1
			if b4 == -1:
				print analogy[-1][1], " is not in the dictionary"
				num_oovs[3] += 1

			if bestw[a] == analogy[-1][1]:
				num_correct += 1
				break

	print num_correct, " correct out of ", len(analogies)
	print num_oovs, " OOV words"
	print "i.e., " + str(float(num_correct) / len(analogies) * 100) + "%"

main()
