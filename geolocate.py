#!/usr/bin/env python3
# geolocate.py : Tweet location classifier
# Ankit Mathur, Nishant Jain and Nitesh Jaswal October 2018

# Formulation of the NB problem:
# Cleaned the training and testing files according to the post on Piazza
# Convert all words to lowercase as the bag-of-words is independent of the case
# Ignoring all punctuations and symbols
# Applying a cutoff of 3 words. That is ignoring all words that occur 3 or less than 3 times
# Applied Laplacian smoothening while calculating posterior probability for words that do not exist in the dictionary of a city by assigning them a value of 0.01
# Found the top five words of a given city by calculating the probability of the word in the given city and dividing it by the probability of finding the word in the entire dataset
# While calulating the posterior probabliltes we are taking the log of priors and summing them because we noticed that the posteriors were reaching an order of 10^-42 and below

import sys
from math import log

# Prints a dictionary
def print_dict(dict):
	for item in dict.keys():
		print(item, dict[item])

# Print the dictionaries of a dictionary(We have some dictionaries that themselves store dictionaries in our code)
def print_dict_of_dict(dict_of_dict):
	for item in dict_of_dict.keys():
		print(item)
		print_dict(dict_of_dict[item])

# Function to check if the word has a punctuation mark and  if it does remove the punctuation mark
def remove_punct(word):
	word = word.lower()
	ls_word = []

	for char in word:
		if ord(char) >= 97 and ord(char) <= 122:
			ls_word += char

	new_word = ''.join(ls_word)
	return new_word

# Read the training file and calculate the the prior_probabilities of word given city and the probability of location depending on the number of tweets from that location
# It also calculates and returns the global frequency of words. That is, the frequency of theword in the entire training data
def read_train(training_file):
	global restricted_words
	freq_w_by_l = {}
	freq_location = {}
	global_freq = {}
	
	with open(training_file, 'r') as file:
		for line in file:
			city, tweet = line.split()[0], line.split()[1:]
			for word in tweet:
				word = remove_punct(word)
				if len(word) > 0 and word not in restricted_words:
				# Initialize with 2 because by default, even if a word doesn't exist, its freq is 1
					if city in freq_w_by_l.keys():
						freq_w_by_l[city][word] = (freq_w_by_l[city][word] + 1) if word in freq_w_by_l[city].keys() else 2
					else:
						freq_w_by_l[city] = {word : 1}
					if word in global_freq:
						global_freq[word] += 1
					else:
						global_freq[word] = 1
			freq_location[city] = (freq_location[city] + 1) if city in freq_location.keys() else 1

	return freq_w_by_l, freq_location, global_freq

# This function converts the frequencies of words to probabilities by dividing by the total count of distinct words for that city
# It also filters the words by cutoff
def freq_to_prob(dict_freq, cutoff):
	# compute denominator
	tot_freq = 0
	for item in dict_freq.keys():
		tot_freq += dict_freq[item] if dict_freq[item] >= cutoff else 1

	# compute individual probablities
	dict_prob = {}
	for item in dict_freq:
		if dict_freq[item] >= cutoff:
			dict_prob[item] = dict_freq[item] / tot_freq

	return dict_prob

# This function simply calculates the total number of distinct words for each city
def calc_tot_freq(freq_w_by_l):
	total_freq_dict = {}
	for city in freq_w_by_l:
		sum = 1
		for word in freq_w_by_l[city].keys():
			sum += freq_w_by_l[city][word] + 0.01
		total_freq_dict[city] = sum
	return total_freq_dict

# Calculates and returns the prior probabilities
def priors(freq_location, freq_w_by_l, cutoff):
	priors_location = freq_to_prob(freq_location, 0)
	
	total_freq_dict = calc_tot_freq(freq_w_by_l)
	
	priors_w_by_l = freq_w_by_l
	for city in freq_w_by_l:
		for word in freq_w_by_l[city]:
			priors_w_by_l[city][word] = (freq_w_by_l[city][word] if freq_w_by_l[city][word] >= cutoff else 0.01)/total_freq_dict[city]

	return priors_w_by_l, priors_location

# This function calculates the log of posterior probabilities. Words that don't exist for a city are smoothed and assigned a value of 0.01 instead of 0
# Calculates and returns the accuracy of our predictions.
# Writes the output to an output file according to the format mentioned in the problem statement
def output_test(testing_file, priors_location, priors_w_by_l, total_freq_dict):
	tot_count = 0
	correct_count = 0
	prior_prob = 1
	
	with open(testing_file, 'r') as file:
		f = open("output-file.txt", "w")
		for line in file:
			tot_count += 1

			actual_city, tweet = line.split()[0], line.split()[1:]
			posteriors_location = {}

			for city in priors_w_by_l.keys():
				posteriors_location[city] = 1
				words_found = 0
				for word in tweet:
					word = remove_punct(word)
					if len(word) > 0:				
						words_found += 1
						if word in priors_w_by_l[city]:			
							prior_prob = priors_w_by_l[city][word]
							posteriors_location[city] += log(prior_prob)
						else:
							prior_prob = (0.01/total_freq_dict[city])
							posteriors_location[city] += log(prior_prob)
				
				posteriors_location[city] += log(priors_location[city])
	
			predicted_city = max(posteriors_location, key=lambda city: posteriors_location[city])
			correct_count += 1 if predicted_city == actual_city else 0

			f.write( predicted_city + " " + actual_city + " " + ' '.join(tweet) + "\n")
		f.close()

		accuracy = correct_count/tot_count * 100

		return accuracy

def predict(training_file, testing_file):
	cutoff = 3
	freq_w_by_l, freq_location, global_freq = read_train(training_file)
	total_freq_dict = calc_tot_freq(freq_w_by_l)
	
	priors_w_by_l, priors_location = priors(freq_location, freq_w_by_l, cutoff)
	accuracy = output_test(testing_file, priors_location, priors_w_by_l, total_freq_dict)
	print("Accuracy: %3.2f%%" % accuracy )

# Finds the top five words for each city according to the Bayes Rule by dividing the priors of that word in the city with the global frequency of that word. 
# We can ignore multiplying the prior probability of the location itself since it will be the same for each word in that city
# It stores the calculated value for each word in the city in a list of tuples. Once the entire list has been generated for the city, it sorts them in descending order of
# probabilities and simply selecteing the first 5 and printing them
	for city in priors_w_by_l:
		print("Top five words in city ", city, " are:-")
		top_five = []
		for word in priors_w_by_l[city]:
			top_five += [(priors_w_by_l[city][word]/global_freq[word], word)]
		top_five.sort(reverse=True)
		top_five = top_five[0:5]
		for i in range(0, 5):
			print(top_five[i][1], end=" ")
		print("\n")
	return None

restricted_words = []
training_file, testing_file = sys.argv[1], sys.argv[2]
predict(training_file, testing_file)

