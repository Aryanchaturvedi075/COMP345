import json
from collections import Counter
import numpy as np
import pandas as pd
import re
import nltk
from nltk.data import find
import gensim
import sklearn
from sympy.parsing.sympy_parser import parse_expr

np.random.seed(0)
nltk.download('word2vec_sample')

########-------------- PART 1: LANGUAGE MODELING --------------########

class NgramLM:
	def __init__(self):
		"""
		N-gram Language Model
		"""
		# Dictionary to store next-word possibilities for bigrams. Maintains a list for each bigram.
		self.bigram_prefix_to_trigram = {}
		
		# Dictionary to store counts of corresponding next-word possibilities for bigrams. Maintains a list for each bigram.
		self.bigram_prefix_to_trigram_weights = {}

	def load_trigrams(self):
		"""
		Loads the trigrams from the data file and fills the dictionaries defined above.

		Parameters
		----------

		Returns
		-------
		"""
		with open("data/tweets/covid-tweets-2020-08-10-2020-08-21.trigrams.txt") as f:
			lines = f.readlines()
			for line in lines:
				word1, word2, word3, count = line.strip().split()
				if (word1, word2) not in self.bigram_prefix_to_trigram:
					self.bigram_prefix_to_trigram[(word1, word2)] = []
					self.bigram_prefix_to_trigram_weights[(word1, word2)] = []
				self.bigram_prefix_to_trigram[(word1, word2)].append(word3)
				self.bigram_prefix_to_trigram_weights[(word1, word2)].append(int(count))

	def top_next_word(self, word1, word2, n=10):
		"""
		Retrieve top n next words and their probabilities given a bigram prefix.

		Parameters
		----------
		word1: str
			The first word in the bigram.
		word2: str
			The second word in the bigram.
		n: int
			Number of words to return.
			
		Returns
		-------
		next_words: list
			The retrieved top n next words.
		probs: list
			The probabilities corresponding to the retrieved words.
		"""
		next_words = []
		probs = []

		# write your code here
		pass

		return next_words, probs
	
	def sample_next_word(self, word1, word2, n=10):
		"""
		Sample n next words and their probabilities given a bigram prefix using the probability distribution defined by frequency counts.

		Parameters
		----------
		word1: str
			The first word in the bigram.
		word2: str
			The second word in the bigram.
		n: int
			Number of words to return.
			
		Returns
		-------
		next_words: list
			The sampled n next words.
		probs: list
			The probabilities corresponding to the retrieved words.
		"""
		next_words = []
		probs = []

		# write your code here
		pass

		return next_words, probs
	
	def generate_sentences(self, prefix, beam=10, sampler=top_next_word, max_len=20):
		"""
		Generate sentences using beam search.

		Parameters
		----------
		prefix: str
			String containing two (or more) words separated by spaces.
		beam: int
			The beam size.
		sampler: Callable
			The function used to sample next word.
		max_len: int
			Maximum length of sentence (as measure by number of words) to generate (excluding "<EOS>").
			
		Returns
		-------
		sentences: list
			The top generated sentences
		probs: list
			The probabilities corresponding to the generated sentences
		"""
		sentences = []
		probs = []

		# write your code here
		pass

		return sentences, probs







#####------------- CODE TO TEST YOUR FUNCTIONS FOR LANGUAGE MODELING

print("======================================================================")
print("Checking Language Model")
print("======================================================================")

# Define your language model object
language_model = NgramLM()
# Load trigram data
language_model.load_trigrams()

print("------------- Evaluating top next word prediction -------------")
next_words, probs = language_model.top_next_word("middle", "of", 10)
for word, prob in zip(next_words, probs):
	print(word, prob)
# Your first 5 lines of output should be exactly:
# a 0.807981220657277
# the 0.06948356807511737
# pandemic 0.023943661971830985
# this 0.016901408450704224
# an 0.0107981220657277

print("------------- Evaluating sample next word prediction -------------")
next_words, probs = language_model.sample_next_word("middle", "of", 10)
for word, prob in zip(next_words, probs):
	print(word, prob)
# My first 5 lines of output look like this: (YOUR OUTPUT CAN BE DIFFERENT!)
# a 0.807981220657277
# pandemic 0.023943661971830985
# august 0.0018779342723004694
# stage 0.0018779342723004694
# an 0.0107981220657277

print("------------- Evaluating beam search -------------")
sentences, probs = language_model.generate_sentences(prefix="<BOS1> <BOS2> trump", beam=10, sampler=language_model.top_next_word)
for sent, prob in zip(sentences, probs):
	print(sent, prob)
print("#########################\n")
# Your first 3 lines of output should be exactly:
# <BOS1> <BOS2> trump eyes new unproven coronavirus treatment URL <EOS> 0.00021893147502903603
# <BOS1> <BOS2> trump eyes new unproven coronavirus cure URL <EOS> 0.0001719607222046247
# <BOS1> <BOS2> trump eyes new unproven virus cure promoted by mypillow ceo over unproven therapeutic URL <EOS> 9.773272077557522e-05

sentences, probs = language_model.generate_sentences(prefix="<BOS1> <BOS2> biden", beam=10, sampler=language_model.top_next_word)
for sent, prob in zip(sentences, probs):
	print(sent, prob)
print("#########################\n")
# Your first 3 lines of output should be exactly:
# <BOS1> <BOS2> biden calls for a 30 bonus URL #cashgem #cashappfriday #stayathome <EOS> 0.0002495268686322749
# <BOS1> <BOS2> biden says all u.s. governors should mandate masks <EOS> 1.6894510541025754e-05
# <BOS1> <BOS2> biden says all u.s. governors question cost of a pandemic <EOS> 8.777606198953028e-07

sentences, probs = language_model.generate_sentences(prefix="<BOS1> <BOS2> trump", beam=10, sampler=language_model.sample_next_word)
for sent, prob in zip(sentences, probs):
	print(sent, prob)
print("#########################\n")
# My first 3 lines of output look like this: (YOUR OUTPUT CAN BE DIFFERENT!)
# <BOS1> <BOS2> trump eyes new unproven coronavirus treatment URL <EOS> 0.00021893147502903603
# <BOS1> <BOS2> trump eyes new unproven coronavirus cure URL <EOS> 0.0001719607222046247
# <BOS1> <BOS2> trump eyes new unproven virus cure promoted by mypillow ceo over unproven therapeutic URL <EOS> 9.773272077557522e-05

sentences, probs = language_model.generate_sentences(prefix="<BOS1> <BOS2> biden", beam=10, sampler=language_model.sample_next_word)
for sent, prob in zip(sentences, probs):
	print(sent, prob)
# My first 3 lines of output look like this: (YOUR OUTPUT CAN BE DIFFERENT!)
# <BOS1> <BOS2> biden is elected <EOS> 0.001236227651321991
# <BOS1> <BOS2> biden dropping ten points given trump a confidence trickster URL <EOS> 5.1049579351466146e-05
# <BOS1> <BOS2> biden dropping ten points given trump four years <EOS> 4.367575122292103e-05







########-------------- PART 2: Semantic Parsing --------------########

class Text2SQLParser:
	def __init__(self):
		"""
		Basic Text2SQL Parser. This module just attempts to classify the user queries into different "categories" of SQL queries.
		"""
		self.parser_files = "data/semantic-parser"
		self.word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
		self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(self.word2vec_sample, binary=False)

		self.train_file = "sql_train.tsv"
		self.test_file = "sql_val.tsv"

	def load_data(self):
		"""
		Load the data from file.

		Parameters
		----------
			
		Returns
		-------
		"""
		self.train_df = pd.read_csv(self.parser_files + "/" + self.train_file, sep="\t")
		self.test_df = pd.read_csv(self.parser_files + "/" + self.test_file, sep="\t")

		self.ls_labels = list(self.train_df["Label"].unique())

	def predict_label_using_keywords(self, question):
		"""
		Predicts the label for the question using custom-defined keywords.

		Parameters
		----------
		question: str
			The question whose label is to be predicted.
			
		Returns
		-------
		label: str
			The predicted label.
		"""
		# Fill in your code here.
		label = ""
		
		return label

	def evaluate_accuracy(self, prediction_function_name):
		"""
		Gives label wise accuracy of your model.

		Parameters
		----------
		prediction_function_name: Callable
			The function used for predicting labels.
			
		Returns
		-------
		accs: dict
			The accuracies of predicting each label.
		main_acc: float
			The overall average accuracy
		"""
		correct = Counter()
		total = Counter()
		main_acc = 0
		main_cnt = 0
		for i in range(len(self.test_df)):
			q = self.test_df.loc[i]["Question"].split(":")[1].split("|")[0].strip()
			gold_label = self.test_df.loc[i]['Label']
			if prediction_function_name(q) == gold_label:
				correct[gold_label] += 1
				main_acc += 1
			total[gold_label] += 1
			main_cnt += 1
		accs = {}
		for label in self.ls_labels:
			accs[label] = (correct[label]/total[label])*100
		return accs, 100*main_acc/main_cnt

	def get_sentence_representation(self, sentence):
		"""
		Gives the average word2vec representation of a sentence.

		Parameters
		----------
		sentence: str
			The sentence whose representation is to be returned.
			
		Returns
		-------
		sentence_vector: np.ndarray
			The representation of the sentence.
		"""
		# Fill in your code here
		sentence_vector = np.zeros(300)

		return sentence_vector
	
	def init_ml_classifier(self):
		"""
		Initializes the ML classifier.

		Parameters
		----------
			
		Returns
		-------
		"""
		# Fill in your code here
		self.classifier = None
	
	def train_label_ml_classifier(self):
		"""
		Train the classifier.

		Parameters
		----------
			
		Returns
		-------
		"""
		# Fill in your code here
		pass
	
	def predict_label_using_ml_classifier(self, question):
		"""
		Predicts the label of the question using the classifier.

		Parameters
		----------
		question: str
			The question whose label is to be predicted.
			
		Returns
		-------
		predicted_label: str
			The predicted label.
		"""
		# Fill in your code here
		predicted_label = ""

		return predicted_label






class MusicAsstSlotPredictor:
	def __init__(self):
		"""
		Slot Predictor for the Music Assistant.
		"""
		self.parser_files = "data/semantic-parser"
		self.train_data = []
		self.test_questions = []
		self.test_answers = []

		self.slot_names = set()

	def load_data(self):
		"""
		Load the data from file.

		Parameters
		----------
			
		Returns
		-------
		"""
		with open(f'{self.parser_files}/music_asst_train.txt') as f:
			lines = f.readlines()
			for line in lines:
				self.train_data.append(json.loads(line))

		with open(f'{self.parser_files}/music_asst_val_ques.txt') as f:
			lines = f.readlines()
			for line in lines:
				self.test_questions.append(json.loads(line))

		with open(f'{self.parser_files}/music_asst_val_ans.txt') as f:
			lines = f.readlines()
			for line in lines:
				self.test_answers.append(json.loads(line))
	
	def get_slots(self):
		"""
		Get all the unique slots.

		Parameters
		----------
			
		Returns
		-------
		"""
		for sample in self.train_data:
			for slot_name in sample['slots']:
				self.slot_names.add(slot_name)
	
	def predict_slot_values(self, question):
		"""
		Predicts the values for the slots.

		Parameters
		----------
		question: str
			The question for which the slots are to be predicted.
			
		Returns
		-------
		slots: dict
			The predicted slots.
		"""
		words = question.split()
		slots = {}
		for slot_name in self.slot_names:
			slots[slot_name] = None
		for slot_name in self.slot_names:
			# Fill in your code to idenfity the slot value. By default, they are initialized to None.
			pass
		return slots
	
	def get_confusion_matrix(self, slot_prediction_function, questions, answers):
		"""
		Find the true positive, true negative, false positive, and false negative examples with respect to the prediction of a slot being active or not (irrespective of value assigned).

		Parameters
		----------
		slot_prediction_function: Callable
			The function used for predicting slot values.
		questions: list
			The test questions
		answers: list
			The ground-truth test answers
			
		Returns
		-------
		tp: dict
			The indices of true positive examples are listed for each slot
		fp: dict
			The indices of false positive examples are listed for each slot
		tn: dict
			The indices of true negative examples are listed for each slot
		fn: dict
			The indices of false negative examples are listed for each slot
		"""
		tp = {}
		fp = {}
		tn = {}
		fn = {}
		for slot_name in self.slot_names:
			tp[slot_name] = []
		for slot_name in self.slot_names:
			fp[slot_name] = []
		for slot_name in self.slot_names:
			tn[slot_name] = []
		for slot_name in self.slot_names:
			fn[slot_name] = []
		for i, question in enumerate(questions):
			# Fill in your code here
			pass
		return tp, fp, tn, fn
	
	def evaluate_slot_prediction_recall(self, slot_prediction_function):
		"""
		Evaluates the recall for the slot predictor. Note: This also takes into account the exact value predicted for the slot 
		and not just whether the slot is active like in the get_confusion_matrix() method

		Parameters
		----------
		slot_prediction_function: Callable
			The function used for predicting slot values.
			
		Returns
		-------
		accs: dict
			The recall for predicting the value for each slot.
		"""
		correct = Counter()
		total = Counter()
		# predict slots for each question
		for i, question in enumerate(self.test_questions):
			i = self.test_questions.index(question)
			gold_slots = self.test_answers[i]['slots']
			predicted_slots = slot_prediction_function(question)
			for name in self.slot_names:
				if name in gold_slots:
					total[name] += 1.0
					if predicted_slots.get(name, None) != None and predicted_slots.get(name).lower() == gold_slots.get(name).lower():
						correct[name] += 1.0
		accs = {}
		for name in self.slot_names:
			accs[name] = (correct[name] / total[name]) * 100
		return accs









class MathParser:
	def __init__(self):
		"""
		Math Word Problem Solver.
		"""
		self.parser_files = "data/semantic-parser"
		self.word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
		self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(self.word2vec_sample, binary=False)

		self.train_file = "math_train.tsv"
		self.test_file = "math_val.tsv"

	def load_data(self):
		"""
		Load the data from file.

		Parameters
		----------
			
		Returns
		-------
		"""
		self.train_df = pd.read_csv(self.parser_files + "/" + self.train_file, sep="\t")
		self.test_df = pd.read_csv(self.parser_files + "/" + self.test_file, sep="\t")
	
	def init_model(self):
		"""
		Initializes the ML classifier.

		Parameters
		----------
			
		Returns
		-------
		"""
		# Fill in your code here
		pass

	def predict_equation_from_question(self, question):
		"""
		Predicts the equation for the question.

		Parameters
		----------
		question: str
			The question whose equation is to be predicted.
			
		Returns
		-------
		equation: str
			The predicted equation.
		"""
		# Fill in your code here.
		eq = ""

		pass

		return eq
	
	def ans_evaluator(self, equation):
		"""
		Parses the equation to obtain the final answer.

		Parameters
		----------
		equation: str
			The equation to be parsed.
			
		Returns
		-------
		final_ans: float
			The final answer.
		"""
		try:
			final_ans = parse_expr(equation, evaluate = True)
		except:
			final_ans = -1000.112
		return final_ans
	
	def evaluate_accuracy(self, prediction_function_name):
		"""
		Gives accuracy of your model.

		Parameters
		----------
		prediction_function_name: Callable
			The function used for predicting equations.
			
		Returns
		-------
		main_acc: float
			The overall average accuracy
		"""
		acc = 0
		tot = 0
		for i in range(len(self.test_df)):
			ques = self.test_df.loc[i]["Question"]
			gold_ans = self.test_df.loc[i]["Answer"]
			pred_eq = prediction_function_name(ques)
			pred_ans = self.ans_evaluator(pred_eq)

			if abs(gold_ans - pred_ans) < 0.1:
				acc += 1
			tot += 1
		return 100*acc/tot









#####------------- CODE TO TEST YOUR FUNCTIONS FOR SEMANTIC PARSING

print()
print()

### PART 1: Text2SQL Parser

print("======================================================================")
print("Checking Text2SQL Parser")
print("======================================================================")

# Define your text2sql parser object
sql_parser = Text2SQLParser()

# Load the data files
sql_parser.load_data()

# Initialize the ML classifier
sql_parser.init_ml_classifier()

# Train the classifier
sql_parser.train_label_ml_classifier()

# Evaluating the keyword-based label classifier. 
print("------------- Evaluating keyword-based label classifier -------------")
accs, _ = sql_parser.evaluate_accuracy(sql_parser.predict_label_using_keywords)
for label in accs:
	print(label + ": " + str(accs[label]))

# Evaluate the ML classifier
print("------------- Evaluating ML classifier -------------")
sql_parser.train_label_ml_classifier()
_, overall_acc = sql_parser.evaluate_accuracy(sql_parser.predict_label_using_ml_classifier)
print("Overall accuracy: ", str(overall_acc))

print()
print()


### PART 2: Music Assistant Slot Predictor

print("======================================================================")
print("Checking Music Assistant Slot Predictor")
print("======================================================================")

# Define your semantic parser object
semantic_parser = MusicAsstSlotPredictor()
# Load semantic parser data
semantic_parser.load_data()

# Look at the slots
print("------------- slots -------------")
semantic_parser.get_slots()
print(semantic_parser.slot_names)

# Evaluate slot predictor
# Our reference implementation got these numbers on the validation set. You can ask others on Slack what they got.
# playlist_owner: 100.0
# music_item: 100.0
# entity_name: 16.666666666666664
# artist: 14.285714285714285
# playlist: 52.94117647058824
print("------------- Evaluating slot predictor -------------")
accs = semantic_parser.evaluate_slot_prediction_recall(semantic_parser.predict_slot_values)
for slot in accs:
	print(slot + ": " + str(accs[slot]))

# Evaluate Confusion matrix examples
print("------------- Confusion matrix examples -------------")
tp, fp, tn, fn = semantic_parser.get_confusion_matrix(semantic_parser.predict_slot_values, semantic_parser.test_questions, semantic_parser.test_answers)
print(tp)
print(fp)
print(tn)
print(fn)

print()
print()


### PART 3. Math Equation Predictor

print("======================================================================")
print("Checking Math Parser")
print("======================================================================")

# Define your math parser object
math_parser = MathParser()

# Load the data files
math_parser.load_data()

# Initialize and train the model
math_parser.init_model()

# Get accuracy
print("------------- Accuracy of Equation Prediction -------------")
acc = math_parser.evaluate_accuracy(math_parser.predict_equation_from_question)
print("Accuracy: ", acc)