"""
  Evaluate efficacy of wiki LSA on Allen AI dataset

  Use latent vectors to calculate cosine distance between Questions/Answers
  Pick the answer with the smallest cosine distance
  Check against correct answer
  Measure accuracy by simple
"""
import argparse
import logging
import random

from gensim.matutils import cossim
from gensim.models import LsiModel

logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO
)

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="path to word2vec/model/timestamp.model")
parser.add_argument("-d", "--data", help="path to training.tsv")
args = parser.parse_args()

# Load model
# Note - model contains dictionary that intentionally omits stopwords
model = LsiModel.load(args.model, mmap='r')

# Load 'training' data
training_data = open(args.data)
training_data.readline()  # advance past header line

correct = 0
total = 0

for line in training_data:
    elements = line.split("\t")
    question_id = elements.pop(0)
    correct_answer = elements.pop(1)

    # Get bag-of-words representation of question and answers
    doc_vectors = [model.id2word.doc2bow(element.split()) for element in elements]
    question = doc_vectors.pop(0)

    # Generate list of tuples:
    # (Cosine similarity, mapped index 0-3 to A-D)
    similarities = [(cossim(model[question], model[answer]), chr(idx + 65)) for idx, answer in
                    enumerate(doc_vectors)]
    chosen_answer = max(similarities)[1]

    correct += correct_answer == chosen_answer
    total += 1

    if not (correct_answer == chosen_answer) and (random.random() > 0.99):
        print("Question: %s" % elements.pop(0))
        print("Correct Answer: %s" % elements[ord(correct_answer) - 65])
        print("Correct Answer similarity: %.4f" % similarities[ord(correct_answer) - 65][0])
        print("Chosen Answer: %s" % elements[ord(chosen_answer) - 65])
        print("Correct Answer similarity: %.4f\n" % similarities[ord(chosen_answer) - 65][0])

print("Correct: %d" % correct)
print("Total: %d" % total)
print("Accuracy: %.4f" % (float(correct) / total))
