"""
  Evaluate efficacy of wiki LSA on Allen AI dataset

  Use latent vectors to calculate cosine distance between Questions/Answers
  Pick the answer with the smallest cosine distance
  Check against correct answer
  Measure accuracy by simple
"""

import logging, gensim, bz2, sys

logging.basicConfig(
  format='%(asctime)s : %(levelname)s : %(message)s',
  level=logging.INFO
)

input_model = sys.argv[1]
input_training = sys.argv[2]

# Load model
# Note - model contains dictionary that intentionally omits stopwords
model = gensim.models.LsiModel.load(input_model, mmap='r')

# Load 'training' data
training_data = open(input_training)
training_data.readline() # advance past header line

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
  similarities = [(gensim.matutils.cossim(question, answer), chr(idx + 65)) for idx, answer in enumerate(doc_vectors)]
  chosen_answer = max(similarities)[1]

  correct += correct_answer == chosen_answer
  total += 1

print("Correct: %d" % correct)
print("Total: %d" % total)
print("Accuracy: %.2f" % (float(correct) / total))

# python ~/code/kaggle/the-allen-ai-science-challenge/lsa_evaluate.py /home/andy/data/allen-ai/lsa/lsa_model_1 /home/andy/data/allen-ai/training_set.tsv
# 2015-11-30 23:10:29,726 : INFO : loading LsiModel object from /home/andy/data/allen-ai/lsa/lsa_model_1
# 2015-11-30 23:10:29,776 : INFO : loading id2word recursively from /home/andy/data/allen-ai/lsa/lsa_model_1.id2word.* with mmap=r
# 2015-11-30 23:10:29,777 : INFO : setting ignored attribute projection to None
# 2015-11-30 23:10:29,777 : INFO : setting ignored attribute dispatcher to None
# 2015-11-30 23:10:29,777 : INFO : loading LsiModel object from /home/andy/data/allen-ai/lsa/lsa_model_1.projection
# 2015-11-30 23:10:29,777 : INFO : loading u from /home/andy/data/allen-ai/lsa/lsa_model_1.projection.u.npy with mmap=r
# Correct: 608
# Total: 2500
# Accuracy: 0.24
