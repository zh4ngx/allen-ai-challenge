"""
  Generate csv submission for Kaggle contest

  Straight up copy of lsa_evaluate with some lines altered
"""

import logging, gensim, bz2, sys

logging.basicConfig(
  format='%(asctime)s : %(levelname)s : %(message)s',
  level=logging.INFO
)

input_model = sys.argv[1]
input_validation = sys.argv[2]
output_file = sys.argv[3]

# Load model
model = gensim.models.LsiModel.load(input_model, mmap='r')

# Load validation set and advance 1 line
validation_set = open(input_validation)
validation_set.readline()

output = open(output_file, "w")
output.write("id,correctAnswer\n")

for line in validation_set:
  elements = line.split("\t")
  question_id = elements.pop(0)

  # Get bag-of-words representation of question and answers
  doc_vectors = [model.id2word.doc2bow(element.split()) for element in elements]
  question = doc_vectors.pop(0)

   # Generate list of tuples:
  # (Cosine similarity, mapped index 0-3 to A-D)
  similarities = [(gensim.matutils.cossim(model[question], model[answer]), chr(idx + 65)) for idx, answer in enumerate(doc_vectors)]
  chosen_answer = max(similarities)[1]

  output.write("%s,%s\n" % (question_id, chosen_answer))

output.close()
