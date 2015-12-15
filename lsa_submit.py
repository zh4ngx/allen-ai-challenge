"""
  Generate csv submission for Kaggle contest

  Straight up copy of lsa_evaluate with some lines altered
"""
import argparse
import logging

from gensim.matutils import cossim
from gensim.models import LsiModel

from utils import generate_timestamp

logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO
)

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="path to word2vec/model/timestamp.model")
parser.add_argument("-p", "--project", help="path to validation_set.tsv and submissions")
args = parser.parse_args()

# Load model
model = LsiModel.load(args.model, mmap='r')

# Load validation set and advance 1 line
validation_set = open("%s/validation_set.tsv" % args.project)
validation_set.readline()

output = open("%s/%s_submission.csv" % (args.project, generate_timestamp()), "w")
output.write("id,correctAnswer\n")

for line in validation_set:
    elements = line.split("\t")
    question_id = elements.pop(0)

    # Get bag-of-words representation of question and answers
    doc_vectors = [model.id2word.doc2bow(element.split()) for element in elements]
    question = doc_vectors.pop(0)

    # Generate list of tuples:
    # (Cosine similarity, mapped index 0-3 to A-D)
    similarities = [(cossim(model[question], model[answer]), chr(idx + 65)) for idx, answer in
                    enumerate(doc_vectors)]
    chosen_answer = max(similarities)[1]

    output.write("%s,%s\n" % (question_id, chosen_answer))

output.close()
