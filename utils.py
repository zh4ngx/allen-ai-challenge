import datetime
import string


# Convert index in set(0, 1, 2, 3) to answer in set('A', 'B', 'C', 'D')
def idx2answer_label(idx):
    return chr(idx + 65)


# preprocess text for querying model
def preprocess_for_model(model, text):
    words = text.lower().translate(None, string.punctuation).split()
    return [word for word in words if model.__contains__(word)]


# Choose answer based on highest score
def choose_answer(score_dict):
    return max(score_dict.iteritems(), key=lambda item: item[1])[0]


# extract elements from training or validation line
# returns question_id, question, answers (dict), correct_answer
def extract_elements(line, is_validation=False):
    elements = line.split("\t")
    question_id = elements.pop(0)
    correct_answer = elements.pop(1) if not is_validation else None

    question = elements.pop(0)
    answers = {idx2answer_label(idx): answer for idx, answer in enumerate(elements)}

    return question_id, question, answers, correct_answer


def generate_timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
