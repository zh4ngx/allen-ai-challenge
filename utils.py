import string

# Convert index in set(0, 1, 2, 3) to answer in set('A', 'B', 'C', 'D')
def idx2answerlabel(idx):
    return chr(idx + 65)

# Preprocess text for querying gensim model
def preprocess_text_gensim(model, text):
    words = text.lower().translate(None, string.punctuation).split()
    return [word for word in words if model.__contains__(word)]

# Choose answer based on highest score
def choose_answer(score_dict):
    return max(score_dict.iteritems(), key=lambda item: item[1])[0]
