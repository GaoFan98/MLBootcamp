import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# NLTK SHELL SYSTEM
# nltk.download_shell()


# # MESSAGES DATA
# messages = [line.rstrip() for line in open('SMSSpamCollection')]
# # GETTING FIRST 10 MESSAGES
# for mess_no, message in enumerate(messages[:10]):
#     print(mess_no, message)
#     print('\n')

messages = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
# SOME INFO ABOUT MESSAGE LABELS COUNT ETC...
messages.groupby('label').describe()

# GET LENGTH OF MESSAGES
messages['length'] = messages['message'].apply(len)
# VISUALIZE LENGTH OF MESSAGES
messages['length'].plot.hist(bins=50)

# HAM AND SPAM VISUALIZATION OF LENGTH COLUMN
messages.hist(column='length', by='label', bins=60, figsize=(12, 4))


# plt.show()


# REMOVING STOPWORDS FROM MESSAGES
def text_process(mess):
    # REMOVING PUNCTUATION FROM STRING
    nopunc = [char for char in mess if char not in string.punctuation]
    # RETURNING BACK STRING WITHOUT PUNCTUATION
    nopunc = ''.join(nopunc)
    # REMOVING ALL STOPWORDS FROM STRING AND RETURN BACK CLEAN STRING
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# GETTING MESSAGES AND CLEANING STRING LIKE DID BELOW
# print(messages['message'].head(5).apply(text_process))

# VECTORIZATION, CREATING MATRIX OF WORD APPERANCE IN MESSAGES
bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])
# EXAMPLE
# GETTING 4th MESSAGE
mess4 = messages['message'][3]
# GETTING HOW MANY TIMES SPECIFIC WORDS SHOW UP IN MESSAGE
bow4 = bow_transformer.transform([mess4])

# DOING THE SAME WITH ENTIRE DATA
messages_bow = bow_transformer.transform(messages['message'])

# TFIDF is a numerical statistic that is intended to reflect how
# important a word is to a document in a collection or corpus.
tfidf_transformer = TfidfTransformer().fit(messages_bow)
# EXAMPLE
tfidf_4 = tfidf_transformer.transform(bow4)
# GETTING HOW IMPORTANT 'UNIVERSITY' WORD
uni_importance = tfidf_transformer.idf_[bow_transformer.vocabulary_['university']]

# APPLY TFIDF TO ENTIRE DATA
messages_tfidf = tfidf_transformer.transform(messages_bow)

# DETECTING SPAM MESSAGES
# CREATING MODEL
spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])
# GRABBING 4th MESSAGE AND CHECK IF PREDICTIONS ARE CORRECT
sp_m_4 = spam_detect_model.predict(tfidf_4)[0]
# DOING THE SAME WITH ENTIRE DATA
all_pred = spam_detect_model.predict(messages_tfidf)

# SPLIT AND TRAIN MODEL
# WE CAN DO EVERYTHING MUCH SHORTER
msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.3,
                                                                random_state=101)

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])

# FIT MODEL ON TRAINING DATA
pipeline.fit(msg_train, label_train)
# PREDICT MODEL
predictions = pipeline.predict(msg_test)
# CLASSIFICATION FINAL REPORT
print(classification_report(label_test, predictions))
print(confusion_matrix(label_test, predictions))
