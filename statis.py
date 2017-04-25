import pickle

predicted = pickle.load(open('real.pkl', 'rb'))
db = pickle.load(open('./databuilder.pkl', 'rb'))
score_labels = lambda x : ['{}_{}'.format(x, l) for l in db.labels]
print(db.labels)