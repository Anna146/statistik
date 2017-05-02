import pickle
import json

predicted = pickle.load(open('real.pkl', 'rb'))
print(predicted[1])
db = pickle.load(open('./databuilder.pkl', 'rb'))
score_labels = lambda x : ['{}_{}'.format(x, l) for l in db.labels]
print(db.labels)

testfiles = [f.split('/')[-1][2:] for f in pickle.load(open('./test_files.pkl','rb'))]
print(testfiles)
f = open('validation_baseline.json', 'w')
inp = []
cnt = 0
for name in testfiles:
    if name[1] != '7':
        try:
            js = json.load(open('jsons3/00'+name+'.json', 'r'))
        except:
            continue
            print('crap')
        cnt += 1
        for i in range(len(js['result'])):
            inp += [{'image_path':'imgs3/00' + name + '_Seite_'+ str(i+1) +'_Bild_0001.tif', 'rects':[{'x1':0,'x2':1,'y1':0,'y2':1, 'label':1}]}]
json.dump(inp,f)
print(cnt)

