from __future__ import division
import json


eps = 10
json_dir = 'jsons3'
itr = 0

field_names = ['aftertax', 'commission', 'tripdate', 'traveller', 'voucherdate', 'duedate', 'pretax', 'pstreet', 'pzip', 'pcity', 'cname', 'cstreet', 'czip', 'currency', 'ccity', 'vatpercent']
sh = open('reference_dict.txt', 'r')
ref_dic = {int(x.split(',')[2].strip()):int(x.split(',')[0]) for x in sh}


def word_match(reals, preds):
    tp = len([i for i in range(len(reals)) if reals[i] == preds[i]])
    return tp, len(reals)-tp

def area(a, b):  # returns None if rectangles don't intersect
    dx = min(a['x2'], b[3]) - max(a['x1'], b[1])
    dy = min(a['y2'], b[4]) - max(a['y1'], b[2])
    if (dx>=0) and (dy>=0):
        return dx*dy
    else:
        return 0

def dist(c1, c2):
    return (abs(c1[0]-c2[0])**2 + abs(c1[1]-c2[1])**2)**0.5

def try_json(j_path, coords, page, conf_thr):
    iou_thr = 0.01
    try:
        true_doc = json.load(open(j_path, 'r'))
    except:
        return None, None
    words = [x for x in true_doc['result'] if x['page'] == page+1][0]['words']
    #if json is faulty
    if len(coords) == 0:
        return None, None
    result = [-1 for x in words]
    #precomputations
    for rec1 in coords:
        rec1['mid'] = ((rec1['y2'] - rec1['y1']) / 2 + rec1['y1'],(rec1['x2'] - rec1['x1']) / 2 + rec1['x1'])
    for rec2 in words:
        rec2 += [((rec2[4] - rec2[2]) / 2 + rec2[2],(rec2[3] - rec2[1]) / 2 + rec2[1])]
    for i in range(len(words)):
        rec2 = words[i]
        min_dist = 100500
        min_dist_lab = None
        for rec1 in coords:
            #too low a score
            if rec1['score'] < conf_thr:
                continue
            di = dist(rec1['mid'], rec2[-1])
            if di < min_dist:
                min_dist = di
                min_dist_lab = rec1['label']
            #first check against iou threshold
            if area(rec1,rec2) / abs(rec2[3]-rec2[1]) / abs(rec2[4]-rec2[2]) > iou_thr:
                result[i] = rec1['label']
        '''
        if result[i] == -1 and min_dist_lab != None:
            result[i] = min_dist_lab
        '''
    return j_path, [ref_dic[x] for x in result]



big_dict = dict()
doc_lst = []
files_dic = dict()

pred = json.load(open('pred.json', 'r'))
doc_lst = list(set(x['image_path'].split('_')[0].split('/')[-1] for x in pred))
files_dic = dict((str(x),[z for z in pred if z['image_path'].find(str(x)) != -1]) for x in doc_lst)
if len(big_dict) == 0:
    big_dict = dict((x,['' for x in range(len(field_names))]) for x in doc_lst)
    print(len(big_dict))
f = open('filelist.txt', 'w')
f_out = open('my_pred.json', 'w')
big = []
cnt = 0
for document_numb in doc_lst:
    doc_res = []
    for pr in files_dic[document_numb]:
        json_path = json_dir + '/' + document_numb + ".json"
        good_j, res = try_json(json_path, pr['rects'],int(pr['image_path'].split('_')[2])-1, 0.1)
        if res != None:
            doc_res += res
    if doc_res != []:
        print(document_numb)
        f.write(document_numb + '\n')
        big += [doc_res]
        cnt += 1
json.dump(big, f_out)
print(cnt)

