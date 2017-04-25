import editdistance
import locale

def calcThreshold(token):
    return len(token) / 5

def annotateDocument(doc,searchData):
    expected = {key: expectedTokens(search) for key,search in searchData.items()}
    return [
        {'token':word[0],
         'left':word[1],
         'top':word[2],
         'right': word[3],
         'bottom': word[4],
         'page': page['page'],
         'labels': findMatchingLabels(word[0],expected)}
        for page in doc["result"]
        for word in page['words']
    ]

def findMatchingLabels(docToken, findableDict):
    token = docToken.lower()
    return [key for key, wanted in findableDict.items() if closeMatch(token,wanted)]

def closeMatch(docToken, searchedTokens):
    tokenLen = len(docToken)
    for expected in searchedTokens:
        threshold = calcThreshold(expected)
        if abs(tokenLen - len(expected)) <= threshold and editdistance.eval(docToken,expected) <= threshold:
            return True
    return False

def expectedTokens(completeValue):
    try:
        completeValue = float(completeValue)
    except:
        None

    if isinstance(completeValue, float):
        tokenList = []
        tokenList.append("{0:.2f}".format(completeValue))
        if completeValue == round(completeValue):
            tokenList.append(str(int(completeValue)))
            tokenList.append(str(int(completeValue))+",-")
        locale.setlocale(locale.LC_ALL, 'de')
        tokenList.append(locale.format("%.2f", completeValue, grouping=True, monetary=True))
        return tokenList
    else:
        return list(map(str.lower,str.split(completeValue, " ")))

annotateDocument({'result':[{'page':1,'words':[['1,00',1,2,3,4],['sdf',1,2,3,4]]}]},{'wurst':'1'})
