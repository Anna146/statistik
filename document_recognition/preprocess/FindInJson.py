import Levenshtein
import json
import copy
import locale


def findIn(searchData, jsonData):
    jsonWordList = []
    for wordList in jsonData["result"]:
        currentList = []
        for words in wordList["words"]:
            words.append([])
            currentList.append(words)
        jsonWordList.append(currentList)

    for searchKey in searchData:
        searchWords = makeNumberWordList(searchData[searchKey])

        for searchWord in searchWords:
            searchWord = str.lower(searchWord)
            pageNr = 0
            threshold = len(searchWord) / 5
            for page in jsonWordList:
                wordNr = 0
                for word in page:
                    if not searchKey in word[5] and match(searchWord, word[0], threshold):
                        word[5].append(searchKey)
                    else:
                        continue
                    jsonData["result"][pageNr]["words"][wordNr] = word
                    wordNr += 1

                pageNr += 1

    return jsonData

def makeNumberWordList(word):

    try:
        word = float(word)
    except:
        None

    wordList = []
    if isinstance(word, float):
        wordList.append("{0:.2f}".format(word))
        if word == round(word):
            wordList.append(str(int(word)))
            wordList.append(str(int(word))+",-")
        locale.setlocale(locale.LC_ALL, 'de')
        wordList.append(locale.format("%.2f", word, grouping=True, monetary=True))
    else:
        splitt = str.split(word, " ")
        for sub in splitt:
            wordList.append(sub)
    return wordList

def match(word, word2, thresh):
    if not (abs(len(word) - len(word2)) <= thresh):
        return False

    wordList = makeNumberWordList(word)
    word2List = makeNumberWordList(word2)

    for entry in wordList:
        for entry2 in word2List:
            if Levenshtein.distance(str.lower(entry2), entry) < thresh:
                return True

    return False



def writeJSONFile(fileName, dictJSON):
    with open(fileName, "w") as outfile:
        json.dump(dictJSON, outfile, indent=4)


def findInJson(dataDictList):
    dictIndex = 0
    for dataDict in dataDictList:
        jsonFileName = dataDict["jsonfile"]
        with open(jsonFileName) as data_file:
            jsonData = json.load(data_file)
            newJsonData = findIn(dataDict, copy.deepcopy(jsonData))
            writeJSONFile("JSONResult" + str(dictIndex) + ".json", newJsonData)
        dictIndex += 1

