import codecs, csv, time

threshold = 2

CORE_DATA = "core"
# CORE DATA
WORD = "text"
LEFT = "left"
RIGHT = "right"
TOP = "top"
BOTTOM = "bottom"
PAGE = "page"

# CALCULATED DATA
H_AXIS = "H_Axis"
V_AXIS = "V_Axis"
RIGHT_OF = "right_of"


def map_axes(wordListWithPositions):
    pages = dict()

    floats = [LEFT, RIGHT, TOP, BOTTOM]
    for item in wordListWithPositions:
        page = item[PAGE]
        word = dict()
        word[CORE_DATA] = item
        for key in word[CORE_DATA]:
            try:
                word[CORE_DATA][key] = float(word[CORE_DATA][key])
            except:
                None
        word[H_AXIS] = []
        word[V_AXIS] = []
        if page not in pages:
            pages[page] = []
        pages[page].append(word)
    for page in pages:
        wordList = pages[page]
        for i in range(len(wordList)):
            data1 = wordList[i]
            for k in range(i + 1, len(wordList)):
                data2 = wordList[k]

                # same Horizontal Axis
                if data1[CORE_DATA][TOP] < data2[CORE_DATA][BOTTOM] and \
                                data1[CORE_DATA][BOTTOM] > data2[CORE_DATA][TOP]:
                    # data2 left of data1
                    if data1[CORE_DATA][LEFT] > data2[CORE_DATA][LEFT]:
                        data1[H_AXIS].append(data2[CORE_DATA])
                        if RIGHT_OF not in data2 or data2[RIGHT_OF][LEFT] > data1[CORE_DATA][LEFT]:
                            data2[RIGHT_OF] = data1[CORE_DATA]
                    else:
                        data2[H_AXIS].append(data1[CORE_DATA])
                        if RIGHT_OF not in data1 or data1[RIGHT_OF][LEFT] > data2[CORE_DATA][LEFT]:
                            data1[RIGHT_OF] = data2[CORE_DATA]

                # same Vertical Axis

                if abs(data1[CORE_DATA][LEFT] - data2[CORE_DATA][LEFT]) <= threshold or abs(
                                data1[CORE_DATA][RIGHT] - data2[CORE_DATA][RIGHT]) <= threshold:
                    # data1 below data2
                    if data1[CORE_DATA][TOP] < data2[CORE_DATA][TOP]:
                        data1[V_AXIS].append(data2[CORE_DATA])
                    else:
                        data2[V_AXIS].append(data1[CORE_DATA])
    return wordList


def readFileToRecords(fileName):
    with codecs.open(fileName, "rU", encoding="UTF-8") as csvfile:
        dictReader = csv.DictReader(csvfile, delimiter=" ")
        result = []
        for dictionary in dictReader:
            result.append(dictionary)
        return result

start = time.time()
csvFile = readFileToRecords("07010000291632.csv")

map_axes(csvFile)
print(time.time()-start)
