import pandas as pd
import math
from pandas import DataFrame as df


def alignPages(dataframe: df):
    pages = dataframe.groupby('page')
    resultPages = []
    for page in pages:
        firstRow = page[1].iloc[0]
        pageBounds = {"left": firstRow.left, "top": firstRow.top, "right": firstRow.right, "bottom": firstRow.bottom}

        lastCenter = [(firstRow.left + firstRow.right) / 2, (firstRow.bottom + firstRow.top) / 2]
        slopeValue = 0
        count = 0

        for index, row in page[1].iloc[1:].iterrows():
            currentCenter = [(row.left + row.right) / 2, (row.bottom + row.top) / 2]

            pageBounds["left"] = min(pageBounds["left"], row.left)
            pageBounds["top"] = min(pageBounds["top"], row.top)
            pageBounds["right"] = max(pageBounds["right"], row.right)
            pageBounds["bottom"] = max(pageBounds["bottom"], row.bottom)

            if currentCenter[0] > lastCenter[0] and abs(currentCenter[1] - lastCenter[1]) < 10:
                xDiff = currentCenter[0] - lastCenter[0]
                yDiff = currentCenter[1] - lastCenter[1]
                #anstieg
                slopeValue += yDiff / xDiff
                count += 1
            lastCenter = currentCenter

        #averaging
        if count!=0:
            slopeValue /= count
        else:
            slopeValue = 0

        angle = -math.atan2(slopeValue, 1)
        pageCenter = {"x": (pageBounds["right"] + pageBounds["left"]) / 2,
                      "y": (pageBounds["top"] + pageBounds["bottom"]) / 2}

        newPage = page[1].copy(True)


        cx = (newPage.left + newPage.right) / 2
        cy = (newPage.top + newPage.bottom) / 2
        offX = pageCenter["x"] + (cx - pageCenter["x"]) * math.cos(angle) - (cy - pageCenter["y"]) * math.sin(angle) - cx
        offY = pageCenter["y"] + (cx - pageCenter["x"]) * math.sin(angle) + (cy - pageCenter["y"]) * math.cos(angle) - cy


        newPage.left = newPage.left + offX
        newPage.right = newPage.right + offX
        newPage.top = newPage.top + offY
        newPage.bottom = newPage.bottom + offY

        resultPages.append(newPage)

    result = pd.concat(resultPages)
    # result.to_csv("C:\\Stolze\\Scan Recognition\\GIT\\unstruct_labelling\\preprocess\\testoutput.json.csv", sep=" ", index=None)
    return result
