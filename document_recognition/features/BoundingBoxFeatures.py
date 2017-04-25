from pandas import DataFrame as df
import pandas as pd

vSearch = 30
hSearch = 380


class BoundingBoxFeatures:

    def getFeatureCount(self):
        return 6;

    def extractFeatures(self, dataframe: df):
        self.lastbox = None
        pages = dataframe.groupby('page')
        resultPages = []
        pageBoundingBoxes = []
        for pagetuple in pages:
            page = pagetuple[1].copy(True)
            boxesData = {"id": {}, "left": {}, "top": {}, "bottom": {}, "right": {}, "lineInBox": {}}

            boundingBoxes = []
            lastRow = None
            self.lastbox = None
            for row in page.iterrows():
                isNew = self.addToBoundingBoxes(boundingBoxes, row)
                if isNew:
                    boxesData["lineInBox"][row[0]] = 0
                else:
                    if row[1].left > lastRow[1].right:
                        boxesData["lineInBox"][row[0]] = boxesData["lineInBox"][lastRow[0]]
                    else:
                        boxesData["lineInBox"][row[0]] = boxesData["lineInBox"][lastRow[0]] + 1

                lastRow = row

            for box in boundingBoxes:
                for item in box["items"]:
                    boxesData["id"][item] = box["id"]
                    boxesData["left"][item] = box["left"]
                    boxesData["right"][item] = box["right"]
                    boxesData["top"][item] = box["top"]
                    boxesData["bottom"][item] = box["bottom"]

            page["boundLeft"] = df.from_dict(boxesData["left"], orient="index")
            page["boundRight"] = df.from_dict(boxesData["right"], orient="index")
            page["boundTop"] = df.from_dict(boxesData["top"], orient="index")
            page["boundBottom"] = df.from_dict(boxesData["bottom"], orient="index")
            page["lineInBox"] = df.from_dict(boxesData["lineInBox"], orient="index")
            page["VpercInBox"] = ((page.top + page.bottom) / 2 - page.boundTop) / (page.boundBottom - page.boundTop)

            pageBoundingBoxes.append(boundingBoxes)
            resultPages.append(page)

        result = pd.concat(resultPages)
        # result.to_csv("C:\\Stolze\\Scan Recognition\\GIT\\unstruct_labelling\\preprocess\\testoutput.json.csv", sep=" ", index=None)
        return result[["boundLeft", "boundRight", "boundTop", "boundBottom", "lineInBox", "VpercInBox"]]

    def addToBoundingBoxes(self, boxes, rowTuple):
        row = rowTuple[1]
        found = False

        if self.lastbox != None and self.lastbox["left"] < row.right + hSearch and self.lastbox["right"] + hSearch > row.left:
            if self.lastbox["top"] < row.bottom + vSearch and self.lastbox["bottom"] + vSearch > row.top:
                found = True
                self.lastbox["left"] = min(self.lastbox["left"], row.left)
                self.lastbox["top"] = min(self.lastbox["top"], row.top)
                self.lastbox["bottom"] = max(self.lastbox["bottom"], row.bottom)
                self.lastbox["right"] = max(self.lastbox["right"], row.right)
                self.lastbox["items"].append(rowTuple[0])
                return False

        for box in boxes:
            # box overlaps token
            if box["left"] < row.right + hSearch and box["right"] + hSearch > row.left:
                if box["top"] < row.bottom + vSearch and box["bottom"] + vSearch > row.top:
                    found = True
                    box["left"] = min(box["left"], row.left)
                    box["top"] = min(box["top"], row.top)
                    box["bottom"] = max(box["bottom"], row.bottom)
                    box["right"] = max(box["right"], row.right)
                    box["items"].append(rowTuple[0])
                    self.lastbox = box
                    return False
                    break

        if not found:
            theBox = {"id": len(boxes), "left": row.left, "right": row.right, "top": row.top, "bottom": row.bottom,
                      "items": [rowTuple[0]]}
            boxes.append(theBox)
            self.lastbox = theBox
        return True
