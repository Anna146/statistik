import csv
import fnmatch
import os
import codecs
import sys


def createJoinedFile(inputCSVfile):
    inputPath = ""
    inputFileName = inputCSVfile
    if inputCSVfile.rfind("/") > 0:
        inputPath = inputCSVfile[:inputCSVfile.rfind("/")] + "/"
        inputFileName = str.replace(inputCSVfile[inputCSVfile.rfind("/"):], "/", "")

    if not str.endswith(inputCSVfile, ".csv"):
        inputCSVfile = inputCSVfile + ".csv"
    fileNames = dict()
    orderKeysToUse = [
        "Vorgang",
        "Firma",
        #    "Zahlsperre",
        "Code",
        "Brutto",
        "R/G",
        #    "Währ.",
        "Ver-Kdnr",
        "Provision",
        "Netto",
        #    "DI",
        "Reisender",
        "Reisedatum",
        "Belegdatum",
        "Rechnungsnr.",
        "Veranstalter",
        "Büro",
        #    "Zahlkreditor",
        #    "SAP-Beleg-Nr",
        #    "Kostenstelle",
        #    "Erlöskonto",
        #    "MWSTSchlüssel",
        "MWSTProzent",
        "Fällig am",
        "BU-NAME",
        "BU-NAME-2",
        "BU-NAME-VERSAND",
        "BU-ORT",
        "BU-STRASSE",
        "BU-POSTFACH",
        "BU-PLZ",
        "BU-PLZ-POSTF",
        "BU-PLZ-GROSS",
        "BU-NAT-KENNZ",
        "BU-GUELTIG-VON",
        "BU-GUELTIG-BI",
        "BU-KOM-TEL-1",
        "BU-KOM-TEL-2",
        "BU-KOM-FAX",
        "BU-KOM-BTX",
        "BU-KASSENKONTO",
        "BU-EMPFAENGER"
    ]

    debitorKeysToUse = [
        "BU-NAME",
        "BU-NAME-2",
        "BU-NAME-VERSAND",
        "BU-ORT",
        "BU-STRASSE",
        "BU-POSTFACH",
        "BU-PLZ",
        "BU-PLZ-POSTF",
        "BU-PLZ-GROSS",
        "BU-NAT-KENNZ",
        "BU-GUELTIG-VON",
        "BU-GUELTIG-BI",
        "BU-KOM-TEL-1",
        "BU-KOM-TEL-2",
        "BU-KOM-FAX",
        "BU-KOM-BTX",
        "BU-KASSENKONTO",
        "BU-EMPFAENGER",
        "BU-KONTO",
        "BU-E-MAIL",
        "BU-VERKAUFSEINHEIT",
        "BU-DEBITOREN-ZUORDNUNG"
    ]

    kreditorKeysToUse = [
        "NAME",
        "NAT-KENNZ",
        "PLZ-NEU",
        "PLZ-POSTF",
        "PLZ-GROSS",
        "ORT",
        "STRASSE",
        "POSTFACH",
        "KOM-TEL-1",
        "KOM-TEL-2",
        "KOM-TELEX",
        "KOM-FAX",
        "KOM-BTX"
    ]

    for file in os.listdir(inputPath + '.'):
        if fnmatch.fnmatch(file, 'R001*'):
            fileNames["debitor"] = inputPath + file
        if fnmatch.fnmatch(file, 'R003*'):
            fileNames["kreditor"] = inputPath + file
        if fnmatch.fnmatch(file, inputFileName):
            fileNames["order"] = inputPath + file


    # read files
    debitorDict = readFileIndexed(fileNames["debitor"], "BU-KDNR")
    kreditorDict = readFileIndexed(fileNames["kreditor"], "VERANSTALTER")
    orderList = readFileToRecords(fileNames["order"])

    finalResult = []
    # join
    for record in orderList:
        firmaId = record["Firma"]
        veranstalter = record["Code"]
        resultingRecord = dict()

        for key in record:
            if key in orderKeysToUse:
                resultingRecord[key] = record[key]

        if firmaId in debitorDict:
            debitorRow = debitorDict[firmaId]
        else:
            sys.stderr.write("no debitor found for BU-KDNR " + firmaId + " SAP-Belegnr = " + record["SAP-Beleg-Nr"] + "\n")
            continue

        if veranstalter in kreditorDict:
            kreditorRow = kreditorDict[veranstalter]
        else:
            sys.stderr.write("no kreditor found for veranstalter " + veranstalter + " SAP-Belegnr = " + record["SAP-Beleg-Nr"]  + "\n")
            continue

        for key in debitorKeysToUse:
            resultingRecord[key] = debitorRow[key]
        for key in kreditorKeysToUse:
            resultingRecord[key] = kreditorRow[key]

        tax = float(resultingRecord["MWSTProzent"]) / 10000.0
        provisionNet = str.replace(resultingRecord["Provision"], ",", ".")
        resultingRecord["provNetto"] = provisionNet
        resultingRecord["provBrutto"] = float(provisionNet) / (1 + tax)
        resultingRecord["provSteuer"] = (float(provisionNet) / (1 + tax)) * tax
        resultingRecord["MWST"] = int(float(resultingRecord["MWSTProzent"]) / 100.0)

        resultingRecord["jsonfile"] = str.replace(inputCSVfile, ".csv", "") + "_json/" + record["Barcode"] + ".json"

        finalResult.append(resultingRecord)

    return finalResult


def readFileIndexed(fileName, indexColumn=None):
    with codecs.open(fileName, "rU", encoding="UTF-8") as csvfile:
        dictReader = csv.DictReader(csvfile, delimiter=";")
        result = dict()
        for dictionary in dictReader:
            result[dictionary[indexColumn]] = dictionary
        return result


def readFileToRecords(fileName):
    with codecs.open(fileName, "rU", encoding="UTF-8") as csvfile:
        dictReader = csv.DictReader(csvfile, delimiter=";")
        result = []
        for dictionary in dictReader:
            result.append(dictionary)
        return result


if __name__ == '__main__':
    result = createJoinedFile(sys.argv[1])
    writer = csv.DictWriter(sys.stdout, result[0].keys(), delimiter=',', lineterminator='\n')
    writer.writeheader()
    for row in result:
        writer.writerow(row)
    sys.exit(0)
