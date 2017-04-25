import FileJoin
import FindInJson

result = FileJoin.createJoinedFile("resultierendebuchungssätze.csv")
FindInJson.findInJson(result)
