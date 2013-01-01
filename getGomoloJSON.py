import json
import string
import pickle

listOfJSON = []
for char in string.uppercase:
        f = open("gomolo" + char + ".pickle", "r")
        movieCharList = pickle.loads(f.read())
        f.close()

        for movie in movieCharList:
            movie.pop('img_data')
            listOfJSON.append(json.dumps(movie))

f = open("gomoloJSONtext", "w")
for movie in listOfJSON:
    f.write(movie + '\n')
f.close()
