import pickle
import string

def convert():
    for char in string.uppercase:
        f = open("gomolo" + char + ".pickle", "r")
        movieCharList = pickle.loads(f.read())
        f.close()
        
        finalmovieCharList = []
        for movie in movieCharList:
            for attr in movie:
                    if not isinstance(movie[attr], list):
                        movie[attr] = [movie[attr]]

            tempMovie = {}
            for attr in movie:
                tempMovie[attr.lower().replace(' ', '_')] = movie[attr]

            finalmovieCharList.append(tempMovie)

        f = open("gomolo" + char + ".pickle", "w")
        f.write(pickle.dumps(finalmovieCharList))
        f.close()

def test():
    obj = pickle.loads(open("gomoloA.pickle", "r").read())
    for key in obj[0]:
        print key, obj[0][key]