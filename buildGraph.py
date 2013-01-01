#!/usr/bin/python
# Program to build a graph from the movie dataset
# Time: 08.55 AM
# Date: 22/08/2012
# Todo:
#

# Python standard libraries
import pickle
import string

# Third party libraries
import networkx as nx


def readPickle(fileName):
    """
        readPickle(fileName)
            input: takes a file name
            returns: a list of dictionaries
    """
    inputFile = open(fileName, "r")
    result = pickle.loads(inputFile.read())
    inputFile.close()
    return result

def main():
    # Reading the movie database
    movieList = []
    for char in string.uppercase:
        movieList.extend(readPickle("gomolo" + char + ".pickle"))

    attributeAndNodes = {}

    # Building the graph
    G = nx.Graph()
    for movie in movieList:

        commonProps = {}

        print G.number_of_nodes(), 
        movieName = movie["name"][0]
        movie.pop("name")
        movie.pop("img_data")
        movie.pop("img_url")
        G.add_node(movieName)

        for attrs in movie:
            G[movieName][attrs] = movie[attrs]

            if attributeAndNodes.has_key(attrs):
                for item in movie[attrs]:
                    if attributeAndNodes[attrs].has_key(item):
                        attributeAndNodes[attrs][item].append(movieName)
                    else:
                        attributeAndNodes[attrs][item] = [movieName]
            else:
                attributeAndNodes[attrs] = {}
                for item in movie[attrs]:
                    attributeAndNodes[attrs][item] = [movieName]

            for item in movie[attrs]:
                movieConnections = attributeAndNodes[attrs][item]

                for movieNeighbor in movieConnections:
                    commonProps[movieNeighbor] = {}
                    if commonProps[movieNeighbor].has_key(attrs):
                        commonProps[movieNeighbor][attrs].append(item)
                    else:
                        commonProps[movieNeighbor][attrs] = [item]

            for key in commonProps:
                G.add_edge(key, movieName)
                G[key][movieName] = commonProps[key]


        for prop in attributeAndNodes:
            #print prop, attributeAndNodes[prop]
            pass
        #raw_input("press any key to continue..")



        """
        nodeList = G.nodes()
        print len(nodeList), 
        movieName = movie["name"][0]
        movie.pop("name")  # Removing name from the dictionary
        G.add_node(movieName)
        for key in movie.keys():
            G[movieName][key] = movie[key]

        for node in nodeList:
            commonAttrs = {}
            movieAttrs = G[movieName].keys()
            movieAttrs.remove('img_data')
            movieAttrs.remove('img_url')
            #print movieAttrs
            for attr in movieAttrs:
                if G[movieName].has_key(attr) and G[node].has_key(attr):
                    intscn = list(set(G[movieName][attr]).intersection(set(G[node][attr])))
                
                    if intscn:
                        commonAttrs[attr] = intscn

            if commonAttrs:
                G.add_edge(movieName, node)
                G[movieName][node]["common"] = commonAttrs
        """

    f = open("builtGraph.pickle", "w")
    GStringData = pickle.dumps(G)
    f.write(GStringData)
    f.close()

if __name__ == "__main__":
    main()