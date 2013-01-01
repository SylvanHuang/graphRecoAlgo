#!/usr/bin/python
# Program to build a generic recommender
# Time: 
# Date: 
# Todo:

# Python standard libraries
import pickle
import string
import random
import json
import copy
import matplotlib.pyplot as plt
import pylab as pl
import time
import re
import sys

# Third party libraries
import networkx as nx

def getRealNeighbors(noisyList):
    '''
    noisyList (list)
    G.neighbors(node) function will return a list that contains neighbors and attributes of a node. What we need is the actual set of neighbors. Hence we're removing the elements that are not integers.
    '''
    tempList = []
    for choice in noisyList:
        if isinstance(choice, int) or type(choice) == int: #checking if the element is an integer
            tempList.append(choice)

    return tempList

def learnGraph(JSONdb):
    '''
    JSONdb (string) - file name of the db - a file of strings, each of which are in JSON format
    Given a database of items, this function generates the item relations graph that can be used to for recommending items to users in a content based manner.
    '''

    #open the file, read each line, parse it and put it onto the itemList
    itemList = []
    fp = open(JSONdb, "r")
    for line in fp:
        itemList.append(json.loads(line))
    fp.close()

    attributeAndNodes = {}  #{attribute1: {value1 : [item1, item2 ...], value2 : [item3, item4 ... ]}, attribute2 : ... }

    #Building the graph
    G = nx.Graph()
    uid = 0

    for item in itemList:
        commonProps = {} #{node1 : {attr1 : value1, attr2 : value2 ... }, node2 ... } represents the edge between given node and node1, with the edge attribute being {attr1 : value1, attr2 : value2 ...}

        G.add_node(uid) #each node is recognized by its unique ID, an integer.

        for attrs in item:
            #set the attribute of the item as the attribute of the node
            if attrs != "playback_singer" and attrs != "year":

                G[uid][attrs] = item[attrs]

                #check if the node already has the attribute.
                #If it does, for every attribute of the item, check if there is a list associated for the value. If the list exists, append the uid to the list. If it doesnt, initialize a list with the uid of the item.
                #If it doesnt, initialize the attributeAndNodes[attrs] to an empty dictionary. For every attribute of the item, update the attributeAndNodes dictionary.
                if attributeAndNodes.has_key(attrs):
                    for attribute in item[attrs]:
                        if attributeAndNodes[attrs].has_key(attribute):
                            attributeAndNodes[attrs][attribute].append(uid)
                        else:
                            attributeAndNodes[attrs][attribute] = [uid]
                else:
                    attributeAndNodes[attrs] = {}
                    for attribute in item[attrs]:
                        attributeAndNodes[attrs][attribute] = [uid]

                #setting up links between items
                #for each attribute of the item, find out which nodes have the same attribute and value. These nodes become the item's neighbors and we put an edge between them.
                for attribute in item[attrs]:
                    itemConnections = attributeAndNodes[attrs][attribute]

                    for itemNeighbor in itemConnections:
                        if itemNeighbor != uid:
                            commonProps[itemNeighbor] = {}
                            if commonProps[itemNeighbor].has_key(attrs):
                                commonProps[itemNeighbor][attrs].append(attribute)
                            else:
                                commonProps[itemNeighbor][attrs] = [attribute]

                #adding edges from current node to other nodes.
                for node in commonProps:
                    #print uid, node, commonProps[node]
                    #raw_input()
                    G.add_edge(node, uid)
                    G[node][uid] = commonProps[node]

        uid += 1

    #write the object onto a pickle file
    fp = open(JSONdb + "_GraphDB.pickle", "w")
    fp.write(pickle.dumps(G))
    fp.close()

def learnPowerMethod(USERdb):
    '''
    USERdb (string) - file name of the db - a file of strings, each of which are in JSON format
    Given a database and user-item associations, this function builds a collaborative data structure that can be used in collaborative filtering.

    '''
    fp = open("_powerMethod.pickle", "w")
    #fp.write(pickle.dumps(_powerMethod))
    fp.close()

def egocentricRecommendation(graphDb, userSequence, dbFileName, uid):
    '''
    graphDb (networkx Graph object): The database containing the relation between items in a graphical form
    userSequence (list) : the sequence in which the user has been associated with items
    This function applies our content based filtering algorithm to generate a score ranging from 0-1 for every item. This object will be written to contentReco.pickle as a pickle object. This pickle object is a dictionary with uid and score as the key and value respectively.
    '''

    #set the initial score for all the nodes to be zero.
    for node in graphDb.nodes():
        graphDb[node]["score"] = 0

    #for every node in the userSequence, increment the neighbor's score for every common attribute. (The increment can also take place in a weight manner)
    sequenceNames = []
    for node, rating in userSequence:
        sequenceNames.append(graphDb[node]["name"][0])
        neighbors = getRealNeighbors(graphDb.neighbors(node))

        for neighbor in neighbors:
            for assoc in graphDb[node][neighbor]:
                graphDb[neighbor]["score"] += 1 #Note: This is not weighted increment.
    #print "user sequence: ", sequenceNames

    #normalizing the scores - divide by the largest score
    weights = [graphDb[node]["score"] for node in graphDb.nodes()]
    maxWeight = float(max(weights))

    for node in graphDb.nodes():
        graphDb[node]["score"] = graphDb[node]["score"] / maxWeight

    #create a dictionary with key and value as node and score respectively
    reco = {}
    for node in graphDb.nodes():
        reco[node] = graphDb[node]["score"]

    #write this object onto a pickle file
    fp = open(dbFileName + "_" + str(uid) + "_contentReco.pickle", "w")
    fp.write(pickle.dumps(reco))
    fp.close()

def buildUserSimilarityDict(G, userSequence, userProfiles, dbFileName, upperlim):
    '''
    G (networkx object): The Graph of items
    userSequence (dictionary) : dictionary of users, with their movie watching sequences
    userProfiles (dictionary) : dictionary of users, with their alpha and relative weights for various attributes
    This function builds a user similarity matrix. The matrix is being simulated using a dictionary. Each entry in the matrix is a tuple, a numerator and a denominator. The ratio gives the similarity.
    '''
    userSimilarity = {}

    users = userSequence.keys()
    users.sort()

    users = users[:upperlim]

    for i in range(len(users)):
        userSimilarity[i] = {}
        useriItems = set(userSequence[i])

        for j in range(i, len(users)):
            userjItems = set(userSequence[j])

            #get the intersecting items from both the users and find the induced subgraph
            intersectingItems = [item for item, rating in list(useriItems.intersection(userjItems))]
            intersectingSubgraph = getSubgraph(G, intersectingItems)

            #find all the attributes of all the edges
            attribs = []
            for edge in getEdges(intersectingSubgraph):
                attribs.extend(intersectingSubgraph[edge[0]][edge[1]].keys())

            #to determine numerator, iterate through all the attributes and add up the relative weights of individual users.
            numerator = 0.0
            for attrib in set(attribs):
                useri = 0.0
                if userProfiles[i]['weights'].has_key(attrib):
                    useri = userProfiles[i]['weights'][attrib]

                userj = 0.0
                if userProfiles[j]['weights'].has_key(attrib):
                    userj = userProfiles[j]['weights'][attrib]
                
                numerator += (useri + userj)

            #get the union of items from both the users and find the induced subgraph
            unionItems = [ item for item, rating in list(useriItems.union(userjItems))]
            unionSubgraph = getSubgraph(G, unionItems)

            #find all the attributes of all the edges
            attribs = []
            for edge in getEdges(unionSubgraph):
                attribs.extend(unionSubgraph[edge[0]][edge[1]].keys())

            #to determine denominator, iterate through all the attributes and add up the relative weights of individual users.
            denominator = 0.0
            for attrib in set(attribs):
                useri = 0.0
                if userProfiles[i]['weights'].has_key(attrib):
                    useri = userProfiles[i]['weights'][attrib]

                userj = 0.0
                if userProfiles[j]['weights'].has_key(attrib):
                    userj = userProfiles[j]['weights'][attrib]

                denominator += (useri + userj)

            userSimilarity[i][j] = [numerator, denominator]

            print i, j, numerator, denominator

    f = open(dbFileName + "_userSimilarity.pickle", "w")
    f.write(pickle.dumps(userSimilarity))
    f.close()

def getSimilarity(userSimilarity, user1, user2):
    if userSimilarity[user1].has_key(user2):
        return userSimilarity[user1][user2][0] / userSimilarity[user1][user2][1]
    else:
        return userSimilarity[user2][user1][0] / userSimilarity[user2][user1][1]

def formClusters(userSimilarity, threshold, dbFileName):
    #clusters = []
    user = userSimilarity.keys()
    user.sort(reverse=True)
    clusters = [[i] for i in user]
    threshold = .3
    for x in user:
        for grp in range(len(clusters)):
            if x not in clusters[grp] and clusters[grp]!=[None]:
                flag = "asd"
                for a in clusters[grp]:
                      if getSimilarity(userSimilarity, a, x) < threshold :
                        flag = False
                        break
                if flag != False :
                    clusters[grp].append(x)
                    if [x] in clusters:
                        clusters[clusters.index([x])]=[None]
    clusters  = [ set(cluster) for cluster in clusters]
    clusters = [list(cluster) for cluster in clusters if clusters.count(cluster) == 1]
    
    f = open(dbFileName + "_clusters.pickle", "w")
    f.write(pickle.dumps(clusters))
    f.close()








    '''
    users = userSimilarity.keys()

    singletons = []

    clusters = []
    for user in users:
        clusters.append(set([user]))

    print clusters

    for user in users:
        singletonFlag = True
        for i in range(len(clusters)):
            clusterMember = True
            #difference between similarity scores of any two members is less than threshold.
            for member1 in clusters[i]:
                for member2 in clusters[i]:
                    if( abs(getSimilarity(userSimilarity, user, member1) - getSimilarity(userSimilarity, user, member2) ) > threshold ):
                        clusterMember = False
            if(clusterMember):
                clusters[i].add(user)
                singletonFlag = False

        if(singletonFlag):
            singletons.append(set([user]))
        #print clusters
        #raw_input()

    #clean all singletons
    temp = []
    for cluster in clusters:
        if len(cluster) != 1:
            temp.append(cluster)
    clusters = temp

    clusters.extend(singletons)

    temp = []
    for cluster in clusters:
        if clusters.count(cluster) == 1:
            temp.append(cluster)
    clusters = temp
    
    for cluster in clusters:
        print cluster
    '''

    f = open("clusters.pickle", "w")
    f.write(pickle.dumps(clusters))
    f.close()

def collaborativeRecommend(uid, G, clusters, userSequence, userSimilarity, dbFileName):
    for node in G.nodes():
        G[node]["score"] = 0.0

    for cluster in clusters:
        #print cluster
        #raw_input()
        if uid in cluster:
            #print uid
            #raw_input("node")

            for user in cluster:
                #print userSequence[user]
                for items in userSequence[user]:
                    if (uid != user):
                        G[items[0]]["score"] += getSimilarity(userSimilarity, uid, user)
                        #print node, G[items[0]]["score"]
                        #raw_input()

    maxWeight = float(max([G[node]["score"] for node in G.nodes()]))

    reco = {}
    for node in G.nodes():
        if maxWeight != 0:
            reco[node] = G[node]["score"] / maxWeight
        else:
            reco[node] = 0.0

    fp = open(dbFileName + "_" + str(uid) + "_collabReco.pickle", "w")
    fp.write(pickle.dumps(reco))
    fp.close()

def combineLists(G, alpha, uid, sequence, dbFileName):
    '''
    G (networkx graph object)
    alpha (float) - a number between 0 and 1.
    This function combines the both the generated lists of items into one composite list.
    '''

    #open both the files that contain the recommendations, combine them using the weighted average method, based in their scores rather than the ranks.
    fp = open(dbFileName + "_" + str(uid) + "_contentReco.pickle", "r")
    contentReco = pickle.loads(fp.read())
    fp.close()
    
    fp = open(dbFileName + "_" + str(uid) + "_collabReco.pickle", "r")
    collabReco = pickle.loads(fp.read())
    fp.close()

    combinedReco = {}
    uids = contentReco.keys()

    for uid in uids:
        if uid in sequence:
            combinedReco.pop(uid)

    #score based weighted average
    for uid in uids:
        combinedReco[uid] = alpha*contentReco[uid] + (1 - alpha)*collabReco[uid]

    '''
    ###########################################
    #rank(or index) based weighted average
    contentRecoRank = {}
    valueKey = [(value, key) for (key, value) in contentReco.items()]
    valueKey.sort(reverse = True)
    for i in range(len(valueKey)):
        contentRecoRank[valueKey[1]] = i

    collabRecoRank = {}
    valueKey = [(value, key) for (key, value) in collabReco.items()]
    valueKey.sort(reverse = True)
    for i in range(len(valueKey)):
        collabRecoRank[valueKey[1]] = i

    for uid in uids:
        combinedReco[uid] = alpha*contentReco[uid] + (1 - alpha)*collabReco[uid]    
    ###########################################
    '''

    recos = [ (rate, item) for item,rate in combinedReco.items()]
    recos.sort(reverse = True)
    recos = [(G[item]["name"][0], rate) for rate,item in recos]
    for reco in recos[:20]:
        print reco

    fp = open(dbFileName + "_" + str(uid) + "combinedReco.pickle", "w")
    fp.write(pickle.dumps(recos))
    fp.close()

def createUserData(graphDB, alpha, numberOfUsers, threshold, maxItems, dbFileName):
    '''
    graphDb (networkx Graph object) : The networkx graph object
    alpha (float) : the probability with which the user chooses a neighboring item
    numberOfUsers (integer) : self explanatory
    threshold (integer) : the minimum number of items that the user needs to associate with
    maxItems (integer) : the maximum number of items that the user needs to associate with
    This function creates user data set in the form of a pickle object. The object is a dictionary with the user as the key and a list as the value. UID identifies the user.
    '''

    userData = {}
    for i in range(numberOfUsers):
        # raw_input("press any key to continue..")
        userList = []
        userList.append( (random.choice(graphDB.nodes()), 1) ) #pick some random node as the starting point. To indicate that the user is associated with the item, we'll use the tuple (item, rating). To just indicate whether the user is associated or not (binary states), we'll use either (item, 0) or (item, 1).
        availableChoices = getRealNeighbors(graphDB.neighbors(userList[0][0]))
        while availableChoices == []:
            userList.append( ( random.choice( list( set(graphDB.nodes()).difference(set([item for item, rating in userList])) ) ), 1 ) ) #pick an item that the user is not associated with before and append to the user's list of associated items
            availableChoices = getRealNeighbors(graphDB.neighbors(userList[-1][0]))

        numberOfItems = random.choice(range(threshold, maxItems + 1)) #choose a random number between threshold and maxItems
        for j in range(numberOfItems): #throw a random number. if it is less than alpha, choose a random element from availableChoices List. If it is greater than alpha, pick a random node.
            if(random.random() < alpha):
                userList.append( ( random.choice(availableChoices), 1) )
            else:
                randomChoice = random.choice( list( set(graphDB.nodes()).difference(set([item for item, rating in userList])) ) )
                userList.append( (randomChoice, 1) )

            availableChoices = getRealNeighbors(graphDB.neighbors(userList[-1][0]))
            while availableChoices == []:
                userList.append( ( random.choice( list( set(graphDB.nodes()).difference(set([item for item, rating in userList])) ) ), 1 ) ) #pick an item that the user is not associated with before and append to the user's list of associated items
                availableChoices = getRealNeighbors(graphDB.neighbors(userList[-1][0]))
            
            availableChoices = list(set(availableChoices).difference(set(userList)))
            
        userData[i] = userList
        
    #write userData onto the file
    fp = open(dbFileName + "_userSequence.pickle", "w")
    fp.write(pickle.dumps(userData))
    fp.close()

def main():
    dbFileName = ""
    for arg in sys.argv:
        if re.match("--db=",arg):
            dbFileName = arg.split("=")[1]
    if not dbFileName:
        print "please specify the database name"
        exit()

    userSequence = ""
    for arg in sys.argv:
        if re.match("--usageData=",arg):
            userSequence = arg.split("=")[1]
    if not userSequence:
        print "please specify the usagedata file name"
        exit()

    if "-buildGraph" in sys.argv:
        #print "have to build graph"
        
        #in case the graph isnt availble, write the learnt graph into a file called GraphDB.pickle
        #comment this out when you already have a learnt graph
        print "building graph from", dbFileName
        learnGraph(dbFileName)
        print "done with building graph.."
        
        
    #load the graph onto an object
    f = open(dbFileName + "_GraphDB.pickle", "r")
    G = pickle.loads(f.read())
    f.close()

    for arg in sys.argv:
        if re.match("-generateSequence=",arg):
            print "generating user sequence.."
            config = arg.split("=")[1].split(",")
            numberOfUsers = int(config[0])
            threshold = int(config[1])
            maxItems = int(config[2])
            alpha = float(config[3])
            
            #in case the userSequence is not available, simulate it synthetically using the graph. The sequences are written to a file called userSequence.pickle
            print "\ncreating synthetic user sequences.."

            createUserData(G, alpha, numberOfUsers, threshold, maxItems, dbFileName)
            print "done creating synthetic user sequences.."

    #load the userSequence onto an object
    f = open(dbFileName + "_userSequence.pickle", "r")
    userSequence = pickle.loads(f.read())
    f.close()

    if "-userProfiles" in sys.argv:
        #print "have to create userProfiles"
        
        #each user is associated with his/her own alpha and the attribute importance list
        print "\ncreating userProfiles.."
        userProfiles = {}
        for sequence in userSequence:
            # initializing alpha and weight vector to each user
            userProfiles[sequence] = {}
            userProfiles[sequence]["alpha"] = 0.5
            userProfiles[sequence]["weights"] = {}  # Key the the attribute and value is the corresponding weight for that attr
            tweakWeights(G, userProfiles[sequence], userSequence[sequence])
            normalizeWeights(userProfiles[sequence])

        f = open(dbFileName + "_userProfiles.pickle", "w")
        f.write(pickle.dumps(userProfiles))
        f.close()
        print "done creating userProfiles.."
        

    #load the userProfiles onto an object
    f = open(dbFileName + "_userProfiles.pickle", "r")
    userProfiles = pickle.loads(f.read())
    f.close()

    if "-reduceDimensions" in sys.argv:
        #print "have to reduce dimensions"
        
        #Dimensionality Reduction
        #find out the relative importance of the attributes, by considering the attribute's relative importance from all users. write the file to attributeRelativeImportance.pickle
        print "\nreducing dimensions"
        attributeRelativeImportance(dbFileName)
        print "done reducing dimensions"
        

    for arg in sys.argv:
        if re.match("-userSimilarity=", arg):
            #print "have to compute user similarity"
            upperlim = int(arg.split("=")[1])

            print "building user similarity"
            buildUserSimilarityDict(G, userSequence, userProfiles, dbFileName, upperlim)
            print "done building user similarity"

    f = open(dbFileName + "_userSimilarity.pickle", "r")
    userSimilarity = pickle.loads(f.read())
    f.close()

    for arg in sys.argv:
        if re.match("-formClusters=", arg):
            #print "have to form clusters"
            threshold = float(arg.split("=")[1])
        
            print "forming clusters.."
            formClusters(userSimilarity, threshold, dbFileName)
            print "done forming clusters.."

    f = open(dbFileName + "_clusters.pickle", "r")
    clusters = pickle.loads(f.read())
    f.close()

    uid = ""
    for arg in sys.argv:
        if re.match("-uid=",arg):
            print "have to recommend using uid"
            uid = int(arg.split("=")[1])

    if uid:
        print "List of movies that the user", uid, "has watched: "
        print [G[itemId]["name"][0] for itemId, rating in userSequence[uid]]

        print "\ngetting egocentric recommendation for user ID", uid
        #get the egocentric recommendation, and write it onto a file called contentReco.pickle
        egocentricRecommendation(G, userSequence[uid], dbFileName, uid)
        print "done with egocentric recommendation for user ID", uid, "\n"

        print "\ngetting collaborative recommendation for user ID", uid
        collaborativeRecommend(uid, G, clusters, userSequence, userSimilarity, dbFileName)
        print "done with collaborative recommendation for user ID", uid, "\n"

        #using contentReco.pickle and collabReco.pickle, generate a combined rank list and write it onto combinedReco.pickle
        combineLists(G, userProfiles[uid]["alpha"], uid, userSequence[uid], dbFileName)

def tweakAlpha(userProfile):
    """
        tweak alpha... [will be extended in future but currenlty NO changes are being made]
    """
    pass

def tweakWeights(G, userProfile, itemSequence):
    """
        G: Item Graph
        userProfile: A dictionary for individual users. It contains two keys, alpha and weights
        itemSequence: A list of tuples of the form (item, rating)
    """

    #get the list of items that the user is associated with
    items = [item for item, rating in itemSequence]

    #G.subgraph() function removes the attributes of the node. Since we do not want that to happen, I've written a function that does it. I couldnt figure out a networkx alternative which might be more efficient.
    H = getSubgraph(G, items)
    #get the relative weights of the user towards each attribute
    for node in H.nodes():
        neighbors = getRealNeighbors(H.neighbors(node)) #to chuck the node's attributes and get only the actual neighbors
        for neighbor in neighbors: #scores must be updated for each individual neighbor. This would also mean that the weights would be added twice. But that wouldnt be a problem since we're normalizing it later on.
            #the weight for each attribute must be computed. Hence iterate through the edge attributes, find out the number of common attribute-value pairs for each attribute. If the userProfiles have been updated for the first time, initialize it with the formula. If the attribute has already been updated once, then increment the weights correspondingly.
            for attrib in H[node][neighbor]:
                numOfCommonAttribs = len(H[node][neighbor][attrib])
                if userProfile["weights"].has_key(attrib):
                    userProfile["weights"][attrib] += 1
                    #userProfile["weights"][attrib] += numOfCommonAttribs*(1.0/len(H[node][attrib]) + 1.0/len(H[neighbor][attrib]))
                else:
                    userProfile["weights"][attrib] = 1
                    #userProfile["weights"][attrib] = numOfCommonAttribs*(1.0/len(H[node][attrib]) + 1.0/len(H[neighbor][attrib]))
    #we still have to normalize the weights, which is done after the function returns

def getEdges(G):
    edges = G.edges()
    temp = []
    for edge in edges:
        if type(edge[0]) == int and type(edge[1]) == int:
            temp.append(edge)
    return temp

def getSubgraph(G, items):
    """
        G (networkx graph object) : item relations graph
        items (list) : list of items consumed.
        We need to get the induced subgraph of G. The networkx package removes all the attributes of the node as well.
    """
    #create a new graph that holds the subgraph
    H = nx.Graph()
    for node in items:
        H.add_node(node)    #add very node
        for attrib in G[node]:
            if not (type(attrib) == int and attrib not in items): #add all the attributes. add only those neighbors that are present in the items list.
                H[node][attrib] = G[node][attrib]
    return H

def normalizeWeights(userProfile):
    """
        userProfile (dictionary) : contains 2 keys, alpha and weights
        Normalize the weights in the weight vector. We'll get the relative importance of each individual attribute is quantified
    """
    sumOfWeights = sum(userProfile["weights"].values())
    for attr in userProfile["weights"]:
        userProfile["weights"][attr] /= float(sumOfWeights)

def attributeRelativeImportance(dbFileName):
    """
        In order to determine the relative importance of each attribute, we take the average out the attribute's weight from the all the users.
    """
    f = open(dbFileName + "_userProfiles.pickle", "r")
    userProfiles = pickle.loads(f.read())
    f.close()
    
    ''''''

    pl.ion()
    fig = pl.figure()
    ax = fig.add_subplot(1,1,1)

    ''''''

    avg = {}
    for userProfile in userProfiles:
        for attr in userProfiles[userProfile]["weights"]:
            if avg.has_key(attr):
                avg[attr] += userProfiles[userProfile]["weights"][attr]
            else:
                avg[attr] = userProfiles[userProfile]["weights"][attr]

        x = []
        y = []
        sumOfWeights = sum(avg.values())
        for attr in avg:
            x.append(attr)
            y.append(avg[attr] / float(sumOfWeights))

        ''''''
        ax.clear()
        ax.plot(y)
        for i in range(len(y)):
            if y[i] > 0.025:
                ax.text(i, y[i], x[i])

        pl.draw()
        pl.savefig(dbFileName + "_itemDimensionalityReduction.png")
        #time.sleep(0.001)
        ''''''

    ''''''
    itemPairs = avg.items()
    x = [x1 for x1,y1 in itemPairs]
    y = [y1 for x1,y1 in itemPairs]
    
    for xcoord in range(len(y)):
        if y[xcoord] > 0.025:
            plt.text(xcoord, y[xcoord], x[xcoord])

    plt.plot(y)
    plt.show()
    ''''''

    f = open(dbFileName + "_attributeRelativeImportance.pickle", "w")
    f.write(pickle.dumps(avg))
    f.close()

if __name__ == "__main__":
    main()