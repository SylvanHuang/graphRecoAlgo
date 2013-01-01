from BeautifulSoup import BeautifulSoup
import re
import urllib
import lxml.html
import string
import json
import pickle

for char in string.uppercase:
    movieInfoList = []
    html = urllib.urlopen('http://www.gomolo.com/indian-movies-list-films-database?SearchChar=' + char)
    soup = BeautifulSoup(html.read())

    #print soup.html.head.title.string
    items = soup.findAll("div",attrs={"id":"divMain"})[0].contents[0].contents 

    movielinks = []
    for item in items:
        try:
            movielinks.append(item.contents[0].contents[0].attrs[0][1])
        except IndexError:
            print "IndexError"
            pass

    #movielinks = ['http://www.gomolo.com/bal-hanuman-2-movie/39179']

    for link in movielinks:

        movieInfo = {}

        arr = link.split("/")
        arr[-2] = arr[-2] + '-cast-crew'
        link = '/'.join(arr)
        
        markup = urllib.urlopen(link)
        markupString = markup.read()
        tree = lxml.html.fromstring(markupString)
        
        movieName = tree.get_element_by_id("LblMvName").text
        movieInfo['name'] = movieName
        #print type(movieName)
        print movieName
        
        year = tree.get_element_by_id("lblMovieyear").text[1:]
        movieInfo['year'] = year
        #print type(year)
        #print year
        
        img_url = tree.get_element_by_id("imgMovie").items()[2][1]
        movieInfo['img_url'] = img_url
        #print type(img_url)

        img_bin = urllib.urlopen(img_url)
        movieInfo['img_data'] = img_bin.read()
        #print type(movieInfo['img_data'])

        """
        f = open("openme.jpg", "w")
        f.write(movieInfo['img_data'])
        f.close()
        """

        cast = []
        elem = tree.get_element_by_id("divCast")
        children = elem.getchildren()
        if children:
            children.pop(0)
            for child in children:
                if child.getchildren()[0].getchildren():
                    cast.append(child.getchildren()[0].getchildren()[0].text)
                    #print type(child.getchildren()[0].getchildren()[0].text)
                else:
                    cast.append(child.getchildren()[0].text)
                    #print type(child.getchildren()[0].text)

            movieInfo['cast'] = cast
            #print cast
            
        elem = tree.get_element_by_id("divCrew")
        children = elem.getchildren()
        if children:
            children.pop(0)
            current = ''
            for child in children:
                crewType = child.get_element_by_id("lblCrewtype").text
                #print crewType, 
                if crewType != u'\xa0':
                    current = crewType
                
                #print type(current)
                if child.getchildren()[1].getchildren():
                    if movieInfo.has_key(current):
                        movieInfo[current].append(child.getchildren()[1].getchildren()[0].text)
                        #print type(child.getchildren()[1].getchildren()[0].text)
                    else:
                        movieInfo[current] = [child.getchildren()[1].getchildren()[0].text]
                        #print type(child.getchildren()[1].getchildren()[0].text)
                    #print child.getchildren()[1].getchildren()[0].text
                else:
                    if movieInfo.has_key(current):
                        movieInfo[current].append(child.getchildren()[1].text)
                        #print type(child.getchildren()[1].text)
                    else:
                        movieInfo[current] = [child.getchildren()[1].text]
                        #print type(child.getchildren()[1].text)
                    #print child.getchildren()[1].text
        
        movieInfoList.append(movieInfo)
        
    tempdb = {}
    tempdb["database"] = movieInfoList

    strs = pickle.dumps(movieInfoList)
    fp = open("gomolo" + char + ".pickle", "w")
    fp.write(strs)
    fp.close()
    print "Successfully written to " + "gomolo" + char + ".pickle"

#strs = json.dumps(tempdb)
#print type(strs)
#outputFile.write(strs)