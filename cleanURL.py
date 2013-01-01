import pickle
import json

inp = open("JSONGomoloText", "r")
out = open("JSONGomoloTextNew", "w")
for line in inp:
    diction = eval(line)
    if diction['img_url'][0] == 'http://img1.gomolo.com/images/common/movavatar108.gif':
        diction.pop(u'img_url')
        #print "popped from " + diction['name']
    out.write(json.dumps(diction) + '\n')

inp.close()
out.close()