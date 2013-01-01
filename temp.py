import matplotlib.pyplot as plt
import pickle

u = pickle.loads(open("JSONGomoloText_userProfiles.pickle","r").read())
itemPairs = u[238]['weights'].items()

x = [x1 for x1,y1 in itemPairs]
y = [y1 for x1,y1 in itemPairs]

print y
print x
print len(y)
print len(x)
for xcoord in range(len(y)):
    print xcoord
    if y[xcoord] > 0.025:
        plt.text(xcoord, y[xcoord], x[xcoord])

plt.plot(y)
plt.show()