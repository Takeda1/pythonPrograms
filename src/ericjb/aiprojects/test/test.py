'''
Created on Aug 29, 2012

@author: Eric Brodersen
'''

newList = [((1, 2), 'South', 1), ((2, 3), 'North', 1), ((3, 4), 'East', 1)]
#print newList[0][0]+newList[0][1]
#newTuple = (1, 2)
#print newTuple[0]+newTuple[1]
for i in newList:
    print i[1]

print newList
newList.reverse()
print newList

aDict1 = {'crap': 5}
aDict2 = {'yay': 3}

aTuple1 = ('crap', 1)
aTuple2 = ('yay', 2)

frontierSet = set()
frontierSet.add(aTuple1)
frontierSet.add(aTuple2)

print frontierSet
frontierSet.remove(('crap', 1))
print frontierSet