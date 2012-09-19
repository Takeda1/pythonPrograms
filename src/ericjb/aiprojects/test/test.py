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

print len(newList)

newSet = set([(1, 2), (2, 3), (3, 4), (4, 5)])
newTriple = (55, 33, newSet)
newTriple[2].add((5, 6))
for i in newTriple[2]:
    print i
x,y = newTriple[0],newTriple[1]

print x,y

newFrozenSet = frozenset(newSet)

print newFrozenSet
print newSet