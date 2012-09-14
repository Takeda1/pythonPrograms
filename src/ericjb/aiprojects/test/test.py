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