'''
Created on Aug 29, 2012

@author: Eric Brodersen
'''
import heapq
import itertools
import numpy as np
import math
from math import log
import copy
import collections


def rollScore(C, X, A=1, B=1, D=0):
    X-=(A*D)
    B=(A*B)

    baseNums = []
    oldNums = []
    newNums = []
    for i in range(C): 
        baseNums.append(i+1)
        oldNums.append(i+1)
    for i in range(B-1):
        newNums = []
        for j in baseNums:
            for k in oldNums:
                newNums.append(j+k)
        oldNums = copy.deepcopy(newNums)
    counter = collections.Counter(oldNums)
    freqs = {}
    total = C**B
    for i in counter.keys():
        freqs[i] = round((counter[i] / float(total)), 3)
    pLess = 0
    for i in freqs.keys():
        if i < X: pLess +=freqs[i]
    pEqual = freqs[X]
    pMore = 1-(pLess+pEqual)
        
    print(counter.values())
    print(freqs.values())
    
    print "% Less",pLess*100
    print "% Equal",pEqual*100
    print "% More",pMore*100
    return pLess, pEqual, pMore

print '?'

"""
def Harmonic(n):
    gamma = 0.57721566490153286060651209008240243104215933593992
    return gamma + log(n) + 0.5/n - 1./(12*n**2) + 1./(120*n**4)
def reverseHarmonic(h):
    error = 1.0
    r = 1.0
    prev = 0
    tested = []
    close = False
    reverse = 0
    bestError = 1
    while (error > .001 or error < -.001) and int(r) not in tested:
        tested.append(int(r))
        if prev < h and close == False:
            r = r + math.ceil(h/100.0)
            prev = r*Harmonic(r)
        elif prev > h:
            close = True
            r = r - math.ceil(h/1000.0)
            prev = r*Harmonic(r)
        error = (h - prev)/float(h)
        if abs(error) < abs(bestError):
            bestError = error
            reverse = int(r)
    return reverse

print reverseHarmonic(50000)
print reverseHarmonic(20)
print reverseHarmonic(1)
print reverseHarmonic(1000)
"""





class BetterPriorityQueue:
    """
    This Class is a combination of the util.PriorityQueue 
    and information found on http://docs.python.org/library/heapq.html
    regarding how to remove tasks
    
    Implements a priority queue data structure. Each inserted item
    has a priority associated with it and the client is usually interested
    in quick retrieval of the lowest-priority item in the queue. This
    data structure allows O(1) access to the lowest-priority item.
    
    Note that this PriorityQueue does not allow you to change the priority
    of an item.  However, you may insert the same item multiple times with
    different priorities
    """  
    def  __init__(self):  
        self.heap = []
        self.entry_finder = {}
        self.REMOVED = '<removed-item>'
        self.counter = itertools.count()
        
    def push(self, item, priority):
        pair = (priority,item)
        heapq.heappush(self.heap,pair)
    def pop(self):
        (priority,item) = heapq.heappop(self.heap)
        return item
    def isEmpty(self):
        return len(self.entry_finder) == 0
    
    def add_item(self, item, priority=0): 
        if item in self.entry_finder:
            self.remove_item(item)
        count = next(self.counter)
        entry = [priority, count, item]
        self.entry_finder[item] = entry
        heapq.heappush(self.heap, entry)
    
    def remove_item(self, item):
        entry = self.entry_finder.pop(item)
        entry[-1] = self.REMOVED
    
    def pop_item(self):
        while self.heap:
            priority, count, item = heapq.heappop(self.heap)
            if item is not self.REMOVED:
                del self.entry_finder[item]
                return item
        raise KeyError('pop from an empty priority queue')
    
"""
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

print newList

numberList = [6, 5, 4, 3, 2, 1]
maxValue = min(numberList)
print maxValue


def function(priority=0):
    print priority

function(1)
function()

frontier = BetterPriorityQueue()

frontier.add_item('A', 1)
frontier.add_item('B', 2)
frontier.add_item('C', 3)
frontier.add_item('D', 4)
frontier.add_item('E', 12)

while frontier.isEmpty() == False:
    print frontier.pop_item()

frontier.add_item('A', 1)
frontier.add_item('B', 2)
frontier.add_item('C', 3)
frontier.add_item('D', 4)
frontier.add_item('E', 12)
frontier.add_item('E', 0)
while frontier.isEmpty() == False:
    print frontier.entry_finder
    print frontier.heap
    print frontier.pop_item()
    print frontier.entry_finder
    print frontier.heap


frontier.add_item('A', 1)
frontier.add_item('B', 2)
frontier.add_item('C', 3)
frontier.add_item('D', 4)
frontier.add_item('E', 12)
frontier.add_item('E', 0)
frontier.add_item('E', 12)

while frontier.isEmpty() == False:
    print frontier.entry_finder
    print frontier.heap
    print frontier.pop_item()
    print frontier.entry_finder
    print frontier.heap
"""




