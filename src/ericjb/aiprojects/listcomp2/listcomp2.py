'''
Created on Sep 3, 2012

@author: Eric Brodersen
'''
test1=['cLYDe', 'Cooper', 'cOPPed', 'thE', 'Copper', 'ClAPPErs', 'Caper', 'of', 'Choice']

def biggestString(x, y):
    if len(x)>len(y):return x 
    else:return y

test2=map(lambda x: x[:5].lower(), test1)
test3=filter(lambda x: len(x)>5, test1)
test4=reduce(lambda x, y:biggestString(x,y), test1)
print test1
print test2
print test3
print test4
#help(s.lstrip)
#dir(s)