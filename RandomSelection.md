# Random strategy selection

## Numpy
```
import numpy as np
"""
Set number of strategies
Initialize with same/uniform probability
get just first dimention
print
"""
Size = 5
gopp = np.ones((1, Size))
gopp = gopp * (1/Size)
gopp = gopp[0]
print(gopp)
"""
lets increase probability of strategy at index 0
set learning rate to 1
normalize probabilities
print
"""
positive_strtaegy_index = 0
LEARNER_RATE = 1
gopp[positive_strtaegy_index] = (1 + np.random.uniform(0,1,1))*LEARNER_RATE*gopp[positive_strtaegy_index]
gopp = gopp/np.sum(gopp)
print(gopp)

"""
lets decrese probability of strategy at index 3
set learning rate to 1
normalize probabilities
print
"""
negative_strtaegy_index = 3
gopp[negative_strtaegy_index] = (1 - np.random.uniform(0,1,1))*LEARNER_RATE*gopp[negative_strtaegy_index]  
gopp = gopp/np.sum(gopp)
print(gopp)

"""
lets sort strategies by probability from most to less probable
print index ranking
"""
probOrderIndex = np.flip(np.argsort(gopp))
print("index ranking")
print(probOrderIndex)
"""
lets sort strategies by probability from most to less probable
print probability ranking
Build comulative probability array
print
"""
opgSrt = np.flip(np.sort(gopp))
print("probability ranking")
print(opgSrt)
comulativeGopp = np.cumsum(opgSrt, axis =-1)
print("comulative probability")
print(comulativeGopp)

"""
random strategy selection
0-0.2 Lets try something really effective if there is such strategy
0.2-0.4
0.4-0.6
0.6-0.8
0.8-1 Lets give a chance to the worst strategy
"""
gopSelp = np.random.uniform(0,1,1)
print(gopSelp)


print(np.where(comulativeGopp >= gopSelp))

next_strategy_index = probOrderIndex[np.where(gopSelp <= comulativeGopp)[0][0]]

print(next_strategy_index)
```
