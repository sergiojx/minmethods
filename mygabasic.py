from bitstring import BitArray
import numpy as np
import math
import sys

print("HOlA MYGABASIC")

# Generate val binary code list
def gcoding(val = 0, minVal = -5.12, maxVal = 5.12, codeLen = 16):
    pnorm = (val - minVal)/(maxVal - minVal)
    maxCodeVal = np.power(2, codeLen) - 1
    intOut = np.rint(pnorm * maxCodeVal)
    return np.array(list(np.binary_repr(int(intOut)).zfill(codeLen))).astype(np.int8) 

# Generate population
def pgenerator(N=100,d = 10):
    return np.random.randint(2, size=(N, d)).astype(np.uint8)

# Generate val from bin lint
def gdeco(bitlist = 0, minVal = -5.12, maxVal = 5.12, codeLen = 16):
    b = BitArray(bitlist)
    dig = b.uint
    rang = (maxVal - minVal)/(np.power(2, codeLen) - 1)
    infBound = minVal + rang*(dig)
    supBound = minVal + rang*(dig + 1)
    return (infBound + supBound)/2

# Workout fitness
def popfitnes(pop = 0,minVal = -5.12, maxVal = 5.12, codeLen = 16, genQty = 2, funct = None):
    splitPop = pop.reshape((pop.shape[0],-1,codeLen))
    lalout = np.zeros((pop.shape[0],genQty))
    index = 0
    for dimSet in splitPop:
        valOut = [gdeco(bitlist = dim, minVal = minVal, maxVal = maxVal, codeLen = codeLen) for dim in dimSet]
        lalout[index] = valOut
        index = index + 1
        
    fitList = [funct(np.expand_dims(varOut, axis=-1)) for varOut in lalout]
    return fitList,lalout

# Workout fitness for GS test functions (mapping functions)
def popfitnes4BinMapping(pop = 0, funct = None):
    fitList = [funct(binString) for binString in pop]
    return fitList

# Return ascendent ordered pop
def rankWeighting(fitList = 0,fitVal = 0, pop = 0, keep = 1):
    rm = int(len(fitList) - (len(fitList)*keep))
    ordAscArr = np.argsort(fitList)
    ordPop = pop[ordAscArr[::]]
    ordFitVal = fitVal[ordAscArr[::]]
    ordFitLst = np.sort(fitList)
    if rm > 0:
        ordPop, ordFitVal, ordFitLst = ordPop[:-rm, :], ordFitVal[:-rm, :],ordFitLst[:-rm]
    divisor = (len(ordFitLst)*(len(ordFitLst) + 1))/2
    rkW = [(len(ordFitLst) - (i+1) + 1)/divisor for i in range(len(ordFitLst))]
    # np.cumsum(rkW)
    return ordPop, ordFitVal,ordFitLst,np.cumsum(rkW)

def costWeighting(fitList = 0,fitVal = 0, pop = 0, keep = 1):
    rm = int(len(fitList) - (len(fitList)*keep))
    ordAscArr = np.argsort(fitList)
    ordPop = pop[ordAscArr[::]]
    ordFitVal = fitVal[ordAscArr[::]]
    ordFitLst = np.sort(fitList)
    if rm > 0:
        ordPop, ordFitVal, ordFitLst = ordPop[:-rm, :], ordFitVal[:-rm, :],ordFitLst[:-rm]
    worst =   ordFitLst[-1] + abs(ordFitLst[-1])*0.1
    ctW = ordFitLst - worst
    smtr = np.sum(ctW)
    ctW = ctW/smtr
    return ordPop, ordFitVal,ordFitLst,np.cumsum(np.absolute(ctW))

# Weighted Parents selection
def weightedSelection(ordPop, distribution, pairsQty):
    parentOut = np.zeros(ordPop.shape).astype(np.uint8)
    for i in range(0,pairsQty*2,2):
        p1 = np.random.uniform(0,1,1)
        p2 = np.random.uniform(0,1,1)
        if np.isnan(distribution).any() == True:
            parentOut[i] = ordPop[0]
            parentOut[i+1] = ordPop[0]
        else:
            parentOut[i] = ordPop[np.where( p1 <= distribution)[0][0]]
            parentOut[i+1] = ordPop[np.where( p2 <= distribution)[0][0]]
       
           
        
    return parentOut

# Tournament Parents selection
def tournamentSelection(ordPop, fitList, pairsQty):
    parentOut = np.zeros(ordPop.shape).astype(np.uint8)
    for i in range(0,pairsQty*2,2):
        cp1 = np.random.randint(len(fitList))
        cp2 = np.random.randint(len(fitList))
        cp3 = np.random.randint(len(fitList))
        cp4 = np.random.randint(len(fitList))
        c0 = cp1
        c1 = cp3
        if fitList[cp1] > fitList[cp2]:
            c0 = cp2
        if fitList[cp3] > fitList[cp4]:
            c1 = cp4
        parentOut[i] = ordPop[c0]
        if fitList[c0] > fitList[c1]:
            parentOut[i] = ordPop[c1]
        
        cp1 = np.random.randint(len(fitList))
        cp2 = np.random.randint(len(fitList))
        cp3 = np.random.randint(len(fitList))
        cp4 = np.random.randint(len(fitList))
        c0 = cp1
        c1 = cp3
        if fitList[cp1] > fitList[cp2]:
            c0 = cp2
        if fitList[cp3] > fitList[cp4]:
            c1 = cp4
        parentOut[i+1] = ordPop[c0]
        if fitList[c0] > fitList[c1]:
            parentOut[i+1] = ordPop[c1]
            
    return parentOut
        
# Generates children      
def childenGen(parents, matingPercent = 0.7, mutaPercent = 0.03):
    childrenOut = np.zeros(parents.shape).astype(np.uint8)
    for i in range(0,parents.shape[0],2):
        P1 = parents[i]
        P2 = parents[i+1]
        matingp = np.random.uniform(0,1,1)
        if matingp <= matingPercent:
            # print('Mating')
            # print(P1)
            # print(P2)
            
            crossover_point = np.random.randint((parents.shape[1] - 1))
            out1 = np.append(P1[0:(crossover_point + 1)], P2[crossover_point + 1:])
            out2 = np.append(P2[0:(crossover_point + 1)], P1[crossover_point + 1:])
            
            mup = np.random.uniform(0,1,parents.shape[1])
            mask = ((1-mutaPercent) <= mup).astype(np.uint8)
            mutate = np.where( mask == 1)
            mutation = np.logical_not(out1[mutate[0]]).astype(np.uint8)
            out1[mutate[0]] = mutation
            
            mup = np.random.uniform(0,1,parents.shape[1])
            mask = ((1-mutaPercent) <= mup).astype(np.uint8)
            mutate = np.where( mask == 1)
            mutation = np.logical_not(out2[mutate[0]]).astype(np.uint8)
            out2[mutate[0]] = mutation
            # print(out1)
            # print(out2)
            childrenOut[i] = out1
            childrenOut[i+1] = out2
        else:
            childrenOut[i] = P1
            childrenOut[i+1] = P2
    
    return childrenOut

    
    
    