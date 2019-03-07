from bitstring import BitArray
from scipy.spatial import distance
import numpy as np
import math
import sys
from random import shuffle

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

# Generate random normalized probabilities for Operator selection in HAEA
def pgenerator4Gop(N=100,d = 10):
    # opg = np.random.uniform(0,1,(N,d))
    # opgSum = np.expand_dims(np.sum(opg, axis=-1), axis=-1)
    # return opg/opgSum
    gopp = np.ones((N, d))
    gopp = gopp * (1/d)
    return gopp

# Generate val from bin lint
def gdeco(bitlist = 0, minVal = -5.12, maxVal = 5.12, codeLen = 16):
    b = BitArray(bitlist)
    dig = b.uint
    rang = (maxVal - minVal)/(np.power(2, codeLen) - 1)
    infBound = minVal + rang*(dig)
    supBound = minVal + rang*(dig + 1)
    return (infBound + supBound)/2

# Workout fitness Multy Modal SHARING
""" 
def popfitnes_MultyM(pop = 0,minVal = -5.12, maxVal = 5.12, codeLen = 16, genQty = 2, funct = None, sigma = 0.2):
    splitPop = pop.reshape((pop.shape[0],-1,codeLen))
    lalout = np.zeros((pop.shape[0],genQty))
    index = 0
    for dimSet in splitPop:
        valOut = [gdeco(bitlist = dim, minVal = minVal, maxVal = maxVal, codeLen = codeLen) for dim in dimSet]
        lalout[index] = valOut
        index = index + 1
    # TO DO: Find neighbors for each individual 
    
    fitList = [funct(np.expand_dims(varOut, axis=-1)) for varOut in lalout]
    for varOut in lalout:
        count = 0
        for idv in pop.shape[0]:
            dx = distance.euclidean(varOut, lalout[idv])
            if dx <= sigma:
                count = count + 1
            
    return fitList,lalout
"""    
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

# Workout fitness for Evolution Strategies
def popfitnes4ES(pop = 0, funct = None):
    fitList = [funct(np.expand_dims(varOut, axis=-1)) for varOut in pop]
    return fitList
    
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
# function for NSGA II multyobjective optimization
def frontSetBuilder(fitListF1,fitListF2,Pop):
    # list of indexed fronts. Each item holds a list of individual indexes belonging to a front
    # in a ascending front rank. Each index identify each individual into the population
    # arragement.
    frontLit = []
    # first front workout
    front = dominance4AllMinTwoFunctions(fitListF1,fitListF2)
    fontprev = front.copy()
    # subsequent nondominated fronts
    while len(front) > 0:
        frontLit.append(front)
        front = dominance4AllMinTwoFunctions(fitListF1,fitListF2,fontprev)
        fontprev = fontprev + front
    

    """
    The crowding-distance computation requires sorting the population
    according to each objective function value in ascending
    order of magnitude
    """
    fronRanks = []
    global_fronRanks = []
    for j in range(len(frontLit)):
        # F1
        frotnFitListF1 = [fitListF1[i] for i in frontLit[j]]
        # distance array. Implicit initial order math with f1 rank order
        # for first funtion reordering distance array is not required!!!
        distance = np.zeros(len(frotnFitListF1))
        distance[0] = np.inf
        distance[len(frotnFitListF1) - 1] = np.inf
        # f1 sorting
        frotnFitListF1_SortedIdxs = np.argsort(frotnFitListF1)
        frotnFitListF1np = np.asarray(frotnFitListF1)[frotnFitListF1_SortedIdxs]

        f1Min = np.amin(frotnFitListF1) 
        f1Max = np.amax(frotnFitListF1) 
        # f1 component
        if len(frotnFitListF1) == 3:
            distance[1] = distance[1] + (frotnFitListF1np[2] - frotnFitListF1np[0])/(f1Max-f1Min)
        elif len(frotnFitListF1) > 3:
            for w in range(1,(len(frotnFitListF1) - 1)):
                distance[w] = distance[w] + (frotnFitListF1np[w + 1] - frotnFitListF1np[w - 1])/(f1Max-f1Min)
        """
        print(frotnFitListF1)
        print(frotnFitListF1_SortedIdxs)
        print(frotnFitListF1np)
        print(distance)
        """
    
    
        # reorder distance to match initial individual "position" into pop.
        np.put(distance, frotnFitListF1_SortedIdxs, distance.copy())
        # print(distance)

        # F2
        frotnFitListF2 = [fitListF2[i] for i in frontLit[j]]
        # f2 sorting
        frotnFitListF2_SortedIdxs = np.argsort(frotnFitListF2)
        frotnFitListF2np = np.asarray(frotnFitListF2)[frotnFitListF2_SortedIdxs]

        # pop sorting
        # this is done for last function
        # raked individuals into a front
        # are returned ordered accoriding to last function evaluation
        frontValList = [Pop[i] for i in frontLit[j]]
        frontValListnp = np.asarray(frontValList)[frotnFitListF2_SortedIdxs]
        # reorder distance array!!!
        distance = distance[frotnFitListF2_SortedIdxs]

        f2Min = np.amin(frotnFitListF2)
        f2Max = np.amax(frotnFitListF2)
        # f2 component
        if len(frotnFitListF2) == 3:
            distance[1] = distance[1] + (frotnFitListF2np[2] - frotnFitListF2np[0])/(f2Max-f2Min)
        elif len(frotnFitListF2) > 3:
            for w in range(1,(len(frotnFitListF2) - 1)):
                distance[w] = distance[w] + (frotnFitListF2np[w + 1] - frotnFitListF2np[w - 1])/(f2Max-f2Min)

        """
        print(frotnFitListF2)
        print(frotnFitListF2_SortedIdxs)
        print(frotnFitListF2np)
        print(distance)
        """
   
        rankedFront = []  
        for w in range(len(frotnFitListF1)):
            rankedFront.append({'rank':distance[w],'indiv':frontValListnp[w],'front':j})
            global_fronRanks.append({'rank':distance[w],'indiv':frontValListnp[w],'front':j})

        fronRanks.append(rankedFront)
    return global_fronRanks,fronRanks,frontLit

def childenGen4MultyObjMin(parents, selection, matingPercent = 0.7, mutaPercent = 0.03):
    childrenOut = np.zeros(parents.shape).astype(np.uint8)
    for i in range(0,parents.shape[0]):
        P1 = parents[i]
        P2 = selection[i]
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
            chSelectionp = np.random.uniform(0,1,1)
            if chSelectionp > 0.5:
                childrenOut[i] = out1
            else:
                childrenOut[i] = out2
        else:
            childrenOut[i] = P1
            
    
    return childrenOut

# SPEAII Raw Fitness workout
def speaIIRawfitness4AllMinTwoFunctions(fitListF1,fitListF2):
    
    Strength = np.zeros(len(fitListF1))
    Rawfitness = np.zeros(len(fitListF1))
    for p in range(len(fitListF1)):
        for q in range(len(fitListF1)):
            if (fitListF1[q] <= fitListF1[p] and fitListF2[q] <= fitListF2[p]) and (fitListF1[q] < fitListF1[p] or fitListF2[q] < fitListF2[p]):
                Strength[q] = Strength[q] + 1
        
    for p in range(len(fitListF1)):
        for q in range(len(fitListF1)):
            if (fitListF1[q] <= fitListF1[p] and fitListF2[q] <= fitListF2[p]) and (fitListF1[q] < fitListF1[p] or fitListF2[q] < fitListF2[p]):
                Rawfitness[p] = Rawfitness[p] + Strength[q]
    
                          
    return  Rawfitness 
    
    
# SPEAII Distance workout
def speaIIDistance4AllMinTwoFunctions(fitListF1,fitListF2):
    
    k = int(math.sqrt(len(fitListF1)))
    D = np.zeros(len(fitListF1))
    distances = np.zeros((len(fitListF1),len(fitListF1)))
    stack = np.stack((fitListF1, fitListF2), axis=1)
    distances = np.zeros((len(fitListF1),len(fitListF1)))
    for p in range(len(fitListF1)):
        for q in range(len(fitListF1)):
            distances[p][q] = distance.euclidean(stack[p], stack[q])
    
    distances.sort(axis=1)
    D = distances[:,k]
    Dreturn = 1/(D + 2)   
        
    return distances,Dreturn

def tournamentSelection4MultyObjMin(ordPop, ordPopDic):
    parentOut = np.zeros(ordPop.shape).astype(np.uint8)
    
    for i in range(len(ordPopDic)):
        
        cp1 = np.random.randint(len(ordPopDic))
        cp2 = np.random.randint(len(ordPopDic))
        cp3 = np.random.randint(len(ordPopDic))
        cp4 = np.random.randint(len(ordPopDic))
        c0 = cp1
        c1 = cp3
        if (ordPopDic[cp1]['front'] < ordPopDic[cp2]['front'] or
            (ordPopDic[cp1]['front'] == ordPopDic[cp2]['front'] and ordPopDic[cp1]['rank'] > ordPopDic[cp2]['rank'])):
            c0 = cp1
        if (ordPopDic[cp2]['front'] < ordPopDic[cp1]['front'] or
            (ordPopDic[cp2]['front'] == ordPopDic[cp1]['front'] and ordPopDic[cp2]['rank'] > ordPopDic[cp1]['rank'])):
            c0 = cp2
            
        if (ordPopDic[cp3]['front'] < ordPopDic[cp4]['front'] or
            (ordPopDic[cp3]['front'] == ordPopDic[cp4]['front'] and ordPopDic[cp3]['rank'] > ordPopDic[cp4]['rank'])):
            c1 = cp3
        if (ordPopDic[cp4]['front'] < ordPopDic[cp3]['front'] or
            (ordPopDic[cp4]['front'] == ordPopDic[cp3]['front'] and ordPopDic[cp4]['rank'] > ordPopDic[cp3]['rank'])):
            c1 = cp4
            
        if (ordPopDic[c0]['front'] < ordPopDic[c1]['front'] or
            (ordPopDic[c0]['front'] == ordPopDic[c1]['front'] and ordPopDic[c0]['rank'] > ordPopDic[c1]['rank'])):
            parentOut[i] = ordPopDic[c0]['indiv']
            
        if (ordPopDic[c1]['front'] < ordPopDic[c0]['front'] or
            (ordPopDic[c1]['front'] == ordPopDic[c0]['front'] and ordPopDic[c1]['rank'] > ordPopDic[c0]['rank'])):
            parentOut[i] = ordPopDic[c1]['indiv']
            
            
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

# Tournament Parents selection
def tournamentSelection4Max(ordPop, fitList, pairsQty):
    parentOut = np.zeros(ordPop.shape).astype(np.uint8)
    for i in range(0,pairsQty*2,2):
        cp1 = np.random.randint(len(fitList))
        cp2 = np.random.randint(len(fitList))
        cp3 = np.random.randint(len(fitList))
        cp4 = np.random.randint(len(fitList))
        c0 = cp1
        c1 = cp3
        if fitList[cp1] < fitList[cp2]:
            c0 = cp2
        if fitList[cp3] < fitList[cp4]:
            c1 = cp4
        parentOut[i] = ordPop[c0]
        if fitList[c0] < fitList[c1]:
            parentOut[i] = ordPop[c1]
        
        cp1 = np.random.randint(len(fitList))
        cp2 = np.random.randint(len(fitList))
        cp3 = np.random.randint(len(fitList))
        cp4 = np.random.randint(len(fitList))
        c0 = cp1
        c1 = cp3
        if fitList[cp1] < fitList[cp2]:
            c0 = cp2
        if fitList[cp3] < fitList[cp4]:
            c1 = cp4
        parentOut[i+1] = ordPop[c0]
        if fitList[c0] < fitList[c1]:
            parentOut[i+1] = ordPop[c1]
            
    return parentOut

# Random Uniform Parents selection for geological ston
# classification
def randomUniformSelection4Geological(objList, Qty):
    selectedList = []
    for i in range(Qty):
        selectedList.append(objList[np.random.randint(Qty)])
            
    return selectedList

# Random Uniform Parents selection
def randomUniformSelection(ordPop, fitList, pairsQty):
    parentOut = np.zeros(ordPop.shape).astype(np.uint8)
    for i in range(0,pairsQty*2,2):
        parentOut[i] = ordPop[np.random.randint(len(fitList))]
        parentOut[i+1] = ordPop[np.random.randint(len(fitList))]
            
    return parentOut

# Random Uniform Parents selection
def randomUniformTripletSelection(ordPop, fitList, fitVal, pairsQty, d):
    parentOut = np.zeros((pairsQty*3, d))
    for i in range(0,pairsQty*3,3):
        parentOut[i] = fitVal[np.random.randint(len(fitList))]
        parentOut[i+1] = fitVal[np.random.randint(len(fitList))]
        parentOut[i+2] = fitVal[np.random.randint(len(fitList))]
            
    return parentOut

# Tournament Parents selection. Return Indexes
def tournamentSelectionIndexes(ordPop, fitList, pairsQty):
    parentOut = np.zeros((ordPop.shape[0],2)).astype(np.uint8)
    for i in range(0,pairsQty,1):
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
        parentOut[i][0] = c0
        if fitList[c0] > fitList[c1]:
            parentOut[i][0] = c1
        
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
        parentOut[i][1] = c0
        if fitList[c0] > fitList[c1]:
            parentOut[i][1] = c1
            
    return parentOut


def tournamentSelection4Max(ordPop, fitList, pairsQty):
    parentOut = np.zeros(ordPop.shape).astype(np.uint8)
    for i in range(0,pairsQty*2,2):
        cp1 = np.random.randint(len(fitList))
        cp2 = np.random.randint(len(fitList))
        cp3 = np.random.randint(len(fitList))
        cp4 = np.random.randint(len(fitList))
        c0 = cp1
        c1 = cp3
        if fitList[cp1] < fitList[cp2]:
            c0 = cp2
        if fitList[cp3] < fitList[cp4]:
            c1 = cp4
        parentOut[i] = ordPop[c0]
        if fitList[c0] < fitList[c1]:
            parentOut[i] = ordPop[c1]
        
        cp1 = np.random.randint(len(fitList))
        cp2 = np.random.randint(len(fitList))
        cp3 = np.random.randint(len(fitList))
        cp4 = np.random.randint(len(fitList))
        c0 = cp1
        c1 = cp3
        if fitList[cp1] < fitList[cp2]:
            c0 = cp2
        if fitList[cp3] < fitList[cp4]:
            c1 = cp4
        parentOut[i+1] = ordPop[c0]
        if fitList[c0] < fitList[c1]:
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

# Multy modal Generates children for Deterministic Crowding     
def childenGen_MultyM(parents, matingPercent = 0.7, mutaPercent = 0.03,minVal = 0,maxVal = 1 ,codeLen = 16,funct = None):
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
            
            # TODO: support more than one dimention 
            gdecoOut1 = gdeco(bitlist = out1, minVal = minVal, maxVal = maxVal, codeLen = codeLen)
            gdecoOut2 = gdeco(bitlist = out2, minVal = minVal, maxVal = maxVal, codeLen = codeLen)
            gdecoP1 = gdeco(bitlist = P1, minVal = minVal, maxVal = maxVal, codeLen = codeLen)
            gdecoP2 = gdeco(bitlist = P2, minVal = minVal, maxVal = maxVal, codeLen = codeLen)
            edistance1 = distance.euclidean(gdecoP1, gdecoOut1) + distance.euclidean(gdecoP2, gdecoOut2)
            edistance2 = distance.euclidean(gdecoP1, gdecoOut2) + distance.euclidean(gdecoP2, gdecoOut1)
            # Child selection
            if edistance1 < edistance2:
                fith = funct(gdecoOut1)
                fitp = funct(gdecoP1)
                if  fith >= fitp:
                    childrenOut[i] = out1
                else:
                    childrenOut[i] = P1
                fith = funct(gdecoOut2)
                fitp = funct(gdecoP2)
                if  fith >= fitp:
                    childrenOut[i + 1] = out1
                else:
                    childrenOut[i + 1] = P2
            else:
                fith = funct(gdecoOut2)
                fitp = funct(gdecoP1)
                if  fith >= fitp:
                    childrenOut[i] = out2
                else:
                    childrenOut[i] = P1
                fith = funct(gdecoOut1)
                fitp = funct(gdecoP2)
                if  fith >= fitp:
                    childrenOut[i + 1] = out1
                else:
                    childrenOut[i + 1] = P2
        else:
            childrenOut[i] = P1
            childrenOut[i+1] = P2
    
    return childrenOut 


def childenGen_MultyM_haeaMax4Geological(objList, pselect, d, imgeo, codeLen, function, mutationP, lnrate):
    childrenOut = []
    LEARNER_RATE = lnrate
    for indiv in range(0, len(objList)):
        probOrderIndex = np.flip(np.argsort(objList[indiv]["gopp"]), axis = -1)
        opgSrt = np.flip(np.sort(objList[indiv]["gopp"]), axis = -1)
        comulativeGopp = np.cumsum(opgSrt, axis = -1)
        # Select genetic operator
        gopSelp = np.random.uniform(0,1,1)
        # 0: Mutation, 1: Mating
        operator = probOrderIndex[np.where(gopSelp <= comulativeGopp)[0][0]]
        # Load individuals
        P1 = objList[indiv]["indivudual"]
        P2 = pselect[indiv]["indivudual"]
        if operator == 0:
            # This is a big mistake, If out1 changes, P1 changes too
            # out1 = P1
            out1 = np.copy(np.flip(P1,0))
            out2 = np.copy(P1)
            
            mup = np.random.uniform(0,1,d)
            # mutaPercent = 1/(population.shape[1])
            mutaPercent = mutationP
            mask = (mup <= mutaPercent).astype(np.uint8)
            mutate = np.where( mask == 1)
            mutation = np.logical_not(out1[mutate[0]]).astype(np.uint8)
            out1[mutate[0]] = mutation
            # out2 = np.copy(out1)
            mup = np.random.uniform(0,1,d)
            # mutaPercent = 1/(population.shape[1])
            mutaPercent = mutationP
            mask = (mup <= mutaPercent).astype(np.uint8)
            mutate = np.where( mask == 1)
            mutation = np.logical_not(out1[mutate[0]]).astype(np.uint8)
            out2[mutate[0]] = mutation
            
            # Main Parent
            # p1fit = function(imgeo, P1, codeLen)
            out1fit = function(imgeo, out1, codeLen)
            out2fit = function(imgeo, out2, codeLen)
            
            edistance1 = distance.euclidean((objList[indiv]["aVal"],objList[indiv]["fqVal"],objList[indiv]["agVal"]), 
                                            (out1fit["aVal"],out1fit["fqVal"],out1fit["agVal"])) 
            edistance2 = distance.euclidean((objList[indiv]["aVal"],objList[indiv]["fqVal"],objList[indiv]["agVal"]), 
                                            (out2fit["aVal"],out2fit["fqVal"],out2fit["agVal"]))
            
            if edistance1 < edistance2:
                if out1fit["cost"] > objList[indiv]["cost"]:
                    # inherit father gopp array
                    out1fit["gopp"] = objList[indiv]["gopp"]
                    # Encrease this operator probability
                    out1fit["gopp"][operator] = (1 + np.random.uniform(0,1,1))*LEARNER_RATE*out1fit["gopp"][operator]
                    # normalize individual operators probabilities
                    out1fit["gopp"] = out1fit["gopp"]/np.sum(out1fit["gopp"])
                    # append to children array
                    childrenOut.append(out1fit)
                elif out1fit["cost"] == objList[indiv]["cost"]:
                    # inherit father gopp array
                    out1fit["gopp"] = objList[indiv]["gopp"]
                    # Encrease this operator probability
                    out1fit["gopp"][operator] = (1 - np.random.uniform(0,1,1))*LEARNER_RATE*out1fit["gopp"][operator]
                    # normalize individual operators probabilities
                    out1fit["gopp"] = out1fit["gopp"]/np.sum(out1fit["gopp"])
                    # append to children array
                    childrenOut.append(out1fit)
                else:
                    # Decrease this operator probability
                    objList[indiv]["gopp"][operator] = (1 - np.random.uniform(0,1,1))*LEARNER_RATE*objList[indiv]["gopp"][operator]
                    # normalize individual operators probabilities
                    objList[indiv]["gopp"] = objList[indiv]["gopp"]/np.sum(objList[indiv]["gopp"])
                    # append to children array
                    childrenOut.append(objList[indiv])
                    
            else:
                if out2fit["cost"] > objList[indiv]["cost"]:
                    # inherit father gopp array
                    out2fit["gopp"] = objList[indiv]["gopp"]
                    # Encrease this operator probability
                    out2fit["gopp"][operator] = (1 + np.random.uniform(0,1,1))*LEARNER_RATE*out2fit["gopp"][operator]
                    # normalize individual operators probabilities
                    out2fit["gopp"] = out2fit["gopp"]/np.sum(out2fit["gopp"])
                    # append to children array
                    childrenOut.append(out2fit)
                elif out2fit["cost"] == objList[indiv]["cost"]:
                    # inherit father gopp array
                    out2fit["gopp"] = objList[indiv]["gopp"]
                    # Encrease this operator probability
                    out2fit["gopp"][operator] = (1 - np.random.uniform(0,1,1))*LEARNER_RATE*out2fit["gopp"][operator]
                    # normalize individual operators probabilities
                    out2fit["gopp"] = out2fit["gopp"]/np.sum(out2fit["gopp"])
                    # append to children array
                    childrenOut.append(out2fit)
                else:
                    # Decrease this operator probability
                    objList[indiv]["gopp"][operator] = (1 - np.random.uniform(0,1,1))*LEARNER_RATE*objList[indiv]["gopp"][operator]
                    # normalize individual operators probabilities
                    objList[indiv]["gopp"] = objList[indiv]["gopp"]/np.sum(objList[indiv]["gopp"])
                    # append to children array
                    childrenOut.append(objList[indiv])
            
        elif operator == 1:
            # 1: Mating
            crossover_point = np.random.randint((d - 1))
            out1 = np.append(P1[0:(crossover_point + 1)], P2[crossover_point + 1:])
            out2 = np.append(P2[0:(crossover_point + 1)], P1[crossover_point + 1:])
            # Main Parent
            # p1fit = function(P1)
            out1fit = function(imgeo, out1, codeLen)
            out2fit = function(imgeo, out2, codeLen)
            
            edistance1 = distance.euclidean((objList[indiv]["aVal"],objList[indiv]["fqVal"],objList[indiv]["agVal"]), 
                                            (out1fit["aVal"],out1fit["fqVal"],out1fit["agVal"])) 
            edistance2 = distance.euclidean((objList[indiv]["aVal"],objList[indiv]["fqVal"],objList[indiv]["agVal"]), 
                                            (out2fit["aVal"],out2fit["fqVal"],out2fit["agVal"]))
            if edistance1 < edistance2:
                if out1fit["cost"] > objList[indiv]["cost"]:
                    # inherit father gopp array
                    out1fit["gopp"] = objList[indiv]["gopp"]
                    # Encrease this operator probability
                    out1fit["gopp"][operator] = (1 + np.random.uniform(0,1,1))*LEARNER_RATE*out1fit["gopp"][operator]
                    # normalize individual operators probabilities
                    out1fit["gopp"] = out1fit["gopp"]/np.sum(out1fit["gopp"])
                    # append to children array
                    childrenOut.append(out1fit)
                elif out1fit["cost"] == objList[indiv]["cost"]:
                    # inherit father gopp array
                    out1fit["gopp"] = objList[indiv]["gopp"]
                    # Encrease this operator probability
                    out1fit["gopp"][operator] = (1 - np.random.uniform(0,1,1))*LEARNER_RATE*out1fit["gopp"][operator]
                    # normalize individual operators probabilities
                    out1fit["gopp"] = out1fit["gopp"]/np.sum(out1fit["gopp"])
                    # append to children array
                    childrenOut.append(out1fit)
                else:
                    # Decrease this operator probability
                    objList[indiv]["gopp"][operator] = (1 - np.random.uniform(0,1,1))*LEARNER_RATE*objList[indiv]["gopp"][operator]
                    # normalize individual operators probabilities
                    objList[indiv]["gopp"] = objList[indiv]["gopp"]/np.sum(objList[indiv]["gopp"])
                    # append to children array
                    childrenOut.append(objList[indiv])
                    
            else:
                if out2fit["cost"] > objList[indiv]["cost"]:
                    # inherit father gopp array
                    out2fit["gopp"] = objList[indiv]["gopp"]
                    # Encrease this operator probability
                    out2fit["gopp"][operator] = (1 + np.random.uniform(0,1,1))*LEARNER_RATE*out2fit["gopp"][operator]
                    # normalize individual operators probabilities
                    out2fit["gopp"] = out2fit["gopp"]/np.sum(out2fit["gopp"])
                    # append to children array
                    childrenOut.append(out2fit)
                elif out2fit["cost"] == objList[indiv]["cost"]:
                    # inherit father gopp array
                    out2fit["gopp"] = objList[indiv]["gopp"]
                    # Encrease this operator probability
                    out2fit["gopp"][operator] = (1 - np.random.uniform(0,1,1))*LEARNER_RATE*out2fit["gopp"][operator]
                    # normalize individual operators probabilities
                    out2fit["gopp"] = out2fit["gopp"]/np.sum(out2fit["gopp"])
                    # append to children array
                    childrenOut.append(out2fit)
                else:
                    # Decrease this operator probability
                    objList[indiv]["gopp"][operator] = (1 - np.random.uniform(0,1,1))*LEARNER_RATE*objList[indiv]["gopp"][operator]
                    # normalize individual operators probabilities
                    objList[indiv]["gopp"] = objList[indiv]["gopp"]/np.sum(objList[indiv]["gopp"])
                    # append to children array
                    childrenOut.append(objList[indiv])
        else:
            # This is a big mistake, If out1 changes, P1 changes too
            # out1 = P1
            out1 = np.copy(P1)
            out2 = np.copy(P1)
            
            mup = np.random.uniform(0,1,d)
            # mutaPercent = 1/(population.shape[1])
            mutaPercent = mutationP
            mask = (mup <= mutaPercent).astype(np.uint8)
            mutate = np.where( mask == 1)
            mutation = np.logical_not(out1[mutate[0]]).astype(np.uint8)
            out1[mutate[0]] = mutation
            # out2 = np.copy(out1)
            mup = np.random.uniform(0,1,d)
            # mutaPercent = 1/(population.shape[1])
            mutaPercent = mutationP
            mask = (mup <= mutaPercent).astype(np.uint8)
            mutate = np.where( mask == 1)
            mutation = np.logical_not(out1[mutate[0]]).astype(np.uint8)
            out2[mutate[0]] = mutation
            
            # Main Parent
            # p1fit = function(imgeo, P1, codeLen)
            out1fit = function(imgeo, out1, codeLen)
            out2fit = function(imgeo, out2, codeLen)
            
            edistance1 = distance.euclidean((objList[indiv]["aVal"],objList[indiv]["fqVal"],objList[indiv]["agVal"]), 
                                            (out1fit["aVal"],out1fit["fqVal"],out1fit["agVal"])) 
            edistance2 = distance.euclidean((objList[indiv]["aVal"],objList[indiv]["fqVal"],objList[indiv]["agVal"]), 
                                            (out2fit["aVal"],out2fit["fqVal"],out2fit["agVal"]))
            
            if edistance1 < edistance2:
                if out1fit["cost"] > objList[indiv]["cost"]:
                    # inherit father gopp array
                    out1fit["gopp"] = objList[indiv]["gopp"]
                    # Encrease this operator probability
                    out1fit["gopp"][operator] = (1 + np.random.uniform(0,1,1))*LEARNER_RATE*out1fit["gopp"][operator]
                    # normalize individual operators probabilities
                    out1fit["gopp"] = out1fit["gopp"]/np.sum(out1fit["gopp"])
                    # append to children array
                    childrenOut.append(out1fit)
                elif out1fit["cost"] == objList[indiv]["cost"]:
                    # inherit father gopp array
                    out1fit["gopp"] = objList[indiv]["gopp"]
                    # Encrease this operator probability
                    out1fit["gopp"][operator] = (1 - np.random.uniform(0,1,1))*LEARNER_RATE*out1fit["gopp"][operator]
                    # normalize individual operators probabilities
                    out1fit["gopp"] = out1fit["gopp"]/np.sum(out1fit["gopp"])
                    # append to children array
                    childrenOut.append(out1fit)
                else:
                    # Decrease this operator probability
                    objList[indiv]["gopp"][operator] = (1 - np.random.uniform(0,1,1))*LEARNER_RATE*objList[indiv]["gopp"][operator]
                    # normalize individual operators probabilities
                    objList[indiv]["gopp"] = objList[indiv]["gopp"]/np.sum(objList[indiv]["gopp"])
                    # append to children array
                    childrenOut.append(objList[indiv])
                    
            else:
                if out2fit["cost"] > objList[indiv]["cost"]:
                    # inherit father gopp array
                    out2fit["gopp"] = objList[indiv]["gopp"]
                    # Encrease this operator probability
                    out2fit["gopp"][operator] = (1 + np.random.uniform(0,1,1))*LEARNER_RATE*out2fit["gopp"][operator]
                    # normalize individual operators probabilities
                    out2fit["gopp"] = out2fit["gopp"]/np.sum(out2fit["gopp"])
                    # append to children array
                    childrenOut.append(out2fit)
                elif out2fit["cost"] == objList[indiv]["cost"]:
                    # inherit father gopp array
                    out2fit["gopp"] = objList[indiv]["gopp"]
                    # Encrease this operator probability
                    out2fit["gopp"][operator] = (1 - np.random.uniform(0,1,1))*LEARNER_RATE*out2fit["gopp"][operator]
                    # normalize individual operators probabilities
                    out2fit["gopp"] = out2fit["gopp"]/np.sum(out2fit["gopp"])
                    # append to children array
                    childrenOut.append(out2fit)
                else:
                    # Decrease this operator probability
                    objList[indiv]["gopp"][operator] = (1 - np.random.uniform(0,1,1))*LEARNER_RATE*objList[indiv]["gopp"][operator]
                    # normalize individual operators probabilities
                    objList[indiv]["gopp"] = objList[indiv]["gopp"]/np.sum(objList[indiv]["gopp"])
                    # append to children array
                    childrenOut.append(objList[indiv])
            
    return childrenOut    
    
def childenGen_MultyM_haeaMax(population, selectedPop, gopp, fitList, function, mutationP, lnrate, minVal, maxVal, codeLen):
    LEARNER_RATE = lnrate
    
    # make room for new offspring
    childrenOut = np.zeros(population.shape).astype(np.uint8)
    # Get indexes of GA operator application probabilities is an descending order
    probOrderIndex = np.flip(np.argsort(gopp), axis =-1)
    # Descending sort of GA operator application probabilities
    opgSrt = np.flip(np.sort(gopp), axis =-1)
    # Comulative prob. ascending sort of GA operator application probabilities
    comulativeGopp = np.cumsum(opgSrt, axis =-1)
    for indiv in range(0, population.shape[0]):
        # Select genetic operator
        gopSelp = np.random.uniform(0,1,1)
        # 0: Mutation, 1: Mating
        operator = probOrderIndex[indiv][np.where(gopSelp <= comulativeGopp[indiv])[0][0]]
        # Load individuals
        P1 = population[indiv]
        P2 = selectedPop[indiv]
        if operator == 0:
            # This is a big mistake, If out1 changes, P1 changes too
            # out1 = P1
            out1 = np.copy(np.flip(P1,0))
            out2 = np.copy(P1)
            
            
            mup = np.random.uniform(0,1,population.shape[1])
            # mutaPercent = 1/(population.shape[1])
            mutaPercent = mutationP
            mask = (mup <= mutaPercent).astype(np.uint8)
            mutate = np.where( mask == 1)
            mutation = np.logical_not(out1[mutate[0]]).astype(np.uint8)
            out1[mutate[0]] = mutation
            # out2 = np.copy(out1)
            mup = np.random.uniform(0,1,population.shape[1])
            # mutaPercent = 1/(population.shape[1])
            mutaPercent = mutationP
            mask = (mup <= mutaPercent).astype(np.uint8)
            mutate = np.where( mask == 1)
            mutation = np.logical_not(out1[mutate[0]]).astype(np.uint8)
            out2[mutate[0]] = mutation
            
            # Main Parent
            p1fit = function(P1)
            out1fit = function(out1)
            out2fit = function(out2)
            
            # TODO: support more than one dimention 
            gdecoOut1 = gdeco(bitlist = out1, minVal = minVal, maxVal = maxVal, codeLen = codeLen)
            gdecoOut2 = gdeco(bitlist = out2, minVal = minVal, maxVal = maxVal, codeLen = codeLen)
            gdecoP1 = gdeco(bitlist = P1, minVal = minVal, maxVal = maxVal, codeLen = codeLen)
            
            edistance1 = distance.euclidean(gdecoP1, gdecoOut1) 
            edistance2 = distance.euclidean(gdecoP1, gdecoOut2)
            
            if edistance1 < edistance2:
                if out1fit > p1fit:
                    childrenOut[indiv] = out1
                    # Encrease this operator probability
                    gopp[indiv][operator] = (1 + np.random.uniform(0,1,1))*LEARNER_RATE*gopp[indiv][operator] 
                elif out1fit == p1fit:
                    childrenOut[indiv] = out1
                    # Decrease this operator probability
                    gopp[indiv][operator] = (1 - np.random.uniform(0,1,1))*LEARNER_RATE*gopp[indiv][operator]
                else:
                    childrenOut[indiv] = P1
                    # Decrease this operator probability
                    gopp[indiv][operator] = (1 - np.random.uniform(0,1,1))*LEARNER_RATE*gopp[indiv][operator]
            else:
                if out2fit > p1fit:
                    childrenOut[indiv] = out2
                    # Encrease this operator probability
                    gopp[indiv][operator] = (1 + np.random.uniform(0,1,1))*LEARNER_RATE*gopp[indiv][operator] 
                elif out2fit == p1fit:
                    childrenOut[indiv] = out2
                    # Decrease this operator probability
                    gopp[indiv][operator] = (1 - np.random.uniform(0,1,1))*LEARNER_RATE*gopp[indiv][operator]
                else:
                    childrenOut[indiv] = P1
                    # Decrease this operator probability
                    gopp[indiv][operator] = (1 - np.random.uniform(0,1,1))*LEARNER_RATE*gopp[indiv][operator]
            
        elif operator == 1:
            # 1: Mating
            crossover_point = np.random.randint((population.shape[1] - 1))
            out1 = np.append(P1[0:(crossover_point + 1)], P2[crossover_point + 1:])
            out2 = np.append(P2[0:(crossover_point + 1)], P1[crossover_point + 1:])
            # Main Parent
            p1fit = function(P1)
            out1fit = function(out1)
            out2fit = function(out2)
            # TODO: support more than one dimention 
            gdecoOut1 = gdeco(bitlist = out1, minVal = minVal, maxVal = maxVal, codeLen = codeLen)
            gdecoOut2 = gdeco(bitlist = out2, minVal = minVal, maxVal = maxVal, codeLen = codeLen)
            gdecoP1 = gdeco(bitlist = P1, minVal = minVal, maxVal = maxVal, codeLen = codeLen)
            
            edistance1 = distance.euclidean(gdecoP1, gdecoOut1) 
            edistance2 = distance.euclidean(gdecoP1, gdecoOut2)
            if edistance1 < edistance2:
                if out1fit > p1fit:
                    childrenOut[indiv] = out1
                    # Encrease this operator probability
                    gopp[indiv][operator] = (1 + np.random.uniform(0,1,1))*LEARNER_RATE*gopp[indiv][operator] 
                elif out1fit == p1fit:
                    childrenOut[indiv] = out1
                    # Decrease this operator probability
                    gopp[indiv][operator] = (1 - np.random.uniform(0,1,1))*LEARNER_RATE*gopp[indiv][operator]
                else:
                    childrenOut[indiv] = P1
                    # Decrease this operator probability
                    gopp[indiv][operator] = (1 - np.random.uniform(0,1,1))*LEARNER_RATE*gopp[indiv][operator]
            else:
                if out2fit > p1fit:
                    childrenOut[indiv] = out2
                    # Encrease this operator probability
                    gopp[indiv][operator] = (1 + np.random.uniform(0,1,1))*LEARNER_RATE*gopp[indiv][operator] 
                elif out2fit == p1fit:
                    childrenOut[indiv] = out2
                    # Decrease this operator probability
                    gopp[indiv][operator] = (1 - np.random.uniform(0,1,1))*LEARNER_RATE*gopp[indiv][operator]
                else:
                    childrenOut[indiv] = P1
                    # Decrease this operator probability
                    gopp[indiv][operator] = (1 - np.random.uniform(0,1,1))*LEARNER_RATE*gopp[indiv][operator]
        # 2: Single Mutation
        else:
            # This is a big mistake, If out1 changes, P1 changes too
            # out1 = P1
            out1 = np.copy(P1)
            out2 = np.copy(P1)
            
            
            mup = np.random.uniform(0,1,population.shape[1])
            # mutaPercent = 1/(population.shape[1])
            mutaPercent = mutationP
            mask = (mup <= mutaPercent).astype(np.uint8)
            mutate = np.where( mask == 1)
            mutation = np.logical_not(out1[mutate[0]]).astype(np.uint8)
            out1[mutate[0]] = mutation
            # out2 = np.copy(out1)
            mup = np.random.uniform(0,1,population.shape[1])
            # mutaPercent = 1/(population.shape[1])
            mutaPercent = mutationP
            mask = (mup <= mutaPercent).astype(np.uint8)
            mutate = np.where( mask == 1)
            mutation = np.logical_not(out1[mutate[0]]).astype(np.uint8)
            out2[mutate[0]] = mutation
            # Main Parent
            p1fit = function(P1)
            out1fit = function(out1)
            out2fit = function(out2)
            
            # TODO: support more than one dimention 
            gdecoOut1 = gdeco(bitlist = out1, minVal = minVal, maxVal = maxVal, codeLen = codeLen)
            gdecoOut2 = gdeco(bitlist = out2, minVal = minVal, maxVal = maxVal, codeLen = codeLen)
            gdecoP1 = gdeco(bitlist = P1, minVal = minVal, maxVal = maxVal, codeLen = codeLen)
            
            edistance1 = distance.euclidean(gdecoP1, gdecoOut1) 
            edistance2 = distance.euclidean(gdecoP1, gdecoOut2)
            
            if edistance1 < edistance2:
                if out1fit > p1fit:
                    childrenOut[indiv] = out1
                    # Encrease this operator probability
                    gopp[indiv][operator] = (1 + np.random.uniform(0,1,1))*LEARNER_RATE*gopp[indiv][operator] 
                elif out1fit == p1fit:
                    childrenOut[indiv] = out1
                    # Decrease this operator probability
                    gopp[indiv][operator] = (1 - np.random.uniform(0,1,1))*LEARNER_RATE*gopp[indiv][operator]
                else:
                    childrenOut[indiv] = P1
                    # Decrease this operator probability
                    gopp[indiv][operator] = (1 - np.random.uniform(0,1,1))*LEARNER_RATE*gopp[indiv][operator]
            else:
                if out2fit > p1fit:
                    childrenOut[indiv] = out2
                    # Encrease this operator probability
                    gopp[indiv][operator] = (1 + np.random.uniform(0,1,1))*LEARNER_RATE*gopp[indiv][operator] 
                elif out2fit == p1fit:
                    childrenOut[indiv] = out2
                    # Decrease this operator probability
                    gopp[indiv][operator] = (1 - np.random.uniform(0,1,1))*LEARNER_RATE*gopp[indiv][operator]
                else:
                    childrenOut[indiv] = P1
                    # Decrease this operator probability
                    gopp[indiv][operator] = (1 - np.random.uniform(0,1,1))*LEARNER_RATE*gopp[indiv][operator]
            
        # normalize individual operators probabilities
        gopp[indiv] = gopp[indiv]/np.sum(gopp[indiv])
                    
    return   childrenOut,gopp

def diffEvolution(population, selectedPop, fitList, fitVal, function, matingPercent, d):
    
    
    
    # make room for new offspring
    childrenOut = np.zeros((population.shape[0],d))
    
    for indiv in range(0, population.shape[0]):
        # Load individuals
        P1 = selectedPop[indiv*3]
        P2 = selectedPop[indiv*3 + 1]
        P3 = selectedPop[indiv*3 + 2]
        R = np.random.randint(0, d, 1)[0]
        for k in range(d):
            matingp = np.random.uniform(0,1,1)
            if matingp <= matingPercent or k == R:
                childrenOut[indiv][k] = P1[k] + 1*(P2[k] - P3[k])
            else:
                childrenOut[indiv][k] = fitVal[indiv][k]
                
    return childrenOut



            
        
          
               

def haeaMin(population, selectedPop, gopp, fitList, function, mutationP, lnrate):
    
    LEARNER_RATE = lnrate
    
    # make room for new offspring
    childrenOut = np.zeros(population.shape).astype(np.uint8)
    # Get indexes of GA operator application probabilities is an descending order
    probOrderIndex = np.flip(np.argsort(gopp), axis =-1)
    # Descending sort of GA operator application probabilities
    opgSrt = np.flip(np.sort(gopp), axis =-1)
    # Comulative prob. ascending sort of GA operator application probabilities
    comulativeGopp = np.cumsum(opgSrt, axis =-1)
    for indiv in range(0, population.shape[0]):
        # Select genetic operator
        gopSelp = np.random.uniform(0,1,1)
        # 0: Mutation, 1: Mating
        operator = probOrderIndex[indiv][np.where(gopSelp <= comulativeGopp[indiv])[0][0]]
        # Load individuals
        P1 = population[indiv]
        P2 = selectedPop[indiv]
        # 1: Mating
        if operator == 1:
            
            crossover_point = np.random.randint((population.shape[1] - 1))
            out1 = np.append(P1[0:(crossover_point + 1)], P2[crossover_point + 1:])
            out2 = np.append(P2[0:(crossover_point + 1)], P1[crossover_point + 1:])
            p1fit = function(P1)
            out1fit = function(out1)
            out2fit = function(out2)
            
            if out1fit < p1fit or out2fit < p1fit:
                if out1fit < out2fit:
                    childrenOut[indiv] = out1
                else:
                    childrenOut[indiv] = out2
                # Encrease this operator probability
                gopp[indiv][operator] = (1 + np.random.uniform(0,1,1))*LEARNER_RATE*gopp[indiv][operator] 
            elif out1fit == p1fit or out2fit == p1fit:
                if out1fit == p1fit:
                    childrenOut[indiv] = out1
                else:
                    childrenOut[indiv] = out2
                # Decrease this operator probability
                gopp[indiv][operator] = (1 - np.random.uniform(0,1,1))*LEARNER_RATE*gopp[indiv][operator]
            else:
                childrenOut[indiv] = P1
                # Decrease this operator probability
                gopp[indiv][operator] = (1 - np.random.uniform(0,1,1))*LEARNER_RATE*gopp[indiv][operator]
        # 0: Mutation
        else:
            # This is a big mistake, If out1 changes, P1 changes too
            # out1 = P1
            out1 = np.copy(P1)
            
            mup = np.random.uniform(0,1,population.shape[1])
            # mutaPercent = 1/(population.shape[1])
            mutaPercent = mutationP
            mask = (mup <= mutaPercent).astype(np.uint8)
            mutate = np.where( mask == 1)
            mutation = np.logical_not(out1[mutate[0]]).astype(np.uint8)
            out1[mutate[0]] = mutation
            p1fit = function(P1)
            out1fit = function(out1)
            if out1fit < p1fit:
                childrenOut[indiv] = out1
                # Encrease this operator probability
                gopp[indiv][operator] = (1 + np.random.uniform(0,1,1))*LEARNER_RATE*gopp[indiv][operator] 
            elif out1fit == p1fit:
                childrenOut[indiv] = out1
                # Decrease this operator probability
                gopp[indiv][operator] = (1 - np.random.uniform(0,1,1))*LEARNER_RATE*gopp[indiv][operator] 
            else:
                childrenOut[indiv] = P1
                # Decrease this operator probability
                gopp[indiv][operator] = (1 - np.random.uniform(0,1,1))*LEARNER_RATE*gopp[indiv][operator] 
                    
        
        # normalize individual operators probabilities
        gopp[indiv] = gopp[indiv]/np.sum(gopp[indiv])
                    
    return   childrenOut,gopp 


def haeaMax(population, selectedPop, gopp, fitList, function, mutationP, lnrate):
    
    LEARNER_RATE = lnrate
    
    # make room for new offspring
    childrenOut = np.zeros(population.shape).astype(np.uint8)
    # Get indexes of GA operator application probabilities is an descending order
    probOrderIndex = np.flip(np.argsort(gopp), axis =-1)
    # Descending sort of GA operator application probabilities
    opgSrt = np.flip(np.sort(gopp), axis =-1)
    # Comulative prob. ascending sort of GA operator application probabilities
    comulativeGopp = np.cumsum(opgSrt, axis =-1)
    for indiv in range(0, population.shape[0]):
        # Select genetic operator
        gopSelp = np.random.uniform(0,1,1)
        # 0: Mutation, 1: Mating
        operator = probOrderIndex[indiv][np.where(gopSelp <= comulativeGopp[indiv])[0][0]]
        # Load individuals
        P1 = population[indiv]
        P2 = selectedPop[indiv]
        # 1: Mating
        if operator == 1:
            
            crossover_point = np.random.randint((population.shape[1] - 1))
            out1 = np.append(P1[0:(crossover_point + 1)], P2[crossover_point + 1:])
            out2 = np.append(P2[0:(crossover_point + 1)], P1[crossover_point + 1:])
            p1fit = function(P1)
            out1fit = function(out1)
            out2fit = function(out2)
            if out1fit > p1fit or out2fit > p1fit:
                if out1fit > out2fit:
                    childrenOut[indiv] = out1
                else:
                    childrenOut[indiv] = out2
                # Encrease this operator probability
                gopp[indiv][operator] = (1 + np.random.uniform(0,1,1))*LEARNER_RATE*gopp[indiv][operator] 
            elif out1fit == p1fit or out2fit == p1fit:
                if out1fit == p1fit:
                    childrenOut[indiv] = out1
                else:
                    childrenOut[indiv] = out2
                # Decrease this operator probability
                gopp[indiv][operator] = (1 - np.random.uniform(0,1,1))*LEARNER_RATE*gopp[indiv][operator]
            else:
                childrenOut[indiv] = P1
                # Decrease this operator probability
                gopp[indiv][operator] = (1 - np.random.uniform(0,1,1))*LEARNER_RATE*gopp[indiv][operator]
        # 0: Mutation
        else:
            # This is a big mistake, If out1 changes, P1 changes too
            # out1 = P1
            out1 = np.copy(P1)
            
            mup = np.random.uniform(0,1,population.shape[1])
            # mutaPercent = 1/(population.shape[1])
            mutaPercent = mutationP
            mask = (mup <= mutaPercent).astype(np.uint8)
            mutate = np.where( mask == 1)
            mutation = np.logical_not(out1[mutate[0]]).astype(np.uint8)
            out1[mutate[0]] = mutation
            p1fit = function(P1)
            out1fit = function(out1)
            if out1fit > p1fit:
                childrenOut[indiv] = out1
                # Encrease this operator probability
                gopp[indiv][operator] = (1 + np.random.uniform(0,1,1))*LEARNER_RATE*gopp[indiv][operator] 
            elif out1fit == p1fit:
                childrenOut[indiv] = out1
                # Decrease this operator probability
                gopp[indiv][operator] = (1 - np.random.uniform(0,1,1))*LEARNER_RATE*gopp[indiv][operator] 
            else:
                childrenOut[indiv] = P1
                # Decrease this operator probability
                gopp[indiv][operator] = (1 - np.random.uniform(0,1,1))*LEARNER_RATE*gopp[indiv][operator] 
                    
        
        # normalize individual operators probabilities
        gopp[indiv] = gopp[indiv]/np.sum(gopp[indiv])
                    
    return   childrenOut,gopp 

# Evolution Strategies
# Reconbine two arrow vectors
def es_reconbination(v1,v2):
    d = v1.shape[0]
    ho = []
    # two parents
    p = 2
    N = np.concatenate((v1, v2), axis=0)
    N = N.reshape(2,d)
    for i in range(0,d):
        # new alpha set for each dimention.
        # 
        alpha = np.random.uniform(0,1,p)
        alphaNormal = alpha/np.sum(alpha)
        h = np.dot(alphaNormal, N)
        ho.append(h[i])
    return np.asarray(ho)

def es_reconbinationMtx(v1,v2):
    # v1 and v2 are squared matrixes
    dOrg = v1.shape[0]
    v1f = v1.flatten()
    v2f = v2.flatten()
    d = v1f.shape[0]
    ho = []
    # two parents
    p = 2
    N = np.concatenate((v1f, v2f), axis=0)
    N = N.reshape(2,d)
    for i in range(0,d):
        # new alpha set for each dimention.
        # 
        alpha = np.random.uniform(0,1,p)
        alphaNormal = alpha/np.sum(alpha)
        h = np.dot(alphaNormal, N)
        ho.append(h[i])
    return np.asarray(ho).reshape((dOrg,dOrg))

def es_mutation(y,s):
    d = y.shape[0]
    mu = np.zeros(d)
    # y and s equal size
    # sigma mutation
    sRedus = np.copy(s)
    sRedus = sRedus/10
    sChange = np.random.normal(mu, sRedus, d)
    s = s + sChange
    # parameter mutation
    yhange = np.random.normal(mu, s, d)
    y = y + yhange
    
    return y,s
# perform reconbination and mutation based on passed pair set
def es_ChildrenGeneration(N,ns,tournamentPairs, minVal, maxVal):
    children_N = np.zeros(N.shape)
    children_ns = np.zeros(ns.shape)
    mainIndex = 0
    for pair in tournamentPairs:
        nRec = es_reconbination(N[pair[0]],N[pair[1]])
        nsRec = es_reconbination(ns[pair[0]],ns[pair[1]])
        children_N[mainIndex],children_ns[mainIndex] = es_mutation(nRec,nsRec)
        # Resulting values are kept into minVal, maxVal interval
        children_N[mainIndex] = np.clip(children_N[mainIndex], minVal, maxVal)
        mainIndex = mainIndex + 1
    return children_N,children_ns

# perform reconbination and mutation based on passed pair set. With rotation matrix
def es_ChildrenGenerationMtx(N,ns,M,ms,tournamentPairs,minVal,maxVal):
    children_N = np.zeros(N.shape)
    children_ns = np.zeros(ns.shape)
    children_M = np.zeros(M.shape)
    children_ms = np.zeros(ms.shape)
    mainIndex = 0
    for pair in tournamentPairs:
        nRec = es_reconbination(N[pair[0]],N[pair[1]])
        nsRec = es_reconbination(ns[pair[0]],ns[pair[1]])
        mRec = es_reconbinationMtx(M[pair[0]],M[pair[1]])
        msRec = es_reconbinationMtx(ms[pair[0]],ms[pair[1]])
        children_N[mainIndex], children_ns[mainIndex], children_M[mainIndex], children_ms[mainIndex] = es_mutationRotMx(nRec,
                                                                                                                        nsRec,
                                                                                                                        mRec,
                                                                                                                        msRec)
        
        # Resulting values are kept into minVal, maxVal interval
        children_N[mainIndex] = np.clip(children_N[mainIndex], minVal, maxVal)
        mainIndex = mainIndex + 1
    return children_N,children_ns,children_M,children_ms

# Next generation. Select best fro Childre and parents
def es_childreAndParentsSelection(N,ns,fitList,children_N,children_ns,children_fitList):
    colony_N = np.concatenate((N,children_N), axis=0)
    colony_s = np.concatenate((ns,children_ns), axis=0)
    colony_fit = np.concatenate((fitList,children_fitList), axis=0)
    bstOrder = np.argsort(colony_fit)
    newPop = np.zeros(N.shape)
    newPop_s = np.zeros(ns.shape)
   
    for i in range(0,N.shape[0]):
        newPop[i] = colony_N[bstOrder[i]]
        newPop_s[i] = colony_s[bstOrder[i]]
        
        
    return newPop,newPop_s
        
# Next generation. Select best fro Childre and parents
def es_childreAndParentsSelectionMtx(N,ns,M,ms,fitList,children_N,children_ns,children_M,children_ms,children_fitList):
    colony_N = np.concatenate((N,children_N), axis=0)
    colony_s = np.concatenate((ns,children_ns), axis=0)
    colony_M = np.concatenate((M,children_M), axis=0)
    colony_ms = np.concatenate((ms,children_ms), axis=0)
    colony_fit = np.concatenate((fitList,children_fitList), axis=0)
    bstOrder = np.argsort(colony_fit)
    newPop = np.zeros(N.shape)
    newPop_s = np.zeros(ns.shape)
    newPop_M = np.zeros(M.shape)
    newPop_ms = np.zeros(ms.shape)
   
    for i in range(0,N.shape[0]):
        newPop[i] = colony_N[bstOrder[i]]
        newPop_s[i] = colony_s[bstOrder[i]]
        newPop_M[i] = colony_M[bstOrder[i]]
        newPop_ms[i] = colony_ms[bstOrder[i]]
        
        
    return newPop,newPop_s,newPop_M,newPop_ms


def es_mutationRotMx(y,s,m,ms):
    d = y.shape[0]
    s_mu = np.zeros(d)
    ms_mu =np.zeros((d, d))
    # y and s equal size
    # m must be d x d
    # ms must be d x d
    # sigma mutation
    sRedus = np.copy(s)
    sRedus = sRedus/10
    sChange = np.random.normal(s_mu, sRedus, d)
    s = s + sChange
    
    # ratation matrix sigma mutation
    msRedus = np.copy(ms)
    msRedus = msRedus/10
    msChange = np.random.normal(ms_mu, msRedus, (d,d))
    ms = ms + msChange
    
    # ratation matrix mutation
    #Matrix mutation
    mhange = np.random.normal(ms_mu, ms, (d,d))
    m = m + mhange
    
    # parameter mutation
    yhange = np.random.normal(s_mu, s, d)
    ydotm = np.dot(m,np.transpose(yhange))
    y = y + np.transpose(ydotm)
    
    
    
    
    return y,s,m,ms