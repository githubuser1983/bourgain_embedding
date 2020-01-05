#!/usr/bin/python
import math
import numpy as np


def dist(x,y):
    return math.sqrt(sum([(x[i]-y[i])**2 for i in range(max(len(x),len(y)))]))


class BourgainEmbedding:
    def __init__(self,dist,fast=False):
        self.dist = dist
        self.fast = fast

    def fit(self,X,verbose=False):
        np.random.seed(123)
        X_emb = []
        R = range(len(X))
        n = len(X)
        self.n = n
        if self.fast == False:
            k = int(math.ceil(math.log(n)/math.log(2)-1))
            T = int(math.ceil(math.log(n)))
        else:
            k = int(math.ceil(math.log(math.log(n))/math.log(2)-1))
            T = int(math.ceil(math.log(math.log(n))))
        self.dict_of_XS = {}
        for i in range(0,k+1):
            for t in range(T):
                S = np.random.choice(R,2**i)
                self.dict_of_XS[(i,t)] = [ X[s] for s in S ]
                for j in R:
                    x = X[j]
                    d = min([ self.dist(x,xs) for xs in self.dict_of_XS[(i,t)]])
                    if i==0 and t==0:
                        X_emb.append([d])
                    else:
                        X_emb[j].append(d)
                    if verbose:
                        print(n,k,T,i,t,j,len(X_emb[j]),(k+1)*T)
        return X_emb
        

    def predict(self,X,verbose=False):
        X_emb = []
        R = range(len(X))
        n = self.n
        if self.fast == False:
            k = int(math.ceil(math.log(n)/math.log(2)-1))
            T = int(math.ceil(math.log(n)))
        else:
            k = int(math.ceil(math.log(math.log(n))/math.log(2)-1))
            T = int(math.ceil(math.log(math.log(n))))       
        for i in range(0,k+1):
            for t in range(T):
                XS = self.dict_of_XS[(i,t)]
                for j in R:
                    x = X[j]
                    d = min([ self.dist(x,xs) for xs in XS])
                    if i==0 and t==0:
                        X_emb.append([d])
                    else:
                        X_emb[j].append(d)
                    if verbose:
                        print(n,k,T,i,t,j,len(X_emb[j]),(k+1)*T)
        return X_emb
        

if __name__ == "__main__":
    X = [ [(i+j)*(i+j+1)/2+j for j in range(0,10+1)] for i in range(0,50)]
    be = BourgainEmbedding(dist)
    X_emb = be.fit(X)
    XX_emb = be.predict(X)
    if X_emb != XX_emb:
        print( X_emb)
        print(XX_emb)
    l = []
    for x in range(len(X)):
        for y in range(len(X)):
            if x != y:
                d1 = dist(X[x],X[y])
                d2 = dist(X_emb[x],X_emb[y])
                l.append(d2*1.0/(d1*1.0))  
                #print x,y,d1,d2
    print( "distortion = %s" % (max(l)/min(l)) )
    print( "upper bound for distortion = %s " % math.log(len(X)))
    # print "\n".join([str(x) for x in X_emb])
    #print "\n".join([",".join([str(a) for a in X_emb[i]]) for i in range(len(X))])
