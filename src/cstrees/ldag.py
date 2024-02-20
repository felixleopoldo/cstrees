from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq

import pandas as pd
import networkx as nx
import numpy as np
import pp


class LDAG(nx.DiGraph):
    
    def plot_graphviz(self, prog="dot", args=""):            
        agraph = nx.nx_agraph.to_agraph(self)
        agraph.layout(prog=prog, args=args)
        return agraph    

# Functions needed for generating LDAG representation from a dataframe representation of the staging
# of a CStree

def _convertToNumeric(df, alarmdf):
    npdf = df.to_numpy()
    vars = list(df.columns)
    n = len(df)
    for v in vars:
        j = vars.index(v)
        states = list(alarmdf[v].drop_duplicates().to_numpy())
        for i in range(n):
            npdf[i,j] = states.index(alarmdf[v].iloc[i])
    numdf = pd.DataFrame(npdf)
    return numdf


def _nodemask(v, w, df):
    dashmask = (df[w] == '-')
    mask = dashmask & (df[v] != '-')
    return mask

def _getCSI(v, w, df):
    dfs = df[_nodemask(v, w, df)]
    n = len(dfs)
    vars = list(dfs.columns)
    d = len(vars)
    CSIs = []
    
    for i in range(n):
        A = []
        B = []
        Bcontexts = []
        for j in range(d):
            if dfs.iloc[i,j] == '*':
                A += [vars[j]]
            elif dfs.iloc[i,j] != '-':
                B += [vars[j]]
                Bcontexts += [dfs.iloc[i,j]]
        CSIs += [[w, A, B, Bcontexts]]
    
    return CSIs

def _collectCSIs(v, df):
    n = len(df)
    vars = list(df.columns)
    vidx = vars.index(v)
    prevvar = vars[vidx - 1]
    CSIs = _getCSI(prevvar,v, df)
    
    return CSIs

def _collectParents(v, df):
    CSIs = _collectCSIs(v,df)
    m = len(CSIs)
    parents = []
    for i in range(m):
        parents += CSIs[i][2]
        
    parents = list(dict.fromkeys(parents))
    
    return parents

def _collectVertexLabels(v, df):
    CSIs = _collectCSIs(v, df)
    m = len(CSIs)
    
    parents = _collectParents(v, df)
    p = len(parents)
    padict = dict.fromkeys(parents)
    
    for i in range(m):
        CSIs[i].pop(0)
        CSIs[i].pop(0)
        
    labels = dict.fromkeys(parents)
    edgeLabels = {}
        
    for k in parents:
        padict[k] = []
        labels[k] = []
        for i in range(m):
            if CSIs[i][0].count(k) == 0:
                padict[k] += [CSIs[i]]
                labels[k] += [CSIs[i][1]]
        edgeLabels[(k, v)] = labels[k]
                
        kvanish = [ len(x[0]) for x in padict[k]]
        if len(kvanish) != 0:
            if len(list(dict.fromkeys(kvanish))) != 1:
                print('Warning: different sized vanishing sets')
    
    edges = list(edgeLabels.keys())
    
    for e in edges:
        if edgeLabels[e] == []:
            del edgeLabels[e]
    
    return edgeLabels

def _collectLabels(df):
    num_nodes = len(list(df.columns))
    labels = {}
    for i in range(num_nodes):
        labels.update(_collectVertexLabels(i, df))
    
    return labels

def _updateEdges(dic, varorder):
    edges = list(dic.keys())
    for i in range(len(edges)):
        edges[i] = (varorder[edges[i][0]], varorder[edges[i][1]])
    
    return edges

def _getDAGmap(df):
    nodes = list(df.columns)
    num_nodes = len(nodes)
    adjmat = np.zeros([num_nodes, num_nodes],int)
    
    for v in nodes:
        v_parents = _collectParents(v, df)
        for w in v_parents:
            adjmat[w,v] = 1
    adjmat = np.matrix(adjmat)
    
    return adjmat
