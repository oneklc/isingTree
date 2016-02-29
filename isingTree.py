
# coding: utf-8

# In[1]:

'''
Created on Feb 25 2016

@author: klc

Test file in: resources/extra-hard-problem.txt

'''


from __future__ import division
import os
import logging
import re
import random
import numpy as np
import logging as logging
from scipy.sparse import csr_matrix
import pdb


# In[ ]:

class IsingSpin():

    def __init__(self):
        self.isingTree = {'name':'', 'numberOfSpins':0, 'numberOfWeights':0, 'nodeWeights':[],'sparseEdgeWeights':[], 'edgeRow':[],'edgeCol':[], 'edgeWeight':[] }
        self.spinConfiguration = {'0':'','1':'','2':'','3':''}
       
    def parseData(self, line):
        '''
        parses the dataline of the input file
        '''
        
        data = line.strip().split(' ')  
        if len(data) != 3:
            logging.error('The provided data doesn\'t seem to contain the expected number of columns/fields.')
            return {}

        if (data[0] == data[1]):
            self.isingTree['nodeWeights'][int(data[0])] = int(data[2])       
        else:
            self.isingTree['edgeRow'].append(int(data[0]))
            self.isingTree['edgeCol'].append(int(data[1]))
            self.isingTree['edgeWeight'].append(int(data[2]))
                  
        return True

    def parseHeader(self, line):
        '''
        parses the problem header information
        '''
        header = line.strip().split(' ')
        
        if len(header) != 4:
            logging.error(
                'The provided header doesn\'t seem to contain the expected number of columns/fields.')
            return {}
        
        logging.debug("header is: " + line )

        self.isingTree['name'] = header[1]
        spins = int(header[2])
        self.isingTree['numberOfSpins'] = spins
        self.isingTree['numberOfWeights'] = int(header[3])
        # pre initilize with 
        self.isingTree['nodeWeights'] = [0 for x in range(spins)]
        return True



    
    def parse(self, analysis_file):
        '''
        parses an analysis file.
        '''

        # initiate variables
        lineCounter = 0

        try:

            f = open(analysis_file, 'r')

            # read the records into the intermediate analysis_files
            skipped_row = 0
            for line in f:
                lineCounter += 1

                # process non data lines
                if line.startswith('c') or line.startswith('C'):
                    skipped_row += 1
                    logging.debug("skipping, comment on line: %s", lineCounter)
                    continue
                    
                if line.startswith('p') or line.startswith('P'):
                    logging.debug("Processing problem header on line: %s", lineCounter)
                    self.parseHeader(line)
                    continue

                    
                parsedLine = self.parseData(line)
                #fail over
                if not parsedLine:
                    skipped_row += 1
                    logging.debug("skipping, failed to parse line: %s", lineCounter)
                    continue

   
            f.close()
    

        except IOError as e:
            logging.warning('IO Error: %s', e)
            logging.warning(
                "NOTE: Load failed. Check logs to verify what was actually loaded. ")
            f.close()

        #Create sparse tree 
        ew = self.isingTree['edgeWeight']
        er = self.isingTree['edgeRow']
        ec = self.isingTree['edgeCol']
        n = self.isingTree['numberOfSpins']
        
        self.isingTree['sparseEdgeWeights'] = csr_matrix((ew ,(er,ec)),shape=(n,n),dtype=np.int8)

        
        return {'skipped': skipped_row,'linesRead': lineCounter, 'error': 'none'}

                                                                                         
                                                                                         

    def analyzeGroundState(self):
        '''
        find ground energy and spin configuration
        '''                                                                         
         #nominate rood node:
        random.seed()
        rootNode = random.randint(0,self.isingTree['numberOfSpins'] - 1)
        logging.info("Root node selected is: %s", rootNode)
        minEnergy = self.groundState(rootNode)
        
         #make spin string better formated    
        spinStr = ''
        for i in range(0, len(self.spinConfiguration)):
            spinStr += self.spinConfiguration[str(i)]
            
        return [minEnergy, spinStr]
        
        

    def groundState(self, parent,child=None, parentSpin=None):
        '''
            parent node id
            child node id (optional)
            parentSpin (optional)
        '''
         #for readability
        
        #1st time through
        if child is None:
            child = parent
            parentSpin=1 #doesn't matter (muliptiply by 0)
            
        edgeWeights = self.isingTree['sparseEdgeWeights']
        nodeWeights = self.isingTree['nodeWeights']
        nodeRowEdges = edgeWeights.getrow(child).toarray()
        nodeColEdges = edgeWeights.transpose().getrow(child).toarray()

        logging.debug("RowEdges %s", nodeRowEdges)
        logging.debug("Column edges %s", nodeColEdges)
        logging.debug("parent = %s, child = %s, parent spin = %s", parent, child, parentSpin)


        childSpin = [-1,1]
        spinGroundState = {'-1':0, '1':0}

        for spin in childSpin:
            columnId = 0
            groundEnergy = 0
            isLeafNode = True
            rowId=0    

            for childRowNode in nodeRowEdges[0]:
                if (childRowNode != 0) and (parent != columnId):
                    isLeafNode = False
                    groundEnergy += self.groundState(child,columnId,spin)
                    logging.debug("Parent is %s with spin %s, child is %s, their edge weight is %s, current groundEnergy= %s", parent, spin, columnId,childRowNode, groundEnergy)

                columnId += 1

            for childColNode in nodeColEdges[0]:
                if (childColNode != 0) and (parent != rowId):
                    isLeafNode = False
                    groundEnergy += self.groundState(child,rowId,spin)
                    logging.debug("Parent is %s with spin %s, child is %s, their edge weight is %s, current groundEnergy= %s", parent,spin, rowId,childColNode, groundEnergy)
                    
                rowId += 1

            if isLeafNode:
                spinGroundState[str(spin)] = self.leafGroundState(parent,child, parentSpin)
            else:
                spinGroundState[str(spin)] = groundEnergy + nodeWeights[child]*spin + self.getEdgeWeight(parent,child)*parentSpin*spin
        
        # which spin configuration is minimal?
        if  spinGroundState['-1'] < spinGroundState['1']:
            self.spinConfiguration[str(child)] = "-"
            return spinGroundState['-1']
        else:
            self.spinConfiguration[str(child)] = "+"
            return spinGroundState['1']

          

    def getEdgeWeight(self, nodeA, nodeB):        
        '''get edge weight, check both possible locations 
            (shouldn't have to do this if matrix is always in upper row form)
        '''
            
        edgeWeight = 0
        edgeWeights = self.isingTree['sparseEdgeWeights']

        if edgeWeights[nodeA,nodeB] != 0:
            return edgeWeights[nodeA,nodeB] 
        else:
            return edgeWeights[nodeB,nodeA]
        
    def leafGroundState(self, parent,leaf, parentSpin):
        '''
             leaf = leaf node id
             parentSpin = spin of parent
        '''
        get_ipython().magic('pdb.settrace ()')
        nodeWeights = self.isingTree['nodeWeights']
        edgeWeights = self.isingTree['sparseEdgeWeights']
        leafWeight = nodeWeights[leaf]

        parentEdgeWeight = self.getEdgeWeight(leaf,parent)
        if (parentEdgeWeight == 0):
            logging.info("Leaf = parent? %s %s", leaf, parent)
        #calculate both possibilites and pick the lower. 
        leafGroundNeg =  -1*leafWeight + -1*parentSpin*parentEdgeWeight
        leafGroundPos = leafWeight + parentSpin*parentEdgeWeight
        
        if  (leafGroundNeg < leafGroundPos):
            self.spinConfiguration[str(leaf)] = "-"
            return leafGroundNeg
        
        self.spinConfiguration[str(leaf)] = "+"
        return leafGroundPos
        


# In[ ]:

##############################################
######  TESTS             ####################
##############################################

import unittest


class IslingSpinAlgoTests(unittest.TestCase):

    def test_Import(self):

        ising = IsingSpin()
        testFile = './resources/extra-hard-problem.txt'

        summary = ising.parse(testFile)

        logging.debug("name is: " +  ising.isingTree['name'])
        self.assertTrue(ising.isingTree['name'] == "test01")
        
        logging.debug("Number of spins is: %s" ,  ising.isingTree['numberOfSpins'])
        self.assertTrue(ising.isingTree['numberOfSpins'] == 4)

        logging.debug("Number of weights is: %s",  ising.isingTree['numberOfWeights'])
        self.assertTrue(ising.isingTree['numberOfWeights'] == 6)
        
        ## verify spareseEdgeWeights is correct
        row = [0,1,1]
        col = [1,2,3]
        data = [1,1,1]
        edge_weights = csr_matrix((data,(row,col)),(4,4))
        
        logging.debug("edge Column %s", ising.isingTree['edgeCol'])
        self.assertTrue(col == ising.isingTree['edgeCol'])

        logging.debug("edge Row %s", ising.isingTree['edgeRow'])
        self.assertTrue(row == ising.isingTree['edgeRow'])
        
        logging.debug("edge weights %s", ising.isingTree['edgeWeight'])
        self.assertTrue(data == ising.isingTree['edgeWeight'])
        
        
        #getting errors with comparing the sparse matrix. Will compare the componets instead        
        #self.assertFalse((edge_weights != ising.isingTree['sparseEdgeWeights']) )
        
        # verify nodeWeights is correct
        logging.debug("node weights %s", ising.isingTree['nodeWeights'])
        self.assertTrue(ising.isingTree['nodeWeights'] == [-1,-1,-1,0])

        
        

    def test_Analysis(self):

        ising = IsingSpin()
        testFile = './resources/extra-hard-problem.txt'
        summary = ising.parse(testFile)
        results = ising.analyzeGroundState()

        logging.info("results are %s, %s", results[0],results[1])
        self.assertTrue(results[0] == -4)
        self.assertTrue(results[1] == "+-++")

        
    def test_Analysis2(self):

        ising = IsingSpin()
        testFile = './resources/extra-realy-hard-problem.txt'
        summary = ising.parse(testFile)
        results = ising.analyzeGroundState()

        logging.info("results are %s, %s", results[0],results[1])
        self.assertTrue(results[0] == -8)
        #not necessairly only optimal solution!
        #self.assertTrue(results[1] == "+++-+++") 

    def test_oneNodeTest(self):
        ising = IsingSpin()
        testFile = './resources/oneNodeTest.txt'
        summary = ising.parse(testFile)
        results = ising.analyzeGroundState()

        logging.info("results are %s, %s", results[0],results[1])
        self.assertTrue(results[0] == -1)
        self.assertTrue(results[1] == "+")
        


# In[ ]:

logging.getLogger().setLevel(logging.INFO)
tests = IslingSpinAlgoTests()
tests.test_Import()
tests.test_Analysis()
tests.test_Analysis2()
tests.test_oneNodeTest()



# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



