from __future__ import division
import os
from collections import Counter
import numpy as np

def get_frequency_vector(dataset):
    for dir_name, subdir_list, file_list in os.walk(dataset):
        print(dir_name)
        cnt=Counter()
        for sd in subdir_list:
            cnt+=get_frequency_vector(dataset+"/"+sd)
        text=""
        for fname in file_list:
            f=open(dataset+"/"+fname,"r")
            text+=f.read()
            f.close()
        cnt+=Counter(text.split())
        return cnt

class DatasetIterator:
    # iterates over document frequency vectors and return a detaset consisting of words for each document
    def __init__(self,dataset,label):
        self.__dict__.update(locals())
        del self.self
        self.i = 0
        self.n = 5 #dataset.shape[0] <- number of document datasets to generate. Set to low number for testing. Set to dataset.shape[0] for full run.
        
    def __iter__(self):
        return self
    
    def next(self):
        if(self.i>=self.n):
            raise StopIteration()
        document=np.zeros((self.dataset[self.i,:].sum(),self.dataset.shape[1]))
        words_accum=0
        for word_id in range(self.dataset[self.i,:].shape[0]):
            document[words_accum:words_accum+self.dataset[self.i,word_id],word_id]=1
            words_accum+=self.dataset[self.i,word_id]
        perm = np.random.permutation(document.shape[0])
        self.i+=1
        return document[perm,:],np.array([self.label[self.i]]*document.shape[0])
 
class News20:
    def __init__(self,path,
                 vector_size=20000, #how many most popular words to use
                 which_set="train", 
                 excluded_words=["the", "of"],
                 most_common_idx=None):
        self.__dict__.update(locals())
        del self.self
        
        self.data=np.loadtxt(path+"/"+which_set+".data")
        self.label=np.loadtxt(path+"/"+which_set+".label")
        self.map=np.loadtxt(path+"/"+which_set+".map",dtype=str)
        self.vocabulary=np.loadtxt(path+"/"+"vocabulary.txt",dtype=str)
        
        # determine frequency of words in the whole set
        full_frequencies=np.zeros(self.vocabulary.shape[0])
        for j in range(self.data.shape[0]):
            full_frequencies[self.data[j,1]-1]+=self.data[j,2]
            
        # extract the indicies of [vector_size] most common words
        if(most_common_idx==None):
            most_common_idx=np.argsort(full_frequencies)[::-1][:vector_size]
            self.most_common_idx=most_common_idx
            print "20 most popular words:"
            print self.vocabulary[most_common_idx[:20]]
        for word in excluded_words:
            most_common_idx=most_common_idx[most_common_idx!=np.where(self.vocabulary==word)[0]]
        print "20 most popular words after exclusion:"
        print self.vocabulary[most_common_idx[:20]]      

        # create matrix of word frequencies per document
        self.documents=np.zeros((np.max(self.data[:,0]),self.vector_size))
        hash_idx=dict(zip(self.most_common_idx,range(self.most_common_idx.shape[0])))
        for j in range(self.data.shape[0]):
            try:
                self.documents[self.data[j,0]-1,hash_idx[self.data[j,1]]]=self.data[j,2]
            except:
                continue
                
    def getDocumentWordFrequencies(self):
        return self.documents,self.label
    
    def getNormalizedDocumentWordFrequencies(self):
        return self.documents/self.documents.sum(axis=1).reshape(-1,1),self.label    
    
    def sampleFromDocument(self,doc_id,num_samples):
        norm_doc=self.documents[doc_id,:]/float(self.documents[doc_id,:].sum())
        output = np.zeros((num_samples,norm_doc.shape[0]))
        for j in range(num_samples):
            output[j,(np.cumsum(norm_doc)<np.random.rand()).sum()-1]=1
        return output
    
    def getDocumentIterator(self):
        return DatasetIterator(self.documents,self.label)
        
# example usage
if __name__=="__main__":
    train=News20("/dos/study/research/mlrg/replicated_softmax/data/20news-bydate_preprocessed")
    x,y=train.getNormalizedDocumentWordFrequencies()
    sample=train.sampleFromDocument(10000,4)
    it = train.getDocumentIterator()
    documents=[d for d in it]    
    print ""
