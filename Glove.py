import numpy as np
import os
from keras.preprocessing.text import Tokenizer
import pickle as pk
class Glove():

    def __init__(self,glove_path,embedding_dim=100):

        if(glove_path is None):
            raise ValueError("Please provide glove path... Exiting")
        self.glove_path=glove_path
        self.embedding_dim=embedding_dim
    
    def prepare_word_embeddings(self):
        embeddings_index = {}
        self.words=[]
        try:
            f = open(os.path.join(self.glove_path, 'glove.6B.'+str(self.embedding_dim)+'d.txt'),encoding='utf-8')
            for line in f:
                values = line.split()
                word = values[0].lower()
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
                self.words.append(word)
            self.embeddings=embeddings_index
            f.close()
        except Exception as e:
            print("Failed Loading Glove Data %s"%(e))

    def load_word_embeddings(self,path):
        try:
            with open(os.path.join(path),'rb') as embeddings_path:
                self.embeddings=pk.load(embeddings_path)
        except Exception as e:
            print("Failed to load embedding index. \n Error: %s"%(e))

    def save_dictionary(self,path):
        None
    
    def load_dictionary(self,path):
        None
    
    def get_embedding_matrix(self):
        tokenizer=Tokenizer(50000)
        tokenizer.fit_on_texts(self.words)
        word_index=tokenizer.word_index
        word_count=(len(word_index))+1
        embedding_matrix=np.zeros((word_count,self.embedding_dim))
        i=0
        for word,index in word_index.items():
            try:
                embedding_matrix[index,:]=self.embeddings[word]
            except:
                i+=1
                pass
        print("Missing Words %s"%(i))
        return embedding_matrix
        
    def clean_mem(self):
        del self.embeddings
        del self.words

    def get_dictionary_size(self):
        None



    
    
