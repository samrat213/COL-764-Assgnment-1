import numpy as np
from collections import Counter, defaultdict
import json
import pickle
import sys
from .dict_cons import SimpleTokenizer, BPETokenizer, WordPieceTokenizer
import time

class Retrival():
    def __init__(self, encoder, index_file_path='./indexfile.idx') -> None:
        self.encoder = encoder
        # # self.IDF = {}
        self.inverted_index = {}
        self.doc_to_index={}
        self.index_to_doc={}
        self.number_doc = 0
        self.normalization_term = []
        self.load_parameters(index_file_path)
        return
    
    def read_indexfile(self, path= './indexfile.idx'):
        with open(path, 'rb') as f:
            loaded_dict = pickle.load(f)
        return loaded_dict
    
    def load_parameters(self, path='./indexfile.idx'):
        ret = self.read_indexfile(path)
        self.inverted_index = ret['inverted_index']
        self.doc_to_index = ret['doc_to_index']
        self.IDF = ret['IDF']
        self.index_to_doc=ret['index_to_doc']
        self.number_doc = ret['number_docs']
        self.normalization_term = ret['normalization_value']
        if type(self.encoder)!=SimpleTokenizer:
            self.encoder.merges = ret['merges']
            self.encoder.vocabulary = ret['vocabulary']

    def process_doc(self, doc):
        text = doc['title'] +' ' + doc['abstract']
        text = self.encoder.remove_non_ascii(text)
        return self.encoder.encode_text(text)
    
    def process_query(self, doc):
        # doc = json.loads(doc.strip())
        text = doc['title'] +' ' + doc['description'] +' ' + doc['narrative']
        text = self.encoder.remove_non_ascii(text)
        tokens = self.encoder.encode_text(text)
        
        token_count = Counter()
        for token in tokens:
            token_count[token]+=1
            
        return set(tokens), token_count
    
    def get_TF(self,tf):
        if tf ==0:
            return 0
        return 1+ np.log2(tf)
    
    def log_normalized_tf(self, tf):
        # for i in range(len(tf)):
        #     if tf[i]!=0:
        #         tf[i] = 1+ np.log2(tf[i])

        # zero_mask = (tf == 0)
        # tf = 1+ np.where(zero_mask, 0, np.log(tf))
        
        # tf = np.where(tf > 0, 1+np.log(tf), 1)

        # tf[tf!=0] = 1+ np.log2(tf[tf!=0])
        # tf[tf==0]=1
        tf[tf!=0] = np.log2(tf[tf!=0])
        tf = 1+ tf
        return tf
    
    def train_tf_idf(self, path = './test'):
        inverted_index = defaultdict(lambda: set())
        doc_count = Counter()
        with open(path,'r', encoding='utf-8') as file:
            collection= file.readlines()
            self.number_doc =len(collection)
            tf = defaultdict(lambda: np.zeros(self.number_doc))
            # tf = np.zeros()
            doc_index= 0
            for doc in collection:
                doc = json.loads(doc.strip())
                tokens = self.process_doc(doc)
                
                no_tokens = len(set(tokens))
                doc_count[doc['doc_id']]+=no_tokens
                for token in tokens:
                    # inverted_index[token].append(doc['doc_id'])
                    # tf[token][doc_index]+=1
                    tf[token][doc_index]+=1
                    inverted_index[token].add(doc['doc_id'])

                    
                
                self.doc_to_index[doc['doc_id']] = doc_index
                self.index_to_doc[doc_index] = doc['doc_id']
                doc_index+=1
                # print(doc_index)
        print('-----Documents processed-----')
        IDF = {token:np.log2(self.number_doc/len(set(docs))) for token, docs in inverted_index.items()}
        print('-----IDF calculated-----')
        self.tf_idf = {token : self.log_normalized_tf(tf[token])*IDF[token] for token in tf.keys()}
        print('-----TF-IDF calculated-----')
        normalization_term = np.zeros(self.number_doc)
        for token in self.tf_idf.keys():
            normalization_term += np.square(self.tf_idf[token])
        self.normalization_term = np.sqrt(normalization_term)
        print('-----Normalization Term calculated-----')

    def tf_idf_query(self, token_freq):
        query_tf_idf = {}
        normalization_term = 0.0
        for token, freq in token_freq.items():
            query_tf_idf[token] = (1+ np.log2(freq))*self.IDF[token]
            normalization_term+=np.square(query_tf_idf[token])
        
        return query_tf_idf, np.sqrt(normalization_term)
    
    def calculate_similarities(self, query):
        tokens, token_freq = self.process_query(query)
        query_tf_idf, query_normalization_term = self.tf_idf_query(token_freq)
        
        similarity = []
        all_tokens = set(self.tf_idf.keys())
        tokens = all_tokens.intersection(tokens)
        for doc_index in range(self.number_doc):
            sim = 0.0
            
            for token in tokens:
                # print(token, doc_index)
                sim += query_tf_idf[token] * self.tf_idf[token][doc_index]
            sim = sim/(query_normalization_term * self.normalization_term[doc_index])
            print(doc_index, sim)
            similarity.append((self.index_to_doc[doc_index], sim))
            
        similarity = sorted(similarity, key=lambda item: item[1], reverse=True)
        return similarity
    
    def run_query(self, query):
        tokens, token_freq = self.process_query(query)
        query_tf_idf, query_normalization_term = self.tf_idf_query(token_freq)
        
        similarity = []
        # all_tokens = set(self.tf_idf.keys())
        # tokens = all_tokens.intersection(tokens)

        for doc_index in range(self.number_doc):
            sim = 0.0
            
            for token in tokens:
                # print(token, doc_index)
                if doc_index in self.inverted_index[token].keys():
                    sim += query_tf_idf[token] * self.inverted_index[token][doc_index]
            sim = sim/(query_normalization_term * self.normalization_term[doc_index])
            print(doc_index, sim)
            similarity.append([query['query_id'], 0, self.index_to_doc[doc_index], sim])
            
        similarity = sorted(similarity, key=lambda item: item[1], reverse=True)
        return similarity[:100]
    
    def save(self, data , path):
        with open(path,'w') as file:
            for line in data:
                file.write('\t'.join(line))
                file.write('\n')

    def retrieve(self, path = './cord19-trec_covid-queries', output_path = './resultfile'):
        ret = [['qid', 'iteration', 'docid', 'relevancy']]
        n=0
        # start
        with open(path,'r', encoding='utf-8') as file:
            for query in file:
                doc = json.loads(query.strip())
                result = self.run_query(doc)

        self.save(data=result, path=output_path)
        return result
    
    def read_qrel(self, path = 'cord19-trec_covid-qrels'):
        self.result_gt = defaultdict(set)
        with open(path,'r', encoding='utf-8') as file:
            for rel in file:
                doc = json.loads(rel.strip())
                if doc['relevance']==0:
                    self.result_gt[doc['query_id']]['irrelevent'].add(self.doc_to_index(doc['doc_id']))
                else:
                    self.result_gt[doc['query_id']]['relevent'].add(self.doc_to_index(doc['doc_id']))

    def parse_result(self,result):
        ret = defaultdict(set)

        for entry in result:
            ret[entry[0]].add(entry[2])

        return ret
    
# F = 2RP/(R+P)
    def score(self, relevent, irrelevent, retrieved, k=10):
        relevent = set(list(relevent)[:k])
        irrelevent = set(list(irrelevent)[:k])
        retrieved = set(list(retrieved)[:k])
        relevent_and_retrieved = len(retrieved.intersection(relevent))
        precision = relevent_and_retrieved/len(retrieved)
        recall = relevent_and_retrieved/len(relevent)
        F = 2*recall*precision/(recall+precision)
        return F

    def calculate_score(self, result):
        scores={10:0, 20:0, 50:0, 100:0}
        for query_id in result.keys():
            relevent = self.result_gt[query_id]['relevent']
            irrelevent = self.result_gt[query_id]['irrrelevent']
            retrieved = result[query_id]
            scores[10] += self.score(relevent, irrelevent, retrieved, 10)
            scores[20] += self.score(relevent, irrelevent, retrieved, 20)
            scores[50] += self.score(relevent, irrelevent, retrieved, 50)
            scores[100] += self.score(relevent, irrelevent, retrieved, 100)
        
        n=len(result.keys())

        score = {k:v/n for k,v in score.items()}

        return score



if __name__=="__main__":
    query_file = sys.argv[1]
    result_file = sys.argv[2]
    index_file = sys.argv[3]
    dict_file = sys.argv[4]
    # print(sys.argv)
    start_time = time.time()
    obj = Retrival(index_file_path=index_file)
    ret = obj.retrieve(path=query_file, output_path=result_file)
    print('Efficiency: ', (time.time()-start_time)/100)

    obj.read_qrel('./cord19-trec_covid-qrels')
    our_result = obj.parse_result(ret)
    print('F1@: ', obj.calculate_score(our_result))