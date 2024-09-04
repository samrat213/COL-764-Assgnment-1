from dict_cons import SimpleTokenizer, BPETokenizer, WordPieceTokenizer
from collections import Counter,defaultdict
import json
import pickle
import sys
import pickle
import numpy as np
import time

class InvertedIndex():
    def __init__(self) -> None:
        self.inverted_index = defaultdict(lambda: [])
        self.doc_to_index={}
        self.index_to_doc={}
        self.number_docs = 0
        self.IDF = {}

        # self.doc_count = Counter()
    
    def numerize_docid(self,doc_count):
        # self.doc_to_index = {item[0]:idx for idx, item in enumerate(sorted(doc_count.items(), key=lambda item: item[1],reverse=True))}
        for idx, item in enumerate(sorted(doc_count.items(), key=lambda item: item[1],reverse=True)):
            self.doc_to_index[item[0]] = idx
            self.index_to_doc[idx] = item[0]
        
        self.number_docs = len(self.doc_to_index)

    def get_idf(self,df):
        return np.log2(1 + self.number_docs/df)
    
    def get_TF(self,tf):
        if tf ==0:
            return 0
        return 1+ np.log2(tf)
        
    def populate_variables(self):
        self.normalization_value =np.zeros(self.number_docs)

        for token in self.inverted_index.keys():
            doc_list = self.inverted_index[token]
            IDF = self.get_idf(len(doc_list))
            self.IDF[token] = IDF
            posting_dict = {}
            for doc_id, token_freq in doc_list:
                doc_index = self.doc_to_index[doc_id]
                tf = self.get_TF(token_freq)
                self.normalization_value[doc_index] += np.square(tf*IDF)
                posting_dict[doc_index] = tf

            self.inverted_index[token] = posting_dict
        
        self.normalization_value = np.sqrt(self.normalization_value)

    def construct_index(self, encoder, path='test', output_name = 'indexfile'):
        doc_count = Counter()
        i=0
        doc_ids = set()
        with open(path,'r', encoding='utf-8') as file:
            for doc in file:
                if i%10000==0:
                    print(i)
                i+=1
                doc = json.loads(doc.strip())
                if doc['doc_id'] in doc_ids:
                    continue
                doc_ids.add(doc['doc_id'])
                text = doc['title'] +' ' + doc['abstract']
                text = encoder.remove_non_ascii(text)
                tokens = encoder.encode_text(text)
                token_freq = Counter(tokens)
                tokens = set(tokens)
                doc_count[doc['doc_id']]+=len(token_freq.keys())
                for token in tokens:
                    self.inverted_index[token].append((doc['doc_id'],token_freq[token]))
        
        self.numerize_docid(doc_count)
        self.populate_variables()
        
        encoder.save(self.inverted_index.keys(), f'./{output_name}.dict')
        data = {
                'doc_to_index': self.doc_to_index, 
                'inverted_index': dict(self.inverted_index),
                'index_to_doc': self.index_to_doc,
                'number_docs': self.number_docs,
                'IDF': self.IDF,
                'normalization_value': self.normalization_value
             }
        data['tokenizer']=0
        
        if type(encoder)!=SimpleTokenizer:
            if type(encoder)!=BPETokenizer:
                data['tokenizer']=1
                data['merges'] = encoder.merges

            else:
                data['tokenizer']=2
                data['vocabulary'] = encoder.vocabulary

            # data['merges'] = encoder.merges
            # data['vocabulary'] = encoder.vocabulary
        self.save_binary(data,path = f'./{output_name}.idx')
         
    def save_binary(self, data, path='indexfile.idx'):
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def read_binary(self, path='indexfile.idx'):
        with open(path, 'rb') as f:
            loaded_dict = pickle.load(f)
        return loaded_dict



# if tokenizer_choice=='0':
#     encoder = SimpleTokenizer()
#     # encoder.encode(path)
#     obj.construct_index(encoder=encoder,
#                         path = path,
#                         output_name=indexFile_name)
def read_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# obj = InvertedIndex()
# # encoder = WordPieceTokenizer()

# time_taken = {0:0}
# encoder = BPETokenizer()
# print("inverted index creation using BPE")
# for i in [99,199,299,399,499,599,699,799,899,999,1099,1199,1299,1399,1499,1599,1699,1799,1899,1999]:
#     start_time = time.time()

#     # print(i)
#     encoder = BPETokenizer()
#     obj = InvertedIndex()


#     encoder.merges = read_pickle(f'./bpe/BPE_{i}.pkl')
#     obj.construct_index(encoder, 'cord19-trec_covid-docs', 
#                         output_name=f'./inverted_index/BPE_{i}'
#                         )
#     print('----------------------',i, time.time()-start_time)
#     time_taken[i+1]=time.time()-start_time

#     del encoder

#     del obj
# with open('./inverted_Progress_BPE.pkl', 'wb') as f:
#     pickle.dump(time_taken, f)


time_taken = {0:0}
print("inverted index creation using WPE")
for i in [99,199,299,399,499,599,699,799,899,999,1099,1199,1299,1399,1499,1599,1699,1799,1899,1999]:
    start_time = time.time()

    # print(i)
    obj = InvertedIndex()
    encoder = WordPieceTokenizer()

    encoder.vocabulary = set(read_pickle(f'./wpe/WPE_{i}.pkl')['vocab'])

    # encoder.merges = read_pickle(f'./bpe/BPE_{i}.pkl')
    obj.construct_index(encoder, 'cord19-trec_covid-docs', 
                        output_name=f'./inverted_index/WPE_{i}'
                        )
    print('----------------------',i, time.time()-start_time)
    time_taken[i+1]=time.time()-start_time
    del encoder
    del obj

with open('./inverted_Progress_WPE.pkl', 'wb') as f:
    pickle.dump(time_taken, f)