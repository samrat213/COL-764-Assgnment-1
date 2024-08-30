from A1.dict_cons import SimpleTokenizer, BPETokenizer, WordPieceTokenizer
from collections import Counter,defaultdict
import json
import pickle

class InvertedIndex():
    def __init__(self, tokenizer=0) -> None:
        if tokenizer==0:
            self.encoder = SimpleTokenizer()
        elif tokenizer==1:
            self.encoder = BPETokenizer()
        elif tokenizer==2:
            self.encoder = WordPieceTokenizer()
        
        self.inverted_index = defaultdict(lambda: [])
        self.doc_to_index={}
        # self.doc_count = Counter()
    
    def numerize_docid(self,doc_count):
        self.doc_to_index = {item[0]:idx for idx, item in enumerate(sorted(doc_count.items(), key=lambda item: item[1],reverse=True))}

    def replace_docid(self):
        for token, doc_list in self.inverted_index.items():
            for i in range(len(doc_list)):
                doc_list[i]= self.doc_to_index[doc_list[i]]

    def construct_index(self, path='test'):
        doc_count = Counter()
        with open(path,'r', encoding='utf-8') as file:
            for doc in file:
                doc = json.loads(doc.strip())
                text = doc['title'] +' ' + doc['abstract']
                text = self.encoder.remove_non_ascii(text)
                tokens = set(self.encoder.encode_text(text))
                no_tokens = len(tokens)
                doc_count[doc['doc_id']]+=no_tokens
                for token in tokens:
                    self.inverted_index[token].append(doc['doc_id'])
        
        self.numerize_docid(doc_count)
        self.replace_docid()
        
        self.encoder.save(self.inverted_index.keys, './indexfile.dict')
        self.save_binary({'doc_to_index':self.doc_to_index, 'inverted_index':self.inverted_index})
        
    
    def save_binary(self, data, path='indexfile.idx'):
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def read_binary(self, path='indexfile.idx'):
        with open(path, 'rb') as f:
            loaded_dict = pickle.load(f)
        return loaded_dict