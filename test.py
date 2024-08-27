import json
import os
import re
from collections import Counter
PATTERN = '''[ ,.:;"']+'''
class SimpleTokenizer():
    def __init__(self) -> None:
        
        pass

    def remove_non_ascii(self, text):
        return text.encode('ascii',errors='ignore').decode()

    def read_file(self, path):
        corpus =[]
        with open(path,'r', encoding='utf-8') as file:
            for doc in file:
                doc = json.loads(doc.strip())
                text = doc['title'] +'. ' + doc['abstract']
                corpus.append(self.remove_non_ascii(text))
        return corpus

    def tokenize_corpus(self, corpus):
        all_tokens=[]
        for doc in corpus:
            tokens = re.split(PATTERN, doc)
            tokens = [token for token in tokens if token]
            all_tokens.extend(tokens)
        
        return all_tokens
    
    def save(self, data , path):
        with open(path,'w') as file:
            file.write('\n'.join(data))

    def encode(self, path):
        corpus = read_file(path)
        tokens = self.tokenize_corpus(corpus)
        self.save(tokens, './output.dict')

# Question 1
# obj =SimpleTokenizer()
# obj.encode('cord19-trec_covid-docs')
class BPETokenizer(SimpleTokenizer):
    def __init__(self, path='test', full=0) -> None:



        # self.word_freq = self.calc_freq(tokens)
        # self.splitted_word = self.split_words(self.word_freq)
        # self.vocabulary = self.get_alphabets()
        self.merges = {}
        self.vocabulary = {}
        self.splitted_word = {}
        self.word_freq = {}
        # del self.tokens
        # self.training()

    def read_file(self, path, full=0):
        corpus =[]
        with open(path,'r', encoding='utf-8') as file:
            lines = file.readlines()
            
            if full==0:
                lines = lines[int(len(lines)/2):]

            for doc in file:
                doc = json.loads(doc.strip())
                text = doc['title'] +'. ' + doc['abstract']
                corpus.append(self.remove_non_ascii(text))
        return corpus
    
    def add_end_of_word(self, tokens):
        for i in range(len(tokens)):
            tokens[i] = tokens[i]+ "Ġ"
        return tokens
    
    def calc_freq(self, tokens):
        token_count = Counter()
        for token in tokens:
            token_count[token]+=1
        
        return token_count
    
    def split_words(self, word_frequency):
        splitted_word = {}
        for word in word_frequency.keys():
            splitted_word[word] = [ch for ch in word]
        
        return splitted_word
    
    def get_alphabets(self):
        # vocabulary = []
        # for word in self.word_freq.keys():
        #     for letter in word:
        #         if letter not in vocabulary:
        #             vocabulary.append(letter)

        vocabulary =set()
        for _, split in self.splitted_word.items():
            vocabulary.union(split)

        vocabulary = list(vocabulary)
        vocabulary.sort()
        return vocabulary
    
    def calculate_pair_freq(self, word_frequency):
        pair_count = Counter()

        for word, freq in word_frequency.items():
            charaters = self.splitted_word[word]
            n = len(charaters)
            if n==1:
                continue

            for i in range(n -1):
                pair_count[(charaters[i], charaters[i+1])] += freq
            
        return pair_count
          
    def get_best_pair(self, pair_freq):
        max_val = max(pair_freq.values())
        ret = list(filter(lambda x: pair_freq[x] == max_val, pair_freq))[0]
        return ret, max_val
    
    def merges_best_pair(self, best_pair):

        for word,split in self.splitted_word.items():
            # n=len(split)
            if len(split)==1:
                continue
            
            i=0
            while i< len(split) - 1:
                if split[i] == best_pair[0] and split[i+1] == best_pair[1]:
                    split = split[:i] + [best_pair[0]+best_pair[1]] + split[i+2:]
                
                i+=1
            
            self.splitted_word[word] = split
        
        return 1
    
    def first_itr(self, path):
        corpus = self.read_file(path)
        tokens = self.tokenize_corpus(corpus)
        tokens = self.add_end_of_word(tokens)
        self.word_freq = self.calc_freq(tokens)
        self.splitted_word = self.split_words(self.word_freq)
        self.vocabulary = self.get_alphabets()
        
    def train(self, vocab_size=500):
        n= len(self.vocabulary)
        for i in range(n,vocab_size):
            pair_frequencies = self.calculate_pair_freq()
            if len(pair_frequencies) ==0:
                break
            best_pair, count = self.get_best_pair(pair_frequencies)
            self.merges[best_pair] = best_pair[0]+best_pair[1]
            self.vocabulary.append(self.merges[best_pair])
            # print(best_pair, count, len(self.vocabulary))
            self.splitted_word = self.merges_best_pair(best_pair)
        
        return
    
    def run_test(self):
        tokens = self.get_all_tokens(self.path, self.full, testing=1)
        tokens = self.add_end_of_word(tokens)
        splitted_words = [[ch for ch in token]  for token in tokens if token]
        # print(splitted_words)

        for pair, merged in self.merges.items():
            for i, split in enumerate(splitted_words):
                j=0
                while j < len(split)-1:
                    if split[j] == pair[0] and split[j+1] == pair[1]:
                        # print(split[:j], [merged], split[j+2:])
                        # print(split, pair)
                        split = split[:j] + [merged] + split[j+2:]
                    j+=1
                splitted_words[i]=split
        
        return sum(splitted_words, [])
    

doc_path = './cord19-trec_covid-docs'
simple_obj = SimpleTokenizer()

obj = BPETokenizer(doc_path)
