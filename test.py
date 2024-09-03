import json
import os
import re
from collections import Counter,defaultdict
import pickle
import time
import matplotlib.pyplot as plt

# from A1.dict_cons import SimpleTokenizer 
PATTERN = '''[ ,.:;"']+'''
start_time = time.time()

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
                text = doc['title'] +' ' + doc['abstract']
                corpus.append(self.remove_non_ascii(text))
        return corpus

    def seperate_words(self, corpus):
        all_tokens=[]
        for doc in corpus:
            tokens = re.split(PATTERN, doc)
            tokens = [token for token in tokens if token]
            all_tokens.extend(tokens)
        
        return all_tokens
    
    def save(self, data , path='./output.dict'):
        with open(path,'w') as file:
            file.write('\n'.join(data))

    def encode(self, path='./output.dict'):
        corpus = self.read_file(path)
        tokens = self.seperate_words(corpus)
        self.save(list(set(tokens)), path)
    
    def encode_text(self,text):
        tokens = self.seperate_words([text])
        return tokens
    
class BPETokenizer(SimpleTokenizer):
    def __init__(self) -> None:



        # self.word_freq = self.calc_freq(tokens)
        # self.splitted_word = self.split_words(self.word_freq)
        # self.vocabulary = self.get_alphabets()
        self.merges = {}
        self.vocabulary = []
        self.splitted_word = {}
        self.word_freq = {}
        # self.start_time = time.time()
        # del self.tokens
        # self.training()

    def read_file(self, path, portion=0.5):
        all_words =[]
        doc_ids =set()
        with open(path,'r', encoding='utf-8') as file:
            lines = file.readlines()
            

            lines = lines[int(len(lines)*portion):]

            for line in lines:
                doc = json.loads(line.strip())
                
                if doc['doc_id'] in doc_ids:
                    continue
                doc_ids.add(doc['doc_id'])
                text = doc['title'] +' ' + doc['abstract']
                all_words.extend(self.seperate_words(self.remove_non_ascii(text)))
                # corpus.append(self.remove_non_ascii(text))
        return all_words
    
    def seperate_words(self, doc):
        tokens = re.split(PATTERN, doc)
        tokens = [token+"Ġ" for token in tokens if token]
        return tokens
    
    def add_end_of_word(self, tokens):
        for i in range(len(tokens)):
            tokens[i] = tokens[i]+ "Ġ"
        return tokens
    
    def calc_freq(self, tokens):
        return Counter(tokens)
        # token_count = Counter()
        # for token in tokens:
        #     token_count[token]+=1
        
        # return token_count
    
    def split_words(self, words):
        splitted_word = {}
        vocabulary =set()

        for word in words:
            splitted_word[word] = list(word)
            vocabulary.update(splitted_word[word])

        vocabulary = list(vocabulary)
        vocabulary.sort()
        
        return splitted_word, vocabulary
       
    def split_words_query(self, words):
        splitted_word = []

        for word in words:
            splitted_word.append([ch for ch in word])

        return splitted_word

    def get_alphabets(self):
        # vocabulary = []
        # for word in self.word_freq.keys():
        #     for letter in word:
        #         if letter not in vocabulary:
        #             vocabulary.append(letter)

        vocabulary =set()
        for word in self.splitted_word.keys():
            vocabulary.update(self.splitted_word[word])

        vocabulary = list(vocabulary)
        vocabulary.sort()
        return vocabulary
    
    def calculate_pair_freq(self):
        pair_count = Counter()
        
        # c = Counter(zip(seq, seq[1:]))
        for word in self.word_freq.keys():
        # for word, charaters in self.splitted_word.items():
            charaters = self.splitted_word[word]
            # pair_count.update(zip(charaters, charaters[1:])*self.word_freq[word])
            n = len(charaters)

            for i in range(n -1):
                pair_count[(charaters[i], charaters[i+1])] += self.word_freq[word]
            
        return pair_count
          
    def get_best_pair(self, pair_freq):
        v = list(pair_freq.values())
        k = list(pair_freq.keys())
        return k[v.index(max(v))]
        max_val = max(pair_freq.values())
        ret = list(filter(lambda x: pair_freq[x] == max_val, pair_freq))[0]
        return ret, max_val
    
    def merges_best_pair(self, best_pair):

        for word in self.splitted_word.keys():
            # n=len(split)
            split = self.splitted_word[word]
            if len(split)==1:
                continue
            
            i=0
            while i< len(split) - 1:
                if split[i] == best_pair[0] and split[i+1] == best_pair[1]:
                    # split = split[:i] + [best_pair[0]+best_pair[1]] + split[i+2:]
                    split[i] =best_pair[0]+best_pair[1]
                    split.pop(i+1)
                
                i+=1
            
            # self.splitted_word[word] = split
        
        return 1
    
    def first_itr(self, path):
        # corpus = self.read_file(path)
        # words = self.seperate_words(corpus)
        # words = self.add_end_of_word(words)
        words = self.read_file(path)

        self.word_freq = self.calc_freq(words)
        self.splitted_word, self.vocabulary = self.split_words(self.word_freq.keys())
        # self.vocabulary = self.get_alphabets()
        
    def train(self, merges=100):
        # points = [time.time()-self.start_time]
        for i in range(merges):
            pair_frequencies = self.calculate_pair_freq()
            if len(pair_frequencies) ==0:
                break
            # best_pair, count = self.get_best_pair(pair_frequencies)
            best_pair = self.get_best_pair(pair_frequencies)
            self.merges[best_pair] = best_pair[0]+best_pair[1]
            self.vocabulary.append(self.merges[best_pair])
            self.merges_best_pair(best_pair)
            # print(best_pair,i)
            print(i)
            # points.append(time.time()-self.start_time)
        
        self.save(self.vocabulary, './output.dict')
        # del self.splitted_word
        # del self.word_freq
        # # del self.vocabulary
        # plt.plot(points)
        # plt.xlabel('#Merges')
        # plt.ylabel('Time(sec)')
        # plt.title('BPE')
        # plt.savefig('./BPE_progress.png')
        return
    
    def encode_text(self, text):
        words = self.seperate_words(text)
        words = self.add_end_of_word(words)
        splitted_words = self.split_words_query(list(set(words)))
        n=len(splitted_words)
        for pair in self.merges.keys():
            for i in range(n):
                split= splitted_words[i]
                j=0
                while j < len(split)-1:
                    if split[j] == pair[0] and split[j+1] == pair[1]:
                        split = split[:j] + [self.merges[pair]] + split[j+2:]
                    j+=1
                splitted_words[i]=split
        
        tokens =  sum(splitted_words, [])
        return tokens
 

class WordPieceTokenizer(SimpleTokenizer):
    def __init__(self) -> None:
        self.merges = {}
        self.vocabulary = set()
        self.splitted_words = {}
        self.word_freq = {}
        # words = self.get_all_words(path,full)
        self.start_time = time.time()
        # self.word_freq = self.calc_freq(words)
        # self.splitted_words = self.split_words(words)
        # self.vocabulary = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
        # self.merges={}
        # del words
        # self.training()

    def read_file(self, path, portion=0.5):
        all_words =[]
        doc_ids =set()
        with open(path,'r', encoding='utf-8') as file:
            lines = file.readlines()
            

            lines = lines[int(len(lines)*portion):]

            for line in lines:
                doc = json.loads(line.strip())
                
                if doc['doc_id'] in doc_ids:
                    continue
                doc_ids.add(doc['doc_id'])
                text = doc['title'] +' ' + doc['abstract']
                all_words.extend(self.seperate_words(self.remove_non_ascii(text)))
                # corpus.append(self.remove_non_ascii(text))
        return all_words
    
    def calc_freq(self, tokens):
        return Counter(tokens)
    
    def seperate_words(self, doc):
        tokens = re.split(PATTERN, doc)
        tokens = [token for token in tokens if token]
        return tokens
    
    def split_words(self, tokens):
        splitted_word = {}
        vocabulary = set()
        for word in tokens:
            splitted_word[word] = ["$$"+ch for ch in word]
            splitted_word[word][0] = splitted_word[word][0][2:]
            vocabulary.union(splitted_word[word])

        return splitted_word, vocabulary
    
    def get_vocabulary(self):
        vocabulary = []

        # for word in self.word_freq.keys():
        #     for letter in word:
        #         if letter not in vocabulary:
        #             vocabulary.append(letter)
        vocabulary =set()
        for word in self.splitted_words.keys():
            vocabulary.union(self.splitted_words[word])

        vocabulary = list(vocabulary)
        vocabulary.sort()
        return vocabulary
    
    def get_best_pair(self, pair_freq):
        v = list(pair_freq.values())
        k = list(pair_freq.keys())
        return k[v.index(max(v))]
    
    #score(A,B) = (freq(AB)*len(vocab))/ (freq(A)*freq(B))
    def calculate_pair_score(self):
        letter_count = Counter()
        pair_count = Counter()

        for word in self.word_freq.keys():
            charaters = self.splitted_words[word]
            n=len(charaters)
            freq = self.word_freq[word]
            if n==1:
                letter_count[charaters[0]]+=freq
                continue

            for i in range(n-1):
                pair = (charaters[i], charaters[i+1])
                pair_count[pair] += freq
                letter_count[charaters[i]] +=freq

            letter_count[charaters[-1]] += freq

        score ={}
        for pair in pair_count.keys():
            score[pair] = pair_count[pair] /(letter_count[pair[0]] * letter_count[pair[1]])
        
        return score

    def merges_best_pair(self, best_pair):

        for word in self.splitted_words.keys():
            split = self.splitted_words[word]
            # n=len(split)
            if len(split)==1:
                continue
            
            i=0
            while i< len(split) - 1:
                if split[i] == best_pair[0] and split[i+1] == best_pair[1]:
                    merge = best_pair[0] + best_pair[1][2:]
                    split = split[:i] + [merge] + split[i+2:]
                
                i+=1
            
            self.splitted_words[word] = split
            # print(split)
        
        return 1
        
    def first_itr(self, path):
        # corpus = self.read_file(path)
        # words = self.seperate_words(corpus)
        # words = self.add_end_of_word(words)
        words = self.read_file(path)

        self.word_freq = self.calc_freq(words)
        self.splitted_words, self.vocabulary = self.split_words(self.word_freq.keys())
        # self.vocabulary = self.get_alphabets()

    def train(self, k=500):
        points = [time.time()-self.start_time]
        for i in range(k):
            score = self.calculate_pair_score()
            if len(score) ==0:
                break
            best_pair = self.get_best_pair(score)

            self.merges[best_pair] = best_pair[0] + best_pair[1][2:]
            self.merges_best_pair(best_pair)
            self.vocabulary.add(self.merges[best_pair])
            # print(merges)
            points.append(time.time()-self.start_time)

            print(i)
                
        self.save(self.vocabulary, './output.dict')
        # del self.splitted_word
        # del self.word_freq
        # del self.vocabulary
        plt.plot(points)
        plt.xlabel('#Merges')
        plt.ylabel('Time(sec)')
        plt.title('WPE')
        plt.savefig('./WPE_progress.png')
        # self.save(self.vocabulary)
        return
    
    def encode_word(self, word):
        tokens = []
        while len(word)>0:
            i=len(word)

            while i>0 and word[:i] not in self.vocabulary:
                i-=1
            
            if i==0:
                return ["[UNK]"]
            
            tokens.append(word[:i])
            word = word[i:]

            if len(word) >0:
                word = '$$'+word

        return tokens
    
    def encode_file(self, path='test', portion =0.5):
        all_words = self.read_file(path, portion)
        tokens = []
        for word in all_words:
            tokens.extend(self.encode_word(word))
        
        return all_words

    def encode_text(self, text):
        words = self.seperate_words(text)

        tokens = []
        for word in words:
            tokens.extend(self.encode_word(word))
        
        return tokens
# Question 1
# obj =SimpleTokenizer()
# bpe =BPETokenizer()


# obj.encode('cord19-trec_covid-docs')
# bpe.first_itr(path = 'cord19-trec_covid-docs')
# bpe.train(merges=1000)
# start_time = time.time()
# print(time.time()-start_time)

wpe = WordPieceTokenizer()
wpe.first_itr('cord19-trec_covid-docs')
wpe.train(150)


# if __name__=="__main__":
