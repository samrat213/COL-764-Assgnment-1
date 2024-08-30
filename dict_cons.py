import json
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
        # del self.tokens
        # self.training()

    def read_file(self, path, portion=0.5):
        corpus =[]
        with open(path,'r', encoding='utf-8') as file:
            lines = file.readlines()
            

            lines = lines[int(len(lines)*portion):]

            for line in lines:
                doc = json.loads(line.strip())
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
    
    def split_words(self, words):
        splitted_word = {}
        for word in words:
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
            vocabulary.update(split)

        vocabulary = list(vocabulary)
        vocabulary.sort()
        return vocabulary
    
    def calculate_pair_freq(self):
        pair_count = Counter()

        for word, freq in self.word_freq.items():
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
        words = self.seperate_words(corpus)
        words = self.add_end_of_word(words)
        self.word_freq = self.calc_freq(words)
        self.splitted_word = self.split_words(self.word_freq.keys())
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
            self.merges_best_pair(best_pair)
            print(best_pair, count, len(self.vocabulary))
        
        return
    
    def encode_text(self, text):
        words = self.seperate_words([text])
        words = self.add_end_of_word(words)
        splitted_words = self.split_words(list(set(words)))

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
        
        tokens =  sum(splitted_words, [])
        return list(set(tokens))


class WordPieceTokenizer():
    def __init__(self, path='test', full=0) -> None:
        self.pattern = r'[ ,.:;"’]+'

        words = self.get_all_words(path,full)

        self.word_freq = self.calc_freq(words)
        self.splitted_words = self.split_words(words)
        self.vocabulary = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] + self.get_vocabulary()
        # self.pair_count = self.calculate_pair_freq()
        self.merges={}
        del words
        # self.training()
    
    def get_training_corpus(self, path, full):
        corpus =[]
        with open(path,'r', encoding='utf-8') as file:
            lines = file.readlines()
            n=len(lines)
            if full==0:
                lines = lines[int(n/2):]
            for line in lines:
                line = json.loads(line.strip())

                corpus.extend([line['title'], line['abstract']])
        return corpus
    
    def get_testing_corpus(self, path, full=0):
        corpus =[]
        with open(path,'r', encoding='utf-8') as file:
            lines = file.readlines()
            n=len(lines)
            if full==0:
                lines = lines[:int(n/2)]
            for line in lines:
                line = json.loads(line.strip())

                corpus.extend([line['title'], line['abstract']])
        return corpus
    
    def get_all_words(self, path, full, testing=0):
        if testing==0:
            corpus = self.get_training_corpus(path, full)
        else:
            corpus = self.get_testing_corpus(path, full)
        all_words=[]
        for line in corpus:
            words = re.split(self.pattern, line)
            
            # Filter out empty words
            words = [word for word in words if word]
            all_words.extend(words)

        return all_words
    
    def get_words(self, corpus):
        all_words=[]
        for line in corpus:
            words = re.split(self.pattern, line)
            
            # Filter out empty words
            words = [word for word in words if word]
            all_words.extend(words)

        return all_words
    
    def calc_freq(self, tokens):
        token_count = Counter()

        for token in tokens:
            token_count[token]+=1
        
        return token_count
    
    def split_words(self, tokens):
        splitted_word = {}
        for word in tokens:
            splitted_word[word] = ["$$"+ch for ch in word]
            splitted_word[word][0] = splitted_word[word][0].removeprefix("$$")
        return splitted_word
    
    def get_vocabulary(self):
        vocabulary = []

        # for word in self.word_freq.keys():
        #     for letter in word:
        #         if letter not in vocabulary:
        #             vocabulary.append(letter)
        vocabulary =set()
        for _, split in self.splitted_words.items():
            vocabulary.union(split)

        vocabulary = list(vocabulary)
        vocabulary.sort()
        return vocabulary
    
    def get_best_pair(self, score):
        max_val = max(score.values())
        ret = list(filter(lambda x: score[x] == max_val, score))[0]
        return ret, max_val
    
    #score(A,B) = (freq(AB)*len(vocab))/ (freq(A)*freq(B))
    def calculate_pair_score(self):
        letter_count = Counter()
        pair_count = Counter()

        for word, freq in self.word_freq.items():
            charaters = self.splitted_words[word]
            n=len(charaters)

            if n==1:
                letter_count[charaters[0]]+=freq
                continue

            for i in range(n-1):
                pair = (charaters[i], charaters[i+1])
                pair_count[pair] += freq
                letter_count[charaters[i]] +=freq

            letter_count[charaters[-1]] += freq

        score ={}
        for pair,count in pair_count.items():
            score[pair] = count /(letter_count[pair[0]] * letter_count[pair[1]])
        
        return score

    def merges_best_pair(self, best_pair):

        for word,split in self.splitted_words.items():
            # n=len(split)
            if len(split)==1:
                continue
            
            i=0
            while i< len(split) - 1:
                if split[i] == best_pair[0] and split[i+1] == best_pair[1]:
                    merge = best_pair[0] + best_pair[1].removeprefix("$$")
                    split = split[:i] + [merge] + split[i+2:]
                
                i+=1
            
            self.splitted_words[word] = split
            print(split)
        
        return 1
        
    def training(self, k=500):
        for i in range(k):
            score = self.calculate_pair_score()
            if len(score) ==0:
                break
            best_pair, count = self.get_best_pair(score)

            self.merges[best_pair] = best_pair[0] + best_pair[1].removeprefix("$$")
            self.merges_best_pair(best_pair)
            self.vocabulary.append(self.merges[best_pair])
            # print(merges)
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
    
    def encode(self, path='test', full=0):
        all_words = self.get_all_words(path, testing=1, full=full)
        tokens = []
        for word in all_words:
            tokens.extend(self.encode_word(word))
        
        return all_words

