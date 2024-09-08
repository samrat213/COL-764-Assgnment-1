# COL-764-Assgnment-1
# package installation 


#BPE and WPE encoder are trained with train() function of their class with one parameter indicating number of merges.

# simple tokenizer 
* Dictionary : /home/scai/phd/aiz248314/IR/Assignment1/simple.dict
* Inverted Index File: /home/scai/phd/aiz248314/IR/Assignment1/simple.idx
* Retrival result File: /home/scai/phd/aiz248314/IR/Assignment1/retreval_simple.txt
* Code(dict_cons.py)
    encoder = SimpleTokenizer()
    encoder.encode(path)

 # BPE files
* Dictionary : /home/scai/phd/aiz248314/IR/Assignment1/BPE.dict
* Inverted Index File: /home/scai/phd/aiz248314/IR/Assignment1/BPE.idx
* Retrival result File: /home/scai/phd/aiz248314/IR/Assignment1/retreval_BPE.txt
* Code(dict_cons.py)
    encoder = BPETokenizer()
    encoder.first_itr(path)
    encoder.train(160)
* 160 is number of merges that we want to find

 # WPE files
* Dictionary : /home/scai/phd/aiz248314/IR/Assignment1/WPE.dict
* Inverted Index File: /home/scai/phd/aiz248314/IR/Assignment1/WPE.idx
* Retrival result File: /home/scai/phd/aiz248314/IR/Assignment1/retreval_WPE.txt
* Code(dict_cons.py)
    encoder = WordPieceTokenizer()
    encoder.first_itr(path)
    encoder.train(100)
* 100 is number of merges that we want to find
