from tf_idf_retrieval import Retrival
import time
import pickle

def read_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pickle(data, path='indexfile.idx'):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    
def save(data , path='./output.dict'):
    with open(path,'w') as file:
        file.write('\n'.join(data))

param = {'efficiency':{0:0}, 'F1@10':{0:0}, 'F1@20':{0:0}, 'F1@50':{0:0}, 'F1@100':{0:0}}
for i in [99,199,299,399,499,599,699,799,899,999,1099,1199,1299,1399,1499,1599,1699,1799,1899,1999]:
    start_time = time.time()
    index_file = f'./inverted_index/BPE_{i}.idx'
    query_file = 'cord19-trec_covid-queries'
    print(i)
    obj = Retrival(index_file_path=index_file)
    print('-----Parameter Loaded -----', time.time()-start_time)
    ret, n = obj.retrieve(path=query_file, output_path='./result_BPE.txt')
    efficiency = (time.time()-start_time)/n
    print('Efficiency: ', efficiency)

    obj.read_qrel('./cord19-trec_covid-qrels')
    our_result = obj.parse_result(ret)
    f1= obj.calculate_score(our_result)
    print('F1@: ', f1)
    param['efficiency'][i]=efficiency
    param['F1@10'][i] = f1[10]
    param['F1@20'][i] = f1[20]
    param['F1@50'][i] = f1[50]
    param['F1@100'][i] = f1[100]

    save_pickle(param, './BPE_retrival.pkl')