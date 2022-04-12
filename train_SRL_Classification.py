import argparse
import pandas as pd
from random import shuffle
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy import sparse
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from collections import  defaultdict
from scipy.spatial import distance
from datetime import datetime, timedelta


from model_SRL_prediction import SRL_KT
from utils import *

## Code adapted from RKT (CIKM 2020) https://github.com/shalini1194/RKT and AKT (KDD 2020) https://github.com/arghosh/AKT


### to exit main function 
import sys

# generate random int values; for action types

from numpy.random import seed
from numpy.random import randint
import pickle
## embedding are saved as tf.tensor
import tensorflow as tf
## for tf excluding cuda 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# seed random number generator
seed(1)

# Seed the behavior of the environment variable
os.environ['PYTHONHASHSEED'] = str(1)
# Seed numpy's instance in case you are using numpy's random number generator, shuffling operations, ...
np.random.seed(1)

# In general seed PyTorch operations
#torch.manual_seed(0)
# 
#torch.manual_seed(0)

## qrelation ablation, time ablation 
torch.manual_seed(12)

#start = torch.cuda.Event(enable_timing=True)
#end = torch.cuda.Event(enable_timing=True)
def read_pickle(fname_dict):
  qsrl_dict ={}
  
  with open(fname_dict, "rb") as fIn:
    qsrl_dict = pickle.load(fIn)

  return(qsrl_dict)


### save embeddings
def write_pickle(df, fname_df):
  
  fout = open(fname_df, 'wb')
  
  pickle.dump(df, fout, protocol=pickle.HIGHEST_PROTOCOL)

  fout.close()


def compute_respcosine(nth_orgaction_index, nth_orgresp_index, USE_dict):
    
    nth_action_embedding = USE_dict[nth_orgaction_index]

    resp_ind = nth_orgresp_index.split('#')

    if len(resp_ind) > 1:
        resp_ind = resp_ind[0:-1]

    ## sum(cosine_sim) / len(resp_ind)
    resp_sim_sum = 0

    for i in range(len(resp_ind)):
        curr_resp_ind = resp_ind[i]

        ### One respInd is ''
        if len(curr_resp_ind) == 0:
            print("ONE Blank resp ")
            cosine_sim = 0
        else:
            curr_resp_embedding = USE_dict[curr_resp_ind]
            cosine_sim = 1 - distance.cosine(nth_action_embedding, curr_resp_embedding)
        
        resp_sim_sum += cosine_sim

    resp_sim_sum = round(resp_sim_sum/len(resp_ind), 3)

    return resp_sim_sum

def compute_respSRL_corr(prob_seq, next_seq, type_inputs, next_seq_types, respID_seq, id_to_orgdict,  respID_to_orgDict, USE_dict):
    corr= np.zeros((prob_seq.shape[0],prob_seq.shape[1], prob_seq.shape[1]))

    #print("next seq ", next_seq, "next seq types ", next_seq_types)

    for i in range(0,prob_seq.shape[0]):
        ### process current row
        for  j in range(0,next_seq.shape[1]):

            ## do so for n_th resp ind

            nth_resp_index = respID_seq[i][j]
            ## tensor to int
            nth_resp_index = nth_resp_index.item()
            ### nth_orgresp_index can be multiple for mcq. take avg
            nth_orgresp_index =  respID_to_orgDict[nth_resp_index]
            
            nth_action_type = next_seq_types[i][j]
            nth_action_type = nth_action_type.item()
            #print("nth org index ", nth_orgresp_index, " nth action type ", nth_action_type)

            

            #nth_qresp_cosine = compute_respcosine(nth_org_index, nth_orgresp_index, USE_dict)

            ## calc rep embedding only if nth action type is a ques attempt  
            if nth_action_type == 1:
                
                for k in range(j+1):
                    ## k will always start from index 0. If k == 0, that means it is a dummy 1st prev input problem ID. Put cosine sim == 0
                    prev_action_index = prob_seq[i][k]
                    prev_action_index = prev_action_index.item()

                    prev_org_index  = id_to_orgdict[prev_action_index]

                    prev_action_embedding = USE_dict[prev_org_index]

                    prev_action_type = type_inputs[i][k]
                    prev_action_type = prev_action_type.item()

                    ### get previous action type. M or S: 1, A: 2, H: 3, V: 4

                    #print("prev prob index: i ", i, " k ", k, "prev ", prev_prob_index)
                    #print("yay ", nth_orgresp_index, "j val ", j, "k val ", k, "kth prev action type ", prev_action_type)

                    if (k > 0) and prev_action_type in [2, 3, 4]:
                        #print( "In cosine compute:  prev index ", prev_action_index)
                        #nth_resp_embedding = USE_dict[nth_orgresp_index]

                        ### calc cosine similarity. Formula: cosine distance = 1 - cosine similarity
                        #cosine_sim = 1 - distance.cosine(nth_resp_embedding, prev_action_embedding)
                        ### took abs val?
                        cosine_sim = compute_respcosine(prev_org_index, nth_orgresp_index, USE_dict)
                        
                        ## round to 4 decimal place and insert value
                        corr[i][j][k] = round(cosine_sim, 3)
                    
                    ## else: corr[i][j][k] is initialized with zero



                    #corr[i][j][k] = corr_dic[nth_prob_index][prev_prob_index]
    return corr

def compute_corr(prob_seq, next_seq, type_inputs, respID_seq, id_to_orgdict,  respID_to_orgDict, USE_dict, item_types):
    ## prob_seq has 1st entry 0. next_seq is the actual prob_ID_seq solved by the std
    #### nrows x (ncol x ncol) square matrix
    #print(" prob_seq ", prob_seq)
    #print("next_seq ", next_seq)
    corr= np.zeros((prob_seq.shape[0],prob_seq.shape[1], prob_seq.shape[1]))

    for i in range(0,prob_seq.shape[0]):
        ### process current row
        for  j in range(0,next_seq.shape[1]):

            ### actual prob ID seq
            nth_prob_index = next_seq[i][j]
            ## convert tensor to int
            nth_prob_index = nth_prob_index.item()
            #print("\n \n ======= nth prob index: i ", i, " j ", j, " nth ", nth_prob_index, "=======\n")
            ### get embedding for nth_prob_index. First get org ID used in use embedding
            #print("nth problem ", nth_prob_index, " to int ", nth_prob_index.item(), " dict ", id_to_orgdict[nth_prob_index.item()])
            nth_org_index = id_to_orgdict[nth_prob_index]

            nth_embedding = USE_dict[nth_org_index]
            ## do so for n_th resp ind

            nth_resp_index = respID_seq[i][j]
            ## tensor to int
            nth_resp_index = nth_resp_index.item()
            ### nth_orgresp_index can be multiple for mcq. take avg
            nth_orgresp_index =  respID_to_orgDict[nth_resp_index]
            ## avoid padding. only add if type is 1 == ques type to previous actions
            nth_type = item_types[i][j]
            nth_type = nth_type.item()


            #nth_qresp_cosine = compute_respcosine(nth_org_index, nth_orgresp_index, USE_dict)


            for k in range(j+1):
                ## k will always start from index 0. If k == 0, that means it is a dummy 1st prev input problem ID. Put cosine sim == 0
                prev_prob_index = prob_seq[i][k]
                prev_prob_index = prev_prob_index.item()

                prev_org_index  = id_to_orgdict[prev_prob_index]

                prev_prob_embedding = USE_dict[prev_org_index]

                #print("prev prob index: i ", i, " k ", k, "prev ", prev_prob_index)

                if (k > 0) and (nth_type == 1):

                    ### calc cosine similarity. Formula: cosine distance = 1 - cosine similarity
                    cosine_sim = 1 - distance.cosine(nth_embedding, prev_prob_embedding)
                    ### took abs val?
                    
                    ## round to 4 decimal place and insert value
                    corr[i][j][k] = round(cosine_sim, 3)
                
                ## else: corr[i][j][k] is initialized with zero



                #corr[i][j][k] = corr_dic[nth_prob_index][prev_prob_index]
    return corr


def get_qresp_cos(df_, id_to_orgdict,  respID_to_orgDict, USE_dict):
    #  AId,StdID,Action_seq,ID_seq,Time_seq,Score_seq,RespID_seq

    outdf = []
    for i in range(df_.shape[0]):

        a, std = df_.iloc[i, 0], df_.iloc[i, 1]
        time_seq, score_seq = df_.iloc[i, 4], df_.iloc[i, 5]


        ## get action_seq, ID_seq, RespId_seq
        # input is string representing a sequence "[2, 3, 4, 6]". We want sequence w/o []
        stringArray_action_org,  stringArray_Id_org, stringArray_respID_org = df_.iloc[i, 2], df_.iloc[i, 3], df_.iloc[i, 6]

        stringArray_action,  stringArray_Id, stringArray_respID = np.array(stringArray_action_org[1:-1].split(",")), np.array(stringArray_Id_org[1:-1].split(",")), np.array(stringArray_respID_org[1:-1].split(","))

        floatArray_action, floatArray_Id, floatArray_respID = stringArray_action.astype(float), stringArray_Id.astype(float), stringArray_respID.astype(float)

        ### save cos for curr row. If SRL, then cos = 0
        qresp_cos = []
        for j in range(len(floatArray_action)):

            j_actiontype = floatArray_action[j]

            if j_actiontype in [2, 3, 4]:
                qresp_cos.append(0)

            if j_actiontype == 1:
                ## get qtag and rspID
                qtag, qrespId = floatArray_Id[j], floatArray_respID[j]
                ## orgId
                orgresp_index =  respID_to_orgDict[qrespId]
                orgqtag_index =  id_to_orgdict[qtag]
                j_qresp_cosval   = compute_respcosine(orgqtag_index, orgresp_index, USE_dict)
                qresp_cos.append(j_qresp_cosval)

        ## conver qresp_cos to str
        qresp_cos_str = str(qresp_cos)
        
        outdf.append((a, std, stringArray_action_org, stringArray_Id_org, time_seq, score_seq, stringArray_respID_org, qresp_cos_str))
    
    outdf = pd.DataFrame(outdf, columns = ['AId', 'StdID', 'Action_seq', 'ID_seq', 'Time_seq', 'Score_seq', 'RespID_seq', 'Qresp_cos'])
    #outdf.to_csv('C:/Drive E/Spring 2019/Actively Learn/AL_Data/AL_RQ1.2_new/EAAI/SRL_df/TESTIN_qrespcos.csv', index = False)
    return(outdf)

        

### tes get data method
def test_get_data():
    df = pd.read_csv("C:/Drive E/Fall 2020/RQ1.2/New folder/Transformer/Files_testrun/sliced_ASSISTments_12-13interactions.csv", sep='\t', encoding = "ISO-8859-1")
    print(df.columns)
    file_1, file_2 = get_data(df, 0, 5, train_split=0.8)

    print(file_1[0:2])

    #file_1.to_csv()



def prepare_batches(train_data, batch_size, randomize=True):
    """Prepare batches grouping padded sequences.

    Arguments:
        data (list of lists of torch Tensor): output by get_data
        batch_size (int): number of sequences per batch

    Output:
        batches (list of lists of torch Tensor)
    """
    # if randomize:
    #     shuffle(train_data)
    batches = []
    ## modified by Effat
    #train_y, train_skill, train_problem, timestamps, train_real_len = train_data[0], train_data[1], train_data[2], train_data[3], train_data[4]
    
    ## user_id,skill_seq,correct_seq,time_seq,problem_seq
    # in AL: AId,StdID,Action_seq,ID_seq,Time_seq,Score_seq,RespID_seq, 'Qresp_cos'
    user_id, ationType_seq, problemID_seq, time_seq,  correct_seq, respID_seq= train_data["StdID"], train_data["Action_seq"], train_data["ID_seq"], train_data["Time_seq"], train_data["Score_seq"], train_data["RespID_seq"]
    
    train_y = correct_seq
    train_problem = problemID_seq
    timestamps = time_seq

    qresp_cos = train_data["Qresp_cos"]

    item_ids = []
    ### also get item types
    item_types = []

    for i in train_problem:
        # input is string representing a sequence "[2, 3, 4, 6]". We want sequence w/o []
        stringArray = np.array(i[1:-1].split(","))
        
        # https://www.delftstack.com/howto/numpy/numpy-convert-string-array-to-float-array/
        floatArray = stringArray.astype(float)
        #floatArray = t_array.astype(float)

        item_ids.append(torch.LongTensor(floatArray))
        #print("item array length ", len(floatArray))
        

        

    for i in ationType_seq:
        # input is string representing a sequence "[2, 3, 4, 6]". We want sequence w/o []
        stringArray = np.array(i[1:-1].split(","))
        # https://www.delftstack.com/howto/numpy/numpy-convert-string-array-to-float-array/
        type_value = stringArray.astype(float)

        item_types.append(torch.LongTensor(type_value))


    timestamp = []

    for t in timestamps:
        # input is string representing a sequence "[2, 3, 4, 6]". We want sequence w/o []
        stringArray = np.array(t[1:-1].split(","))
        # https://www.delftstack.com/howto/numpy/numpy-convert-string-array-to-float-array/
        floatArray = stringArray.astype(float)
        timestamp.append(torch.LongTensor(floatArray))


    resp_ids = []

    for r in respID_seq:
        # input is string representing a sequence "[2, 3, 4, 6]". We want sequence w/o []
        stringArray = np.array(r[1:-1].split(","))
        # https://www.delftstack.com/howto/numpy/numpy-convert-string-array-to-float-array/
        floatArray = stringArray.astype(float)
        resp_ids.append(torch.LongTensor(floatArray))


    qresp_val = []

    for r in qresp_cos:
        # input is string representing a sequence "[2, 3, 4, 6]". We want sequence w/o []
        stringArray = np.array(r[1:-1].split(","))
        # https://www.delftstack.com/howto/numpy/numpy-convert-string-array-to-float-array/
        floatArray = stringArray.astype(float)
        qresp_val.append(torch.LongTensor(floatArray))



    ### SRL events have label -1
    labels = []
    #checkFormat  = lambda x : 0 if x < 1 else 1

    for i in train_y:
        # input is string representing a sequence "[2, 3, 4, 6]". We want sequence w/o []
        stringArray = np.array(i[1:-1].split(","))
        # https://www.delftstack.com/howto/numpy/numpy-convert-string-array-to-float-array/
        floatArray = stringArray.astype(float)
        ### scale by 4
        floatArray = floatArray/4

        floatArray_1 = []

        for num in floatArray:
            ### handle neg input first
            if num < 0:
                floatArray_1.append(-1)

            elif (num > 0) and (num < 1):
                floatArray_1.append(0)
            
            else:
                floatArray_1.append(1)

        ### convert to np array
        floatArray_1 = np.array(floatArray_1).astype(float)


        labels.append(torch.LongTensor(floatArray_1))

    #print("labels ", labels)


    ### Comment out
    #item_ids = [torch.LongTensor(i) for i in train_problem]
 
    ## item and labels have first input 0: to predict from the second inputs, as required by the DKT?
    item_inputs = [torch.cat( (torch.zeros(1, dtype=torch.long), i) )[:-1] for i in item_ids]
    # skill_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), s))[:-1] for s in skill_ids]
    label_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), l))[:-1] for l in labels]
    type_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), t_))[:-1] for t_ in item_types]

    qresp_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), t_))[:-1] for t_ in qresp_val]


    ### add item_types
    #data = list(zip(item_inputs, label_inputs, item_ids, timestamp, labels))
    data = list(zip(item_inputs, label_inputs, type_inputs, item_ids, timestamp, item_types, resp_ids, qresp_inputs, labels))
    #print("len data ", len(data))

    for k in range(0, len(data), batch_size):
        batch = data[k:k + batch_size]
        seq_lists = list(zip(*batch))
        #print("seq_lists  =======\n",seq_lists)

        inputs_and_ids = [pad_sequence(seqs, batch_first=True, padding_value=0)
                          for seqs in seq_lists[:-1]]

        # Pad labels with -1
        labels = pad_sequence(seq_lists[-1], batch_first=True, padding_value=-1)  # Pad labels with -1
        #print("labels ========\n ", labels)
        ### batches.append([*a, b]): results in a nested list: [[1, 2, 3, [9, 99, 99]]]. list b is added as an element to list a: [1, 2, 3, list b]
        batches.append([*inputs_and_ids, labels])
        #print("=======end preparing batches ========\n")
        #break 
    
    #print("=======end printing batch contents ========\n")
    return batches


### test prepare batch method
def test_get_batch():
    #df = pd.read_csv("C:/Drive E/Fall 2020/RQ1.2/New folder/Transformer/Files_testrun/folds_df/sliced_ASSISTments_12-13interactionsval_fold_0.csv",  encoding = "ISO-8859-1")
    df = pd.read_csv('C:/Drive E/Spring 2019/Actively Learn/AL_Data/AL_RQ1.2_new/EAAI/SRL_Splits/SRLFold_0Val.csv',  encoding = "ISO-8859-1")
    #df = pd.read_csv('C:/Drive E/Spring 2019/Actively Learn/AL_Data/AL_RQ1.2_new/EAAI/Splits/NoSRLFold_0Training.csv',  encoding = "ISO-8859-1")
    print(df.columns, " rows ", len(df))
    batches = prepare_batches(df, 2)
    ### outputs 5 tensors: item_inputs, label_inputs, item_ids, timestamp, labels
    ## with item_types, prints 6 tensors: item_inputs, label_inputs, item_ids, timestamp, item_types, labels
    print(batches[0])

    #file_1.to_csv()

### test prepare batch method
def test_get_corr(id_to_orgDict, respid_to_orgDict, USE_dict):
    #df = pd.read_csv("C:/Drive E/Fall 2020/RQ1.2/New folder/Transformer/Files_testrun/folds_df/sliced_ASSISTments_12-13interactionsval_fold_0.csv",  encoding = "ISO-8859-1")
    df = pd.read_csv('C:/Drive E/Spring 2019/Actively Learn/AL_Data/AL_RQ1.2_new/EAAI/SRL_Splits/SRLFold_4Training.csv',  encoding = "ISO-8859-1")
    print(df.columns, " rows ", len(df))

    ## slice rows with mult respID
    rows = [15, 16]
    df_slice = df.loc[rows, :]
    print(df_slice)
    batches = prepare_batches(df_slice, 1)

    ### outputs 5 tensors: item_inputs, label_inputs, item_ids, timestamp, labels
    ## with item_types, prints 6 tensors: item_inputs, label_inputs, item_ids, timestamp, item_types, labels
    #print(batches[0])
   

   ### ques corr
    #for item_inputs, label_inputs, type_inputs, item_ids, timestamp, item_types, resp_ids, labels in batches:
    #    rel = compute_corr(item_inputs, item_ids, resp_ids, id_to_orgdict, respID_to_orgDict, USE_dict)
    #    break

    for item_inputs, label_inputs, type_inputs, item_ids, timestamp, item_types, resp_ids, labels in batches:
            #rel = compute_corr(item_inputs, item_ids, type_inputs, resp_ids, id_to_orgDict, respid_to_orgDict, USE_dict)
        rel_resp = compute_respSRL_corr(item_inputs, item_ids, type_inputs, item_types, resp_ids, id_to_orgDict, respid_to_orgDict, USE_dict)
        print(rel_resp)
        break


   

def train_test_split(data, split=0.8):
    n_samples = data[0].shape[0]
    split_point = int(n_samples*split)
    train_data, test_data = [], []
    for d in data:
        train_data.append(d[:split_point])
        test_data.append(d[split_point:])
    return train_data, test_data


def compute_auc(preds, labels):
    ## rkt code
    preds = preds[labels >= 0].flatten()
    labels = labels[labels >= 0].float()
    if len(torch.unique(labels)) == 1:  # Only one class
        auc = accuracy_score(labels, preds.round())
        acc = auc
    else:
        auc = roc_auc_score(labels, preds)
        acc = accuracy_score(labels, preds.round())
    return auc, acc



def compute_loss(preds, labels, criterion):
 
    # OLD code for original rkt
    preds = preds[labels >= 0].flatten()
    labels = labels[labels >= 0].float()
    return criterion(preds, labels)

def computeRePos(time_seq, time_span):
    batch_size = time_seq.shape[0]
    size = time_seq.shape[1]
    #print("batch size ", batch_size, "size ", size)
    #print("org inp ", time_seq)

    ### repeats the original rows size*size time. Say 1st row P = [1, 22, 3] and size = 3. so P will be repeated 3 times resulting 9 length col.
    ### reshaping with nrows, 9 times P. and extra dimension. Bcz reshape always should be equal to org dim https://deeplizard.com/learn/video/fCVuiW9AFzY
    #t_row = torch.unsqueeze(time_seq, axis=1).repeat(1,size,1).reshape((batch_size, size*size,1))
    ### t_col will be epeated as [1, 1, 1], [22, 22, 22], [3, 3, 3]. 
    ## Basically, for each element in P, subtract all other elements resulting in a square matrix for batch size
    #t_col = torch.unsqueeze(time_seq,axis=-1).repeat(1, 1, size,).reshape((batch_size, size*size,1))
    #print(" row ", t_row)
    #print("t_col ", t_col)

    #print("testing 1st row - 1st col w/o abs ", t_row[0] - t_col[0])


    # The code snippet shown here uses a backslash ( \ ) in torch.abs() to split the logic across multiple lines. When the code is executed, 
    #Python will ignore the backslash, and treat the code on the next line as a direct continuation of the current line.

    # https://subscription.packtpub.com/book/data/9781838989217/1/ch01lvl1sec04/introduction-to-pytorch
    time_matrix= (torch.abs( torch.unsqueeze(time_seq, axis=1).repeat(1,size,1).reshape((batch_size, size*size,1)) - \
                 torch.unsqueeze(time_seq,axis=-1).repeat(1, 1, size,).reshape((batch_size, size*size,1)) ) ) 

    # time_matrix[time_matrix>time_span] = time_span
    time_matrix = time_matrix.reshape((batch_size,size,size))


    return (time_matrix)


def test_time_Repos():
    val_df = pd.read_csv("C:/Drive E/Fall 2020/RQ1.2/New folder/Transformer/Files_testrun/folds_df/sliced_ASSISTments_12-13interactionsval_fold_0.csv",  encoding = "ISO-8859-1")
    val_batches = prepare_batches(val_df, 2, randomize=False)

    for item_inputs, label_inputs, item_ids, timestamp, item_types, labels in val_batches:
        
        t_timestamp = timestamp[:, :3]
        ### timespan never used
        time = computeRePos(t_timestamp, 200)
        print(time)

        break



### deleted method get_corr_data

def train(train_data, val_data, timespan,  model, optimizer, logger, saver, num_epochs, batch_size, grad_clip, id_to_orgDict, respid_to_orgDict, USE_dict):
    """Train RKT model.
    Arguments:
        train_data (list of tuples of torch Tensor)
        val_data (list of tuples of torch Tensor)
        model (torch Module)
        optimizer (torch optimizer)
        logger: wrapper for TensorboardX logger
        saver: wrapper for torch saving
        num_epochs (int): number of epochs to train for
        batch_size (int)
        grad_clip (float): max norm of the gradients
    """
    ## for AL dataset, nn.CrossEntropyLoss --which is equivalen to sparse categorical crossentopry
    #criterion = nn.BCEWithLogitsLoss()
    
    criterion = nn.BCEWithLogitsLoss()

    step = 0
    metrics = Metrics()
   
    
    for epoch in range(num_epochs):

        train_batches = prepare_batches(train_data, batch_size)
        val_batches = prepare_batches(val_data, batch_size)

        # Training
        ## added item_types
        for item_inputs, label_inputs, type_inputs, item_ids, timestamp, item_types, resp_ids, qresp_inputs, labels in train_batches:
            rel = compute_corr(item_inputs, item_ids, type_inputs, resp_ids, id_to_orgDict, respid_to_orgDict, USE_dict, item_types)
            rel_resp = compute_respSRL_corr(item_inputs, item_ids, type_inputs, item_types, resp_ids, id_to_orgDict, respid_to_orgDict, USE_dict)

            

             ## comment out cuda
            item_inputs = item_inputs #item_inputs.cuda()
            time = computeRePos(timestamp, timespan)
            label_inputs =  label_inputs# label_inputs.cuda()
            item_ids = item_ids
           
            ## OUTPUT of model: lin_out(outputs), attn
            ## rel is numpy. convert it to tensor
            preds, weights = model(item_inputs, label_inputs, type_inputs, item_ids, torch.Tensor(rel), torch.Tensor(rel_resp), time, item_types, qresp_inputs)
            
            loss = compute_loss(preds, labels, criterion)


            preds = torch.sigmoid(preds).cpu()
            ### train_auc, train_acc = compute_auc(preds, labels) shows error Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
            # https://stackoverflow.com/questions/55466298/pytorch-cant-call-numpy-on-variable-that-requires-grad-use-var-detach-num
            preds = preds.detach().numpy() 

            train_auc, train_acc = compute_auc(preds, labels)
            #print(" auc ", " acc ", train_acc)

            model.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            step += 1
            metrics.store({'loss/train': loss.item()})
            metrics.store({'auc/train': train_auc})
            metrics.store({'acc/train': train_acc})

            # Logging
            if step == len(train_batches)-1:
                torch.save(weights, 'weight_tensor_rel')
            # print(step)
            if step % 1000 == 0:
                logger.log_scalars(metrics.average(), step)

               

            ### break entire train/val
            #break
            
            

        # Validation
        # ### evaluate model by model.eval()
        model.eval()
        #### added item_types
        for item_inputs, label_inputs, type_inputs, item_ids, timestamp, item_types, resp_ids, qresp_inputs, labels in val_batches:
           
            rel = compute_corr(item_inputs, item_ids, type_inputs, resp_ids, id_to_orgDict, respid_to_orgDict, USE_dict, item_types)
            rel_resp = compute_respSRL_corr(item_inputs, item_ids, type_inputs, item_types, resp_ids, id_to_orgDict, respid_to_orgDict, USE_dict)

            item_inputs = item_inputs #item_inputs.cuda()
            # skill_inputs = skill_inputs.cuda()
            time = computeRePos(timestamp, timespan)
            label_inputs = label_inputs # label_inputs.cuda()
            item_ids = item_ids # item_ids.cuda()
            # skill_ids = skill_ids.cuda()
            with torch.no_grad():
               
                preds,weights = model(item_inputs, label_inputs, type_inputs, item_ids, torch.Tensor(rel), torch.Tensor(rel_resp), time, item_types, qresp_inputs)

                preds = torch.sigmoid(preds).cpu()


            val_auc, val_acc = compute_auc(preds, labels)
            metrics.store({'auc/val': val_auc, 'acc/val': val_acc})
           

            ### break entire val
            #break
            
        model.train()

        # Save model

        average_metrics = metrics.average()
        logger.log_scalars(average_metrics, step)
        print(average_metrics)
        #stop = saver.save(average_metrics['auc/val'], model)
        stop = saver.save(average_metrics['acc/val'], model)
        if stop:
            break

        ### break entire train/val
        #break


def do_prediction_ablation(args, ablation):
 ## ForSRL, the id_to_orgDict contains re-indexed Ques + SRL ID together. Need to change ID_to_orgDict if analyze srl dataset on only ques attempt values for baselines
    id_to_orgDict = read_pickle("./Data/id_to_orgInd_srl.pkl")
    respid_to_orgDict = read_pickle("./Data/SRL_respid_to_orgInd.pkl")
    ## use SRL embedding dict
    USE_dict      = read_pickle("./Data/SRL_USE_embeddings.pkl")

    #test_get_corr(id_to_orgDict, respid_to_orgDict, USE_dict)
    
    ## CV fold 
    fold_num = args.fold

    train_fname = './Data/SRL_Splits/old/SRLFold_'+ str(fold_num) +'Training.csv'
    val_fname = './Data/SRL_Splits/old/SRLFold_' + str(fold_num) + 'Val.csv'
    test_fname = './Data/SRL_Splits/old/SRLFold_' + str(fold_num) + 'Testing.csv'

    ## read df then calc qrep cos
    train_data = pd.read_csv( train_fname,  encoding = "ISO-8859-1")
    val_data =  pd.read_csv(val_fname,  encoding = "ISO-8859-1")
    test_data = pd.read_csv(test_fname,  encoding = "ISO-8859-1")
    
    train_df = train_data
    val_df =  val_data
    test_df = test_data#

    train_df = get_qresp_cos(train_data, id_to_orgDict,  respid_to_orgDict, USE_dict) 
    val_df =  get_qresp_cos(val_data, id_to_orgDict,  respid_to_orgDict, USE_dict)
    test_df = get_qresp_cos(test_data, id_to_orgDict,  respid_to_orgDict, USE_dict)

    ablation_fname = "  time___" + str(ablation["time_include"]) + "  qrelation___" + str(ablation["qrelation_include"]) + "  qrespCos__" + str(ablation["qrespCos_include"]) + "  RespSRL__" + str(ablation["respSRLrelation_include"])
 

    
    
    model = SRL_KT(args.embed_size, args.num_attn_layers, args.num_heads,
                  args.max_pos, args.drop_prob, id_to_orgDict, USE_dict, ablation)

    optimizer = Adam(model.parameters(), lr=args.lr)

    
    
    # Reduce batch size until it fits on GPU
    
    
   
    fold_ = str(fold_num)
    
    
    param_str = ablation_fname + " fold = " + fold_
    logger = Logger_1(os.path.join(args.logdir, param_str))
    saver = Saver(args.savedir, param_str)


    train(train_df, val_df, args.timespan, model, optimizer, logger, saver, args.num_epochs,
            args.batch_size, args.grad_clip, id_to_orgDict, respid_to_orgDict, USE_dict)
    

    
    print("Done")
    
    
    
    ### For testing
    


    fold_ = str(fold_num)
    param_str = ablation_fname + " fold = " + fold_
    saver = Saver(args.savedir, param_str)
    

    model = saver.load()

    test_batches = prepare_batches(test_df, args.batch_size, randomize=False)
    test_preds = np.empty(0)

    # Predict on test set
    ## ### evaluate model by model.eval()
    model.eval()
    correct = np.empty(0)
     ### CHANGED by EFFAT: model takes 9 params in calling test_batches. include timestamp
    # item_inputs, label_inputs, type_inputs, item_ids, timestamp, item_types, resp_ids, qresp_inputs, labels

    batch_count = 0
    ### added item_types
  

    for item_inputs, label_inputs, type_inputs, item_ids, timestamp, item_types, resp_ids, qresp_inputs, labels in test_batches:
        
       
        rel = compute_corr(item_inputs, item_ids, type_inputs, resp_ids, id_to_orgDict, respid_to_orgDict, USE_dict, item_types)
        rel_resp = compute_respSRL_corr(item_inputs, item_ids, type_inputs, item_types, resp_ids, id_to_orgDict, respid_to_orgDict, USE_dict)
        
        item_inputs = item_inputs
        label_inputs = label_inputs
        item_ids = item_ids

        time = computeRePos(timestamp, args.timespan)
        with torch.no_grad():
           
            preds, weights = model(item_inputs, label_inputs, type_inputs, item_ids, torch.Tensor(rel), torch.Tensor(rel_resp), time, item_types, qresp_inputs)
          
            pred_2 =preds.squeeze(-1)
            ## Now, for each list, extract prediction from 2nd index and onwards.
             ## change: get from 2nd index. Bcz
            ## original DKT code requires >=2 sequences. So, the first label and prediction is ignored. From a student's second problem attempt
            ## we extract model's performance features. 
            
            pred_2 = pred_2[:, 1:]
            ### extract labels from index 1 to onwards. labels is 2D tensor, i.e., [list1, list 2, list3, ...]
            labels_2 = labels[:, 1:]

            ### flatten pred_2 array instead of preds
            preds = torch.sigmoid(pred_2[labels_2 >= 0]).flatten().cpu().numpy()


            test_preds = np.concatenate([test_preds, preds])

        ### change: get 2nd index. Matches with SAKT num test samples. 
        labels_2 = labels[:, 1:]

        ### get labels from labels_2
        labels = labels_2[labels_2 >=0]
      
        correct = np.concatenate([correct, labels])

        if len(test_preds) != len(correct):
            print("mismatch ", batch_count, " correct ", len(correct), " pred ", len(test_preds), "pred initi ", len(pred_init))

        #break
        batch_count +=1

        #break

    print("len arrays ", len(correct), " pred ", len(test_preds), "batch count ", batch_count)
    auc_test = roc_auc_score(correct, test_preds)
    acc_test = accuracy_score(correct, test_preds.round())

    print("auc_test = ", auc_test)
    print("acc_test = ", acc_test)

    len_test =  "len arrays " + str(len(correct)) + " pred " + str(len(test_preds)) +  " batch count " + str(batch_count)
    auc_test_str = "auc_test = " + str(auc_test)
    acc_test_str = "accuracy_test = " + str(acc_test)

    logger.log_dowrite(len_test)
    logger.log_dowrite(auc_test_str)
    logger.log_dowrite(acc_test_str)
    
    logger.close()
    print("Done Fold k = ", fold_num)

    #return

    #sys.exit(0)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RKT.')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--logdir', type=str, default='./runs/SRL_Classification')
    parser.add_argument('--savedir', type=str, default='./save/SRL_Classification')
    parser.add_argument('--max_length', type=int, default=10)
    ### embed size should be a multiple of num_head
    parser.add_argument('--embed_size', type=int, default= 512)
    parser.add_argument('--num_attn_layers', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default= 4)
    ### SRL df have max 15 length. Take 20 max seq length
    parser.add_argument('--max_pos', type=int, default= 20)
    parser.add_argument('--drop_prob', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=200)
    #parser.add_argument('--lr', type=float, default= 0.0005) # For fold 0 qrelation
    parser.add_argument('--lr', type=float, default= 1e-3)
    parser.add_argument('--grad_clip', type=float, default=10)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--timespan', default=100000, type=int)
    parser.add_argument('--fold', type=int, default=0, help='Select a fold to run.')

    args = parser.parse_args()
    print(args)
    #test_get_data()
    #test_get_batch()
    start_k = 0
    end_k   = 5

    ### do ablation
   ### For full model, all entries are set to True. For ablation, set any of the entires to False
    ablation = {"time_include": True, "qrelation_include": True, "qrespCos_include": True, "respSRLrelation_include": True}

    for i in range(start_k, end_k):
        do_prediction_ablation(args, ablation)
        #fold = args.fold
        #print("fold = ", fold)

        break


    
    
    
    
    
    
    
    

