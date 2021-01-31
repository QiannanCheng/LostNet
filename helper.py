#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import time, os, glob
import math
import pickle
import numpy as np
import matplotlib as mpl
from nltk import word_tokenize

mpl.use('Agg') #Linux终端没有GUI，如何使用matplotlib绘图
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def save_object(obj, filename):
    """save an object into file"""
    with open(filename, 'wb') as output:
        pickle.dump(obj, output)


def load_object(filename):
    """load object from file"""
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
    return obj


def save_plot(points, filepath, filetag, epoch):
    """Generate and save the plot"""
    path_prefix = os.path.join(filepath, filetag + '_plot_')
    path = path_prefix + 'epoch_{}.png'.format(epoch)
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)  # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    ax.plot(points)
    fig.savefig(path)
    plt.close(fig)  # close the figure
    for f in glob.glob(path_prefix + '*'):
        if f != path:
            os.remove(f)


def initialize_out_of_vocab_words(dimension, choice='random'):
    """Returns a vector of size dimension given a specific choice."""
    if choice == 'random':
        """Returns a random vector of size dimension where mean is 0 and standard deviation is 1."""
        return np.random.normal(size=dimension)
    elif choice == 'zero':
        """Returns a vector of zeros of size dimension."""
        return np.zeros(shape=dimension)


def init_embedding_weights(dictionary,emb_dim):
    """initialize weight parameters for the embedding layer."""
    pretrained_weight=np.empty([len(dictionary),emb_dim], dtype=np.float32)
    for i in range(len(dictionary)):
        if dictionary.idx2word[i] in dictionary.word2vec:
            pretrained_weight[i]=dictionary.word2vec[dictionary.idx2word[i]]
        else:
            pretrained_weight[i]=initialize_out_of_vocab_words(emb_dim)
    return pretrained_weight


def convert_to_minutes(s):
    """Converts seconds to minutes and seconds"""
    m = math.floor(s / 60) #向下取整
    s -= m * 60
    return '%dm %ds' % (m, s)


def show_progress(since, percent):
    """Prints time elapsed and estimated time remaining given the current time and progress in %"""
    now = time.time()
    s = now - since #时间流逝
    es = s / percent #估计一个epoch的总时间
    rs = es - s #估计剩余时间
    return '%s (- %s)' % (convert_to_minutes(s), convert_to_minutes(rs))


def get_batches_idx(num, batch_size):
    """ used to shuffle the dataset at each epoch"""
    idx_list=np.arange(num,dtype='int32') #array [0,1,...,num-1]
    np.random.shuffle(idx_list) #随机洗牌idx_list的顺序

    batches=[] #[array1,array2,...]
    batch_start=0
    for i in range(num//batch_size):
        batches.append(idx_list[batch_start:batch_start+batch_size])
        batch_start+=batch_size
    if batch_start!=num: #有剩余数据
        batches.append(idx_list[batch_start:])

    return batches


def get_candid_tensor(batch_data,anchor_candidates,candid_query_num):
    max_candid_next_q_length=0 #添加<eos>但不添加<sos>的最大候选查询建议长度
    for data_obj in batch_data:
        target_next_q_length=len(data_obj.next_query)-1 #添加<eos>但不添加<sos>的目标next_query长度
        if max_candid_next_q_length<target_next_q_length:
            max_candid_next_q_length=target_next_q_length
        candid_next_queries=anchor_candidates[data_obj.query_text]
        for candid_next_query in candid_next_queries:
            candid_next_q_length=len(candid_next_query)-1 #添加<eos>但不添加<sos>的候选next_query长度
            if max_candid_next_q_length<candid_next_q_length:
                max_candid_next_q_length=candid_next_q_length

    batch_size=len(batch_data)

    #(batch_size, candid_query_num, max_candid_next_q_length+1)
    candid_next_q=np.zeros((batch_size,candid_query_num,max_candid_next_q_length+1),dtype=np.int32) #添加<sos>和<eos>
    #(batch_size, candid_query_num)
    candid_next_q_len=np.zeros((batch_size,candid_query_num),dtype=np.int32) #添加<eos>但不添加<sos>的候选next_query长度

    for d in range(len(batch_data)):  #每一条数据
        candid_next_queries=anchor_candidates[batch_data[d].query_text]
        for c in range(candid_query_num-1):
            candid_next_query=candid_next_queries[c] #sentence_obj
            candid_next_q[d,c,:len(candid_next_query)]=candid_next_query.seq #添加<sos>和<eos>
            candid_next_q_len[d,c]=len(candid_next_query)-1 #添加<eos>但不添加<sos>
        target_next_query=batch_data[d].next_query #sentence_obj
        candid_next_q[d,candid_query_num-1,:len(target_next_query)]=target_next_query.seq #添加<sos>和<eos>
        candid_next_q_len[d,candid_query_num-1]=len(target_next_query)-1 #添加<eos>但不添加<sos>

    #(batch_size, candid_query_num)
    label=np.zeros((batch_size, candid_query_num),dtype=np.int32)
    label[:,-1]=1 #最后一个候选为目标查询建议

    return (candid_next_q,candid_next_q_len,label)


def batch_to_tensor(batch_data,max_query_len,max_doc_len): #batch_data: [data1,data2,...]
    """convert a batch of data to model input tensor"""
    max_session_num=0 #不包括当前session的最大历史session数目
    max_query_num=0 #历史session和当前session一起计算
    max_click_num=0 #历史session和当前session一起计算
    max_next_q_length=0 #添加<eos>但不添加<sos>的最大查询长度
    for data_obj in batch_data:
        session_num=len(data_obj.sessions)-1 #不包括当前session的历史session数目
        if max_session_num<session_num:
            max_session_num=session_num
        for session_obj in data_obj.sessions:
            query_num=len(session_obj) #一个session内的查询数目
            if max_query_num<query_num:
                max_query_num=query_num
            for query_clicks in session_obj.clicks:
                click_num=len(query_clicks) #一个查询对应的点击文档数目
                if max_click_num<click_num:
                    max_click_num=click_num
        next_q_length=len(data_obj.next_query)-1 #添加<eos>但不添加<sos>的查询长度
        if max_next_q_length<next_q_length:
            max_next_q_length=next_q_length

    batch_size=len(batch_data)
    candid_doc_num=len(batch_data[0].candid_docs) #每个查询的候选文档数目

    #(batch_size, max_session_num, max_query_num, max_query_len)
    hist_query_input=np.zeros((batch_size, max_session_num, max_query_num, max_query_len),dtype=np.int32)
    #(batch_size, max_session_num, max_query_num, max_click_num, max_doc_len)
    hist_doc_input=np.zeros((batch_size, max_session_num, max_query_num, max_click_num, max_doc_len),dtype=np.int32)
    #(batch_size)
    session_num=np.zeros((batch_size),dtype=np.int32)
    #(batch_size, max_session_num)
    hist_query_num=np.zeros((batch_size, max_session_num),dtype=np.int32)
    #(batch_size, max_session_num, max_query_num)
    hist_query_len=np.zeros((batch_size, max_session_num, max_query_num),dtype=np.int32)
    #(batch_size, max_session_num, max_query_num)
    hist_click_num=np.zeros((batch_size, max_session_num, max_query_num),dtype=np.int32)
    #(batch_size, max_session_num, max_query_num, max_click_num)
    hist_doc_len=np.zeros((batch_size, max_session_num, max_query_num, max_click_num),dtype=np.int32)

    #(batch_size, max_query_num, max_query_len)
    cur_query_input=np.zeros((batch_size, max_query_num, max_query_len),dtype=np.int32)
    #(batch_size, max_query_num, max_click_num, max_doc_len)
    cur_doc_input=np.zeros((batch_size, max_query_num, max_click_num, max_doc_len),dtype=np.int32)
    #(batch_size)
    cur_query_num=np.zeros((batch_size),dtype=np.int32)
    #(batch_size, max_query_num)
    cur_query_len=np.zeros((batch_size, max_query_num),dtype=np.int32)
    #(batch_size, max_query_num)
    cur_click_num=np.zeros((batch_size, max_query_num),dtype=np.int32)
    #(batch_size, max_query_num, max_click_num)
    cur_doc_len=np.zeros((batch_size, max_query_num, max_click_num),dtype=np.int32)

    #(batch_size, max_query_len)
    query=np.zeros((batch_size, max_query_len),dtype=np.int32)
    #(batch_size)
    q_len=np.zeros((batch_size),dtype=np.int32)
    #(batch_size, candid_doc_num, max_doc_len)
    doc=np.zeros((batch_size, candid_doc_num, max_doc_len),dtype=np.int32)
    #(batch_size, candid_doc_num)
    d_len=np.zeros((batch_size, candid_doc_num),dtype=np.int32)
    #(batch_size, candid_doc_num)
    y=np.zeros((batch_size, candid_doc_num),dtype=np.int32)
    #(batch_size, max_next_q_length+1)
    next_q=np.zeros((batch_size, max_next_q_length+1),dtype=np.int32) #添加<sos>和<eos>
    #(batch_size)
    next_q_len=np.zeros((batch_size),dtype=np.int32) #添加<eos>但不添加<sos>的查询长度

    for d in range(len(batch_data)): #每一条数据
        hist_sessions=batch_data[d].sessions[:-1] #[session_obj1,session_obj2,...]
        session_num[d]=len(hist_sessions)
        for s in range(len(hist_sessions)): #每一个历史session
            hist_queries=hist_sessions[s].queries #[sentence_obj1,sentence_obj2,...]
            hist_query_num[d,s]=len(hist_queries)
            hist_queries_clicks=hist_sessions[s].clicks  #[[sentence_obj1,sentence_obj2,...],[...],...]
            for q in range(len(hist_queries)): #每一个历史查询
                hist_query=hist_queries[q] #sentence_obj
                hist_query_input[d,s,q,:len(hist_query)]=hist_query.seq
                hist_query_len[d,s,q]=len(hist_query)
                hist_query_clicks=hist_queries_clicks[q] #[sentence_obj1,sentence_obj2,...]
                hist_click_num[d,s,q]=len(hist_query_clicks)
                for c in range(len(hist_query_clicks)): #每一个历史点击文档
                    hist_query_click=hist_query_clicks[c] #sentence_obj
                    hist_doc_input[d,s,q,c,:len(hist_query_click)]=hist_query_click.seq
                    hist_doc_len[d,s,q,c]=len(hist_query_click)

        cur_session=batch_data[d].sessions[-1] #session_obj
        cur_queries=cur_session.queries #[sentence_obj1,sentence_obj2,...]
        cur_query_num[d]=len(cur_queries)
        cur_queries_clicks=cur_session.clicks #[[sentence_obj1,sentence_obj2,...],[...],...]
        for q in range(len(cur_queries)): #每一个查询
            cur_query=cur_queries[q] #sentence_obj
            cur_query_input[d,q,:len(cur_query)]=cur_query.seq
            cur_query_len[d,q]=len(cur_query)
            cur_query_clicks=cur_queries_clicks[q] #[sentence_obj1,sentence_obj2,...]
            cur_click_num[d,q]=len(cur_query_clicks)
            for c in range(len(cur_query_clicks)): #每一个点击文档
                cur_query_click=cur_query_clicks[c] #sentence_obj
                cur_doc_input[d,q,c,:len(cur_query_click)]=cur_query_click.seq
                cur_doc_len[d,q,c]=len(cur_query_click)

        now_query=batch_data[d].query #sentence_obj
        query[d,:len(now_query)]=now_query.seq
        q_len[d]=len(now_query)
        now_candid_docs=batch_data[d].candid_docs #[sentence1_obj,sentence2_obj,...]
        for k in range(len(now_candid_docs)): #每一个候选文档
            now_candid_doc=now_candid_docs[k] #sentence_obj
            doc[d,k,:len(now_candid_doc)]=now_candid_doc.seq
            d_len[d,k]=len(now_candid_doc)
        y[d]=batch_data[d].label #[0,1,0,...]
        now_next_query=batch_data[d].next_query #sentence_obj
        next_q[d,:len(now_next_query)]=now_next_query.seq #添加<sos>和<eos>
        next_q_len[d]=len(now_next_query)-1 #添加<eos>但不添加<sos>

    return (hist_query_input, hist_doc_input, session_num, hist_query_num, hist_query_len, hist_click_num, hist_doc_len,
            cur_query_input, cur_doc_input, cur_query_num, cur_query_len, cur_click_num, cur_doc_len,
            query, q_len, doc, d_len, y, next_q, next_q_len, max_next_q_length)


def generate_predicting_text(predicting_ids,predicting_len,dictionary):
    """
    :param predicting_ids: (batch_size, predicting_len_max)
    :param predicting_len: (batch_size) include<eos>
    :param dictionary: idx2word -> [word1,word2,...]
    :return: next query string text (not include <sos> and <eos>)
    """
    batch_predicting_text=[]
    batch_size=predicting_ids.shape[0]
    for i in range(batch_size):
        ids=predicting_ids[i] #(predicting_len_max)
        len=predicting_len[i] #int
        word_list=[]
        for j in range(len-1): #0,...,len-2
            word=dictionary.idx2word[ids[j]]
            word_list.append(word)
        if ids[len-1]!=dictionary.word2idx[dictionary.eos_token]:
            word_list.append(dictionary.idx2word[ids[len-1]])
        batch_predicting_text.append(' '.join(word_list))

    return batch_predicting_text


def generate_target_text(batch_data,dictionary,max_query_len):
    batch_target_text=[]
    batch_query_text=[]
    for data in batch_data:
        ids=data.next_query.seq[1:-1] #去除<sos>和<eos>
        word_list=[]
        for id in ids:
            word=dictionary.idx2word[id]
            word_list.append(word)
        batch_target_text.append(' '.join(word_list))

        terms=word_tokenize(data.query_text)
        terms=terms[:max_query_len]
        batch_query_text.append(' '.join(terms))

    return batch_target_text,batch_query_text


