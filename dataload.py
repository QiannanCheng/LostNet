#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os,json
import random
import numpy as np
from nltk import word_tokenize
from collections import Counter

class Dictionary(object):
    def __init__(self):
        self.word2idx={} #{'word1':idx1,'word2':idx2,...}
        self.idx2word=[] #['word1','word2',...]
        self.word2vec={} #{'word1':np_vec1,'word2':np_vec2,...}
        self.pad_token='<pad>'#padding
        self.idx2word.append(self.pad_token)
        self.word2idx[self.pad_token]=len(self.idx2word)-1
        self.unk_token='<unk>' #unknown
        self.idx2word.append(self.unk_token)
        self.word2idx[self.unk_token]=len(self.idx2word)-1
        self.sos_token='<sos>' #start of sentence
        self.idx2word.append(self.sos_token)
        self.word2idx[self.sos_token]=len(self.idx2word)-1
        self.eos_token='<eos>' #end of sentence
        self.idx2word.append(self.eos_token)
        self.word2idx[self.eos_token]=len(self.idx2word)-1

    def build_dict(self,in_file,max_words,filter_frequency,frequency_limit):
        assert os.path.exists(in_file)

        word_count=Counter() #{'word1':count1,'word2':count2,...}
        with open(in_file,'r') as f:
            for line in f:
                user=json.loads(line.strip())
                for session in user:
                    for query in session['query']:
                        query_terms=word_tokenize(query)
                        word_count.update(query_terms)
                    for clicks in session['click']:
                        for click in clicks:
                            click_terms=word_tokenize(click)
                            word_count.update(click_terms)

        print('total unique words = ',len(word_count))
        if filter_frequency:
            for word,count in word_count.items():
                if count>frequency_limit:
                    self.idx2word.append(word)
                    self.word2idx[word]=len(self.idx2word)-1
        else:
            most_common=word_count.most_common(max_words) #[('word1',count1),('word2',count2),...]
            for word,count in most_common:
                self.idx2word.append(word)
                self.word2idx[word]=len(self.idx2word)-1

    def load_word_embedding(self,in_file):
        with open(in_file,'rb') as f:
            for line in f:
                line=line.decode('utf-8','ignore').strip()
                word,vec=line.split(' ',1)
                if word in self.idx2word:
                    self.word2vec[word]=np.array(list(map(float,vec.split())))
        print('number of OOV words = ',len(self.idx2word)-len(self.word2vec)) #out of vocabulary 这一部分单词的embedding需要随机初始化

    def __len__(self):
        return len(self.idx2word)

class Sentence(object):
    def __init__(self,tag=False):
        self.seq=[] #[word_idx1,word_idx2,...] int
        self.is_next_query=tag

    def sen2seq(self,sentence,dictionary,max_len):
        """convert a sentence(string) to a sequence of word indices"""
        terms=word_tokenize(sentence)
        terms=terms[:max_len]
        if self.is_next_query:
            #添加<sos>和<eos>
            terms=[dictionary.sos_token]+terms+[dictionary.eos_token]
        for term in terms:
            if term in dictionary.word2idx:
                self.seq.append(dictionary.word2idx[term])
            else:
                self.seq.append(dictionary.word2idx[dictionary.unk_token])

    def __len__(self):
        return len(self.seq)

class Session(object):
    def __init__(self):
        self.queries=[] #[sentence1,sentence2,...]
        self.clicks=[] #[[sentence1,sentence2,...],[...],...]

    def build_queries(self,queries_corpus,dictionary,max_query_len):
        for query in queries_corpus:
            sen_obj=Sentence()
            sen_obj.sen2seq(query,dictionary,max_query_len)
            self.queries.append(sen_obj)

    def build_clicks(self,clicks_corpus,dictionary,max_doc_len,click_num_limit):
        for query in clicks_corpus:
            if len(query)>click_num_limit:
                query=random.sample(query,click_num_limit) #随机抽取其中click_num_limit个点击文档
            query_clicks=[]
            for click in query:
                sen_obj=Sentence()
                sen_obj.sen2seq(click,dictionary,max_doc_len)
                query_clicks.append(sen_obj)
            self.clicks.append(query_clicks)

    def __len__(self):
        assert len(self.queries)==len(self.clicks)

        return len(self.queries)

class Data(object):
    def __init__(self):
        self.sessions=[] #[session1,session2,...] 包括当前session
        self.query_text="" #用于识别query的候选查询建议
        self.anchor_tag=False #如果当前发布的查询为一个session的倒数第二个查询，则为True
        self.query=Sentence()
        self.next_query=Sentence(tag=True) #添加<sos>和<eos>
        self.candid_docs=[] #[sentence1,sentence2,...]
        self.label=[] #[0,1,0,...] int

    def build_sessions(self,hist_session_corpus,cur_session_corpus,idx,dictionary,max_query_len,max_doc_len,click_num_limit): #idx: 当前查询的索引
        #构建历史session
        for sess in hist_session_corpus:
            sess_obj=Session()
            sess_obj.build_queries(sess['query'],dictionary,max_query_len)
            sess_obj.build_clicks(sess['click'],dictionary,max_doc_len,click_num_limit)
            self.sessions.append(sess_obj)
        #构建当前session
        sess_obj=Session()
        sess_obj.build_queries(cur_session_corpus['query'][:idx+1],dictionary,max_query_len) #包括当前query
        clicks_corpus=cur_session_corpus['click'][:idx] #不包括当前查询对应的点击文档
        clicks_corpus.append([cur_session_corpus['query'][idx]]) #当前查询->当前查询对应的一个点击文档
        sess_obj.build_clicks(clicks_corpus,dictionary,max_doc_len,click_num_limit)
        self.sessions.append(sess_obj)

    def build_others(self,cur_session_corpus,idx,dictionary,max_query_len,max_doc_len):
        self.query_text=cur_session_corpus['query'][idx]
        if idx==len(cur_session_corpus['query'])-2:
            self.anchor_tag=True
        self.query.sen2seq(cur_session_corpus['query'][idx],dictionary,max_query_len)
        self.next_query.sen2seq(cur_session_corpus['query'][idx+1],dictionary,max_query_len)

        for key,value in cur_session_corpus['candid'][idx]:
            sen_obj=Sentence()
            sen_obj.sen2seq(key,dictionary,max_doc_len)
            self.candid_docs.append(sen_obj)
            self.label.append(int(value)) #0/1

class Dataset(object):
    def __init__(self,max_query_len,max_doc_len,hist_session_num_limit,click_num_limit):
        self.dataset=[] #[data1,data2,...]
        self.max_query_len=max_query_len
        self.max_doc_len=max_doc_len
        self.hist_session_num_limit=hist_session_num_limit
        self.click_num_limit=click_num_limit

    def parse(self,in_file,dictionary,max_example=None):
        """parses the content of a file."""
        assert os.path.exists(in_file)

        total_data=0
        with open(in_file,'r') as f:
            for line in f:
                user_corpus=json.loads(line.strip()) #一个用户的数据
                #从一个用户的第二个session开始解析（保证至少有一个历史session）
                for i in range(1,len(user_corpus)):
                    if i>self.hist_session_num_limit: #i代表历史session的数目
                        hist_session_corpus=user_corpus[i-self.hist_session_num_limit:i]
                    else:
                        hist_session_corpus=user_corpus[:i]
                    cur_session_corpus=user_corpus[i]
                    #从一个session的第一个查询开始解析，直到一个session的倒数第二个查询（保证具有next query）
                    for j in range(0,len(cur_session_corpus['query'])-1):
                        data_obj=Data()
                        data_obj.build_sessions(hist_session_corpus,cur_session_corpus,j,dictionary,self.max_query_len,self.max_doc_len,self.click_num_limit)
                        data_obj.build_others(cur_session_corpus,j,dictionary,self.max_query_len,self.max_doc_len)
                        self.dataset.append(data_obj)
                        total_data+=1
                        if total_data==max_example:
                            return

    def __len__(self):
        return len(self.dataset)

