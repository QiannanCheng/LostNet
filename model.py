#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense
np.random.seed(82)
tf.set_random_seed(82)

class LostNet:
    def __init__(self,
                 vocab_size,
                 emb_dim,
                 max_query_len,
                 max_doc_len,
                 query_encoder_units,
                 query_atten_units,
                 doc_encoder_units,
                 doc_atten_units,
                 hidden_units,
                 atten_hidden_units,
                 memory_k,
                 memory_H,
                 decoder_units,
                 vocab,
                 learning_rate,
                 candid_query_num,
                 pretrained_weight,
                 re_lambda):
        self.vocab_size=vocab_size
        self.emb_dim=emb_dim
        self.max_query_len=max_query_len
        self.max_doc_len=max_doc_len
        self.query_encoder_units = query_encoder_units
        self.query_atten_units = query_atten_units
        self.doc_encoder_units = doc_encoder_units
        self.doc_atten_units = doc_atten_units
        self.hidden_units=hidden_units
        self.atten_hidden_units=atten_hidden_units
        self.memory_k=memory_k
        self.memory_H=memory_H
        self.decoder_units=decoder_units
        self.vocab=vocab
        self.learning_rate=learning_rate
        self.candid_query_num=candid_query_num
        self.pretrained_weight=pretrained_weight
        self.re_lambda=re_lambda

        self.build_graph()

    def build_graph(self):
        self.params=self.init_params()

        """
        Model Input
        """
        #(batch_size, max_session_num, max_query_num, max_query_len)
        self.hist_query_input=tf.placeholder(tf.int32,[None, None, None, self.max_query_len])
        #(batch_size, max_session_num, max_query_num, max_click_num, max_doc_len)
        self.hist_doc_input = tf.placeholder(tf.int32,[None, None, None, None, self.max_doc_len])
        #(batch_size)
        self.session_num=tf.placeholder(tf.int32,[None])
        #(batch_size, max_session_num)
        self.hist_query_num=tf.placeholder(tf.int32,[None, None])
        #(batch_size, max_session_num, max_query_num)
        self.hist_query_len=tf.placeholder(tf.int32,[None, None, None])
        #(batch_size, max_session_num, max_query_num)
        self.hist_click_num=tf.placeholder(tf.int32,[None, None, None])
        #(batch_size, max_session_num, max_query_num, max_click_num)
        self.hist_doc_len=tf.placeholder(tf.int32,[None, None, None, None])

        #(batch_size, max_query_num, max_query_len)
        self.cur_query_input=tf.placeholder(tf.int32,[None, None, self.max_query_len])
        #(batch_size, max_query_num, max_click_num, max_doc_len)
        self.cur_doc_input=tf.placeholder(tf.int32,[None, None, None, self.max_doc_len])
        #(batch_size)
        self.cur_query_num=tf.placeholder(tf.int32,[None])
        #(batch_size, max_query_num)
        self.cur_query_len=tf.placeholder(tf.int32,[None, None])
        #(batch_size, max_query_num)
        self.cur_click_num=tf.placeholder(tf.int32,[None, None])
        #(batch_size, max_query_num, max_click_num)
        self.cur_doc_len=tf.placeholder(tf.int32,[None, None, None])

        #(batch_size, max_query_len)
        self.q=tf.placeholder(tf.int32,[None, self.max_query_len])
        #(batch_size)
        self.q_len=tf.placeholder(tf.int32,[None])
        #(batch_size, candid_doc_num, max_doc_len)
        self.d=tf.placeholder(tf.int32,[None, None, self.max_doc_len])
        #(batch_size, candid_doc_num)
        self.d_len=tf.placeholder(tf.int32,[None, None])
        #(batch_size, candid_doc_num)
        self.y=tf.placeholder(tf.int32,[None, None])

        #(batch_size, memory_k, 2)
        self.indices=tf.placeholder(tf.int32,[None, self.memory_k, 2])
        #(batch_size)
        self.slots_num=tf.placeholder(tf.int32,[None])

        """
        Query Encoder (Bi-LSTM + Inner Attention)
        """
        #(batch_size*max_session_num*max_query_num, max_query_len)
        hist_query_input_reshape=tf.reshape(self.hist_query_input,[-1,self.max_query_len])
        #(batch_size*max_query_num, max_query_len)
        cur_query_input_reshape=tf.reshape(self.cur_query_input,[-1,self.max_query_len])
        #(batch_size*max_session_num*max_query_num + batch_size*max_query_num + batch_size, max_query_len)
        all_query=tf.concat((hist_query_input_reshape,cur_query_input_reshape,self.q),axis=0)
        #(batch_size*max_session_num*max_query_num + batch_size*max_query_num + batch_size, max_query_len, emb_dim)
        all_query_emb=tf.nn.embedding_lookup(self.params['Wemb'],all_query)

        #(batch_size*max_session_num*max_query_num)
        hist_query_len_reshape=tf.reshape(self.hist_query_len,[-1])
        #(batch_size*max_query_num)
        cur_query_len_reshape=tf.reshape(self.cur_query_len,[-1])
        #(batch_size*max_session_num*max_query_num + batch_size*max_query_num + batch_size)
        all_query_len=tf.concat((hist_query_len_reshape,cur_query_len_reshape,self.q_len),axis=0)

        with tf.variable_scope('query_encoder'):
            lstm_fw_cell=tf.contrib.rnn.BasicLSTMCell(self.query_encoder_units)
            lstm_bw_cell=tf.contrib.rnn.BasicLSTMCell(self.query_encoder_units)
            #(batch_size*max_session_num*max_query_num + batch_size*max_query_num + batch_size, query_encoder_units)
            init_fw=lstm_fw_cell.zero_state(tf.shape(all_query_emb)[0],dtype=tf.float32)
            #(batch_size*max_session_num*max_query_num + batch_size*max_query_num + batch_size, query_encoder_units)
            init_bw=lstm_bw_cell.zero_state(tf.shape(all_query_emb)[0],dtype=tf.float32)
            #query_encoder_outputs: 二元组(output_fw, output_bw),(batch_size*max_session_num*max_query_num + batch_size*max_query_num + batch_size, max_query_len, query_encoder_units)
            query_encoder_outputs,query_encoder_state=tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,cell_bw=lstm_bw_cell,
                                                                                      inputs=all_query_emb,sequence_length=all_query_len,
                                                                                      initial_state_fw=init_fw,
                                                                                      initial_state_bw=init_bw,dtype=tf.float32)
        #(batch_size*max_session_num*max_query_num + batch_size*max_query_num + batch_size, max_query_len, query_encoder_units*2)
        query_encoder_outputs=tf.concat(query_encoder_outputs,2) #连接前向和后向units

        #(batch_size, max_query_len, query_encoder_units*2)
        _,q_hiddens=tf.split(query_encoder_outputs,num_or_size_splits=[tf.shape(hist_query_input_reshape)[0]+tf.shape(cur_query_input_reshape)[0],tf.shape(self.q)[0]],axis=0)

        with tf.variable_scope('attention_layer1'):
            #(batch_size*max_session_num*max_query_num + batch_size*max_query_num + batch_size, max_query_len, query_atten_units)
            query_ratios=tf.layers.dense(inputs=query_encoder_outputs,use_bias=True,units=self.query_atten_units,activation=tf.nn.tanh)
            #(batch_size*max_session_num*max_query_num + batch_size*max_query_num + batch_size, max_query_len, 1)
            query_ratios=tf.layers.dense(inputs=query_ratios,use_bias=True,units=1,activation=None)
        #(batch_size*max_session_num*max_query_num + batch_size*max_query_num + batch_size, max_query_len, 1)
        query_atten_mask=tf.expand_dims(tf.sequence_mask(all_query_len,self.max_query_len,dtype=tf.float32),2)
        #(batch_size*max_session_num*max_query_num + batch_size*max_query_num + batch_size, 1, 1)
        all_query_len_reshape=tf.reshape(all_query_len,[-1,1,1])
        #(batch_size*max_session_num*max_query_num + batch_size*max_query_num + batch_size, 1, 1)
        query_adjust=tf.cast(tf.equal(all_query_len_reshape,0),dtype=tf.float32)
        #(batch_size*max_session_num*max_query_num + batch_size*max_query_num + batch_size, max_query_len, 1)
        query_atten=tf.exp(query_ratios)/(tf.expand_dims(tf.reduce_sum(tf.exp(query_ratios)*query_atten_mask,1),1)+query_adjust)*query_atten_mask #(batch_size*max_session_num*max_query_num + batch_size*max_query_num + batch_size, 1, 1)
        #(batch_size*max_session_num*max_query_num + batch_size*max_query_num + batch_size, query_encoder_units*2)
        all_query_vec=tf.reduce_sum(query_encoder_outputs*query_atten,1)

        #hist_query_vec: (batch_size*max_session_num*max_query_num, query_encoder_units*2)
        #cur_query_vec: (batch_size*max_query_num, query_encoder_units*2)
        #q_vec: (batch_size, query_encoder_units*2)
        hist_query_vec,cur_query_vec,self.q_vec=tf.split(all_query_vec,num_or_size_splits=[tf.shape(hist_query_input_reshape)[0],tf.shape(cur_query_input_reshape)[0],tf.shape(self.q)[0]],axis=0)
        #(batch_size, max_session_num, max_query_num, query_encoder_units*2)
        self.hist_query_vec=tf.reshape(hist_query_vec,[-1,tf.shape(self.hist_query_input)[1],tf.shape(self.hist_query_input)[2],self.query_encoder_units*2])
        #(batch_size, max_query_num, query_encoder_units*2)
        self.cur_query_vec=tf.reshape(cur_query_vec,[-1,tf.shape(self.cur_query_input)[1],self.query_encoder_units * 2])

        """
        Document Encoder (Bi-LSTM + Inner Attention)
        """
        #(batch_size*max_session_num*max_query_num*max_click_num, max_doc_len)
        hist_doc_input_reshape=tf.reshape(self.hist_doc_input,[-1,self.max_doc_len])
        #(batch_size*max_query_num*max_click_num, max_doc_len)
        cur_doc_input_reshape=tf.reshape(self.cur_doc_input,[-1,self.max_doc_len])
        #(batch_size*candid_doc_num, max_doc_len)
        d_reshape=tf.reshape(self.d,[-1,self.max_doc_len])
        #(batch_size*max_session_num*max_query_num*max_click_num + batch_size*max_query_num*max_click_num + batch_size*candid_doc_num, max_doc_len)
        all_doc=tf.concat((hist_doc_input_reshape,cur_doc_input_reshape,d_reshape),axis=0)
        #(batch_size*max_session_num*max_query_num*max_click_num + batch_size*max_query_num*max_click_num + batch_size*candid_doc_num, max_doc_len, emb_dim)
        all_doc_emb=tf.nn.embedding_lookup(self.params['Wemb'],all_doc)

        #(batch_size*max_session_num*max_query_num*max_click_num)
        hist_doc_len_reshape=tf.reshape(self.hist_doc_len,[-1])
        #(batch_size*max_query_num*max_click_num)
        cur_doc_len_reshape=tf.reshape(self.cur_doc_len,[-1])
        #(batch_size*candid_doc_num)
        d_len_reshape=tf.reshape(self.d_len,[-1])
        #(batch_size*max_session_num*max_query_num*max_click_num + batch_size*max_query_num*max_click_num + batch_size*candid_doc_num)
        all_doc_len=tf.concat((hist_doc_len_reshape,cur_doc_len_reshape,d_len_reshape),axis=0)

        with tf.variable_scope('document_encoder'):
            dlstm_fw_cell=tf.contrib.rnn.BasicLSTMCell(self.doc_encoder_units)
            dlstm_bw_cell=tf.contrib.rnn.BasicLSTMCell(self.doc_encoder_units)
            #(batch_size*max_session_num*max_query_num*max_click_num + batch_size*max_query_num*max_click_num + batch_size*candid_doc_num, doc_encoder_units)
            dinit_fw=dlstm_fw_cell.zero_state(tf.shape(all_doc_emb)[0],dtype=tf.float32)
            #(batch_size*max_session_num*max_query_num*max_click_num + batch_size*max_query_num*max_click_num + batch_size*candid_doc_num, doc_encoder_units)
            dinit_bw=dlstm_bw_cell.zero_state(tf.shape(all_doc_emb)[0],dtype=tf.float32)
            #doc_encoder_outputs: 二元组(output_fw, output_bw),(batch_size*max_session_num*max_query_num*max_click_num + batch_size*max_query_num*max_click_num + batch_size*candid_doc_num, max_doc_len, doc_encoder_units)
            doc_encoder_outputs,doc_encoder_state=tf.nn.bidirectional_dynamic_rnn(cell_fw=dlstm_fw_cell,cell_bw=dlstm_bw_cell,
                                                                                  inputs=all_doc_emb,sequence_length=all_doc_len,
                                                                                  initial_state_fw=dinit_fw,
                                                                                  initial_state_bw=dinit_bw,dtype=tf.float32)
        #(batch_size*max_session_num*max_query_num*max_click_num + batch_size*max_query_num*max_click_num + batch_size*candid_doc_num, max_doc_len, doc_encoder_units*2)
        doc_encoder_outputs=tf.concat(doc_encoder_outputs,2) #连接前向和后向units

        with tf.variable_scope('attention_layer2'):
            #(batch_size*max_session_num*max_query_num*max_click_num + batch_size*max_query_num*max_click_num + batch_size*candid_doc_num, max_doc_len, doc_atten_units)
            doc_ratios=tf.layers.dense(inputs=doc_encoder_outputs,use_bias=True,units=self.doc_atten_units,activation=tf.nn.tanh)
            #(batch_size*max_session_num*max_query_num*max_click_num + batch_size*max_query_num*max_click_num + batch_size*candid_doc_num, max_doc_len, 1)
            doc_ratios=tf.layers.dense(inputs=doc_ratios,use_bias=True,units=1,activation=None)
        #(batch_size*max_session_num*max_query_num*max_click_num + batch_size*max_query_num*max_click_num + batch_size*candid_doc_num, max_doc_len, 1)
        doc_atten_mask=tf.expand_dims(tf.sequence_mask(all_doc_len,self.max_doc_len,dtype=tf.float32),2)
        #(batch_size*max_session_num*max_query_num*max_click_num + batch_size*max_query_num*max_click_num + batch_size*candid_doc_num, 1, 1)
        all_doc_len_reshape=tf.reshape(all_doc_len,[-1,1,1])
        #(batch_size*max_session_num*max_query_num*max_click_num + batch_size*max_query_num*max_click_num + batch_size*candid_doc_num, 1, 1)
        doc_adjust=tf.cast(tf.equal(all_doc_len_reshape,0),dtype=tf.float32)
        #(batch_size*max_query_num*max_click_num+batch_size*candid_doc_num, max_doc_len, 1)
        doc_atten=tf.exp(doc_ratios)/(tf.expand_dims(tf.reduce_sum(tf.exp(doc_ratios)*doc_atten_mask,1),1)+doc_adjust)*doc_atten_mask  #(batch_size*max_session_num*max_query_num*max_click_num + batch_size*max_query_num*max_click_num + batch_size*candid_doc_num, 1, 1)
        #(batch_size*max_session_num*max_query_num*max_click_num + batch_size*max_query_num*max_click_num + batch_size*candid_doc_num, doc_encoder_units*2)
        all_doc_vec=tf.reduce_sum(doc_encoder_outputs*doc_atten,1)

        #hist_doc_vec: (batch_size*max_session_num*max_query_num*max_click_num, doc_encoder_units*2)
        #cur_doc_vec: (batch_size*max_query_num*max_click_num, doc_encoder_units*2)
        #d_vec: (batch_size*candid_doc_num, doc_encoder_units*2)
        hist_doc_vec,cur_doc_vec,d_vec=tf.split(all_doc_vec,num_or_size_splits=[tf.shape(hist_doc_input_reshape)[0],tf.shape(cur_doc_input_reshape)[0],tf.shape(d_reshape)[0]],axis=0)
        #(batch_size, max_session_num, max_query_num, max_click_num, doc_encoder_units*2)
        hist_doc_vec=tf.reshape(hist_doc_vec,[-1,tf.shape(self.hist_doc_input)[1],tf.shape(self.hist_doc_input)[2],tf.shape(self.hist_doc_input)[3],self.doc_encoder_units*2])
        #(batch_size, max_session_num, max_query_num, max_click_num, 1)
        hist_Dmask=tf.expand_dims(tf.sequence_mask(self.hist_click_num,tf.shape(self.hist_doc_input)[3],dtype=tf.float32),4)
        #(batch_size, max_session_num, max_query_num, 1)
        adjust1=tf.cast(tf.equal(tf.expand_dims(self.hist_click_num,3),0),dtype=tf.float32)
        #(batch_size, max_session_num, max_query_num, doc_encoder_units*2)
        self.hist_Dvec=tf.reduce_sum(hist_doc_vec*hist_Dmask,3)/(tf.cast(tf.expand_dims(self.hist_click_num,3),dtype=tf.float32)+adjust1) #(batch_size, max_session_num, max_query_num, 1)
        #(batch_size, max_query_num, max_click_num, doc_encoder_units*2)
        cur_doc_vec=tf.reshape(cur_doc_vec,[-1,tf.shape(self.cur_doc_input)[1],tf.shape(self.cur_doc_input)[2],self.doc_encoder_units*2])
        #(batch_size, max_query_num, max_click_num, 1)
        cur_Dmask=tf.expand_dims(tf.sequence_mask(self.cur_click_num,tf.shape(self.cur_doc_input)[2],dtype=tf.float32),3)
        #(batch_size, max_query_num, 1)
        adjust2=tf.cast(tf.equal(tf.expand_dims(self.cur_click_num,2),0),dtype=tf.float32)
        #(batch_size, max_query_num, doc_encoder_units*2)
        self.cur_Dvec=tf.reduce_sum(cur_doc_vec*cur_Dmask,2)/(tf.cast(tf.expand_dims(self.cur_click_num,2),dtype=tf.float32)+adjust2) #(batch_size, max_query_num, 1)
        #(batch_size, candid_doc_num, doc_encoder_units*2)
        self.d_vec=tf.reshape(d_vec,[-1,tf.shape(self.d)[1],self.doc_encoder_units * 2])

        """
        GRU
        """
        with tf.variable_scope('gru'):
            #(batch_size*max_session_num, max_query_num, query_encoder_units*2 + doc_encoder_units*2)
            hist_gru_input=tf.reshape(tf.concat((self.hist_query_vec,self.hist_Dvec),3),[-1,tf.shape(self.hist_query_vec)[2],self.query_encoder_units*2+self.doc_encoder_units*2])
            #(batch_size, max_query_num, query_encoder_units*2 + doc_encoder_units)
            cur_gru_input=tf.concat((self.cur_query_vec,self.cur_Dvec),2)
            #(batch_size*max_session_num + batch_size, max_query_num, query_encoder_units*2 + doc_encoder_units*2)
            gru_input=tf.concat((hist_gru_input,cur_gru_input),0)
            #(batch_size*max_session_num+batch_size)
            gru_input_len=tf.concat((tf.reshape(self.hist_query_num,[-1]),self.cur_query_num),0)

            gru_cell=tf.nn.rnn_cell.GRUCell(self.hidden_units)
            #(batch_size*max_session_num+batch_size, hidden_units)
            gru_init_state=gru_cell.zero_state(tf.shape(gru_input)[0],tf.float32)
            #gru_outputs:(batch_size*max_session_num+batch_size, max_query_num, hidden_units)
            #gru_state:(batch_size*max_session_num+batch_size, hidden_units)
            #若max_steps=3,num_steps=2: 对于2以后的padding不进行计算，last_state将重复第2步的last_state直至第3步，而outputs中超过第2步的结果将会被置为0
            gru_outputs,gru_state=tf.nn.dynamic_rnn(cell=gru_cell,inputs=gru_input,sequence_length=gru_input_len,
                                                    initial_state=gru_init_state,dtype=tf.float32)

        """
        Inner Attention
        """
        with tf.variable_scope('attention_layer3'):
            #(batch_size*max_session_num+batch_size, max_query_num, atten_hidden_units)
            ratios=tf.layers.dense(inputs=gru_outputs,use_bias=True,units=self.atten_hidden_units,activation=tf.nn.tanh)
            #(batch_size*max_session_num+batch_size, max_query_num, 1)
            ratios=tf.layers.dense(inputs=ratios,use_bias=True,units=1,activation=None)
        #(batch_size*max_session_num+batch_size, max_query_num, 1)
        atten_mask=tf.expand_dims(tf.sequence_mask(gru_input_len,tf.shape(ratios)[1],dtype=tf.float32),2)
        #(batch_size*max_session_num+batch_size, 1, 1)
        query_num=tf.concat((tf.reshape(self.hist_query_num,[-1,1,1]),tf.reshape(self.cur_query_num,[-1,1,1])),0)
        #(batch_size*max_session_num+batch_size, 1, 1)
        adjust3=tf.cast(tf.equal(query_num,0),dtype=tf.float32)
        #(batch_size*max_session_num+batch_size, max_query_num, 1)
        atten=tf.exp(ratios)/(tf.expand_dims(tf.reduce_sum(tf.exp(ratios)*atten_mask,1),1)+adjust3)*atten_mask #(batch_size*max_session_num+batch_size, 1, 1)
        #(batch_size*max_session_num+batch_size, hidden_units)
        sess_rep=tf.reduce_sum(gru_outputs*atten,1)
        #hist_sess_rep: (batch_size*max_session_num, hidden_units)
        #cur_sess_rep: (batch_size, hidden_units)
        hist_sess_rep,self.cur_sess_rep=tf.split(sess_rep,num_or_size_splits=[tf.shape(hist_gru_input)[0],tf.shape(cur_gru_input)[0]],axis=0)

        #(batch_size, hidden_units)
        self.x=self.cur_sess_rep #用于生成memory read vector

        #(batch_size, max_session_num, hidden_units)
        self.hist_sess_rep=tf.reshape(hist_sess_rep,[-1,tf.shape(self.hist_query_input)[1],self.hidden_units]) #用于生成historical session interests存入memory

        """
        Personalized User Memory
        """
        #writing operation
        #(batch_size, 1, hidden_units)
        zero_padding=tf.zeros(shape=tf.shape(tf.expand_dims(self.x,1)),dtype=tf.float32)
        #(batch_size, 1+max_session_num, hidden_units)
        p_signal=tf.concat((zero_padding,self.hist_sess_rep),axis=1)
        #(batch_size, memory_k, hidden_units)
        self.updated_M=tf.gather_nd(params=p_signal,indices=self.indices)

        #reading operation
        #(batch_size, hidden_units, 1)
        self.r=tf.nn.tanh(tf.expand_dims(self.x, 2))
        #(batch_size, memory_k, 1)
        self.w=tf.matmul(self.updated_M, self.r) #内积
        #(batch_size, memory_k, 1)
        self.memory_mask=tf.expand_dims(tf.sequence_mask(self.slots_num, self.memory_k, dtype=tf.float32), 2)
        #(batch_size, 1 ,1)
        w_max=tf.expand_dims(tf.reduce_max(self.w+(1-self.memory_mask)*(-1000),1),1)
        #(batch_size, memory_k, 1)
        self.z=tf.exp((self.w-w_max)*self.memory_mask)/tf.expand_dims(tf.reduce_sum(tf.exp((self.w-w_max)*self.memory_mask)*self.memory_mask,1),1)*self.memory_mask
        #(batch_size, hidden_units)
        self.o=tf.reduce_sum(self.updated_M*self.z,1)
        #(batch_size, 1, hidden_units)
        read_out=tf.expand_dims(self.o,1)
        for j in range(2,self.memory_H+1):  #Hops j=2,...,H
            #(batch_size, hidden_units)
            self.r=tf.nn.tanh(tf.matmul(self.x+self.o,self.params['Rmatrix'][j-2]))
            #(batch_size, hidden_units, 1)
            self.r=tf.expand_dims(self.r,2)
            #(batch_size, memory_k, 1)
            self.w=tf.matmul(self.updated_M,self.r) #内积
            #(batch_size, 1, 1)
            self.w_max=tf.expand_dims(tf.reduce_max(self.w+(1-self.memory_mask)*(-1000),1),1)
            #(batch_size, memory_k, 1)
            self.z=tf.exp((self.w-self.w_max)*self.memory_mask)/tf.expand_dims(tf.reduce_sum(tf.exp((self.w-self.w_max)*self.memory_mask)*self.memory_mask,1),1)*self.memory_mask
            #(batch_size, hidden_units)
            self.o=tf.reduce_sum(self.updated_M*self.z,1)
            #(batch_size, memory_H, hidden_units)
            read_out=tf.concat((read_out,tf.expand_dims(self.o,1)),axis=1)
        #(1, memory_H)
        alpha=tf.nn.softmax(logits=self.params['alpha'],axis=1)
        #(batch_size, 1, memory_H)
        self.read_out_atten=tf.tile(tf.expand_dims(alpha,0),[tf.shape(read_out)[0],1,1])  #(1, 1, memory_H) 第0维扩展为batch_size
        #(batch_size, hidden_units)
        self.p=tf.squeeze(tf.matmul(self.read_out_atten,read_out),axis=1)

        """
        Fusion Gating Mechanism
        """
        #(batch_size, hidden_units)
        fusion_gate=tf.matmul(self.x,self.params['fusionW'])+tf.matmul(self.p,self.params['fusionV'])
        #(batch_size,hidden_units)
        fusion_gate=tf.nn.sigmoid(fusion_gate) #值域(0,1)
        #(batch_size, hidden_units)
        self.intent=fusion_gate*self.x+(1-fusion_gate)*self.p

        """
        Document Ranker
        """
        #(batch_size, query_encoder_units*2 + hidden_units)
        q_intent=tf.concat((self.q_vec,self.intent),1)
        with tf.variable_scope('projection_layer1'):
            #(batch_size, doc_encoder_units*2)
            q_intent=tf.layers.dense(inputs=q_intent,use_bias=True,units=self.doc_encoder_units*2,activation=tf.nn.tanh)
        #(batch_size, doc_encoder_units*2, 1)
        q_intent=tf.expand_dims(q_intent,2)
        #(batch_size, candid_doc_num)
        self.click_prob=tf.squeeze(tf.nn.sigmoid(tf.matmul(self.d_vec,q_intent)),axis=2) #测试时，用于计算文档排序评估指标MAP,MRR,NDCG

        #(batch_size, candid_doc_num)
        self.y=tf.cast(self.y,dtype=tf.float32)
        self.loss_1=tf.reduce_mean(-1*(self.y*tf.log(self.click_prob+1e-10)+(1-self.y)*tf.log(1-self.click_prob+1e-10))) #所有元素的均值

        """
        Query Recommender
        """
        #(batch_size, max_next_q_length+1)
        self.next_q=tf.placeholder(tf.int32,[None, None]) #添加<sos>和<eos>
        #(batch_size)
        self.next_q_len=tf.placeholder(tf.int32,[None]) #添加<eos>但不添加<sos>的查询长度

        with tf.variable_scope('projection_layer2'):
            #(batch_size, decoder_units)
            decoder_h0=tf.layers.dense(inputs=self.intent,use_bias=True,units=self.decoder_units,activation=tf.nn.tanh)
        max_next_q_length=tf.shape(self.next_q)[1]-1 #添加<eos>但不添加<sos>的最大查询长度，为self.next_q_len中的最大值

        #制造decoder输入和输出的错位
        b_s=tf.shape(decoder_h0)[0]
        #(batch_size, max_next_q_length)
        decoder_input=tf.strided_slice(self.next_q,[0,0],[b_s,-1],[1,1])
        #(batch_size, max_next_q_length)
        decoder_output=tf.strided_slice(self.next_q,[0,1],[b_s,max_next_q_length+1],[1,1])

        #(batch_size, max_next_q_length, emb_dim)
        decoder_input_emb=tf.nn.embedding_lookup(self.params['Wemb'],decoder_input)

        output_layer=Dense(self.vocab_size,kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))  # 不添加softmax激活函数
        decoder_cell=tf.contrib.rnn.GRUCell(self.decoder_units)

        with tf.variable_scope('decoder'):
            training_LuongAttention=tf.contrib.seq2seq.LuongAttention(num_units=self.decoder_units,memory=q_hiddens,memory_sequence_length=self.q_len)  #q_hiddens:(batch_size, max_query_len, query_encoder_units*2)
            training_atten_cell=tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell,attention_mechanism=training_LuongAttention,attention_layer_size=self.decoder_units,alignment_history=False,output_attention=True)
            training_atten_state=training_atten_cell.zero_state(b_s,tf.float32).clone(cell_state=decoder_h0)
            training_helper=tf.contrib.seq2seq.TrainingHelper(inputs=decoder_input_emb,sequence_length=self.next_q_len,time_major=False) #sequence_length: int32
            training_decoder=tf.contrib.seq2seq.BasicDecoder(cell=training_atten_cell,helper=training_helper,initial_state=training_atten_state,output_layer=output_layer)
            #training_decoder_outputs.rnn_output:(batch_size, max_next_q_length, vocab_size)
            #training_decoder_outputs.sample_id:(batch_size, max_next_q_length)
            #training_decoder_state.cell_state:(batch_size, decoder_units)
            #training_decoder_seq_len:(batch_size) 与next_q_len相同
            training_decoder_outputs,training_decoder_state,training_decoder_seq_len=tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
                                                                                                                       output_time_major=False, #batch major
                                                                                                                       impute_finished=True,
                                                                                                                       maximum_iterations=max_next_q_length) #maximum_iterations: int32

        #(batch_size, max_next_q_length, vocab_size)
        training_logits=tf.identity(training_decoder_outputs.rnn_output)
        #(batch_size, max_next_q_length)
        decoder_mask=tf.sequence_mask(self.next_q_len,max_next_q_length,dtype=tf.float32)
        self.loss_2=tf.contrib.seq2seq.sequence_loss(logits=training_logits,targets=decoder_output,weights=decoder_mask,average_across_timesteps=True,average_across_batch=True)

        #reg=tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.re_lambda), tf.trainable_variables()) #所有参数矩阵元素的平方和*lambda/2
        #self.L2=reg/tf.cast(b_s,dtype=tf.float32)
        #self.loss=self.loss_1+self.loss_2+self.L2
        self.loss=self.loss_1+self.loss_2

        optimizer=tf.train.AdamOptimizer(self.learning_rate)
        gradients=optimizer.compute_gradients(self.loss)
        self.capped_gradients=[(tf.clip_by_value(grad,-5.,5.),var) for grad,var in gradients if grad is not None]
        self.train_op=optimizer.apply_gradients(self.capped_gradients)
        self.saver=tf.train.Saver(max_to_keep=1) #只保存最后一代的模型

        self.maximum_iterations=tf.placeholder(tf.int32,[])

        with tf.variable_scope('decoder',reuse=True):
            predicting_LuongAttention=tf.contrib.seq2seq.LuongAttention(num_units=self.decoder_units,memory=q_hiddens,memory_sequence_length=self.q_len)  #q_hiddens:(batch_size, max_query_len, query_encoder_units*2)
            predicting_atten_cell=tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell,attention_mechanism=predicting_LuongAttention,attention_layer_size=self.decoder_units,alignment_history=False, output_attention=True)
            predicting_atten_state=predicting_atten_cell.zero_state(b_s,tf.float32).clone(cell_state=decoder_h0)
            #(batch_size)
            start_tokens=tf.tile(tf.constant([self.vocab['<sos>']],dtype=tf.int32),[b_s])
            predicting_helper=tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.params['Wemb'],start_tokens=start_tokens,end_token=self.vocab['<eos>']) #start_tokens/end_token: int32
            predicting_decoder=tf.contrib.seq2seq.BasicDecoder(cell=predicting_atten_cell,helper=predicting_helper,initial_state=predicting_atten_state,output_layer=output_layer)
            #predicting_decoder_outputs.rnn_output:(batch_size, predicting_decoder_seq_len_max, vocab_size) predicting_decoder_seq_len_max是指predicting_decoder_seq_len中元素的最大值
            #predicting_decoder_outputs.sample_id:(batch_size, predicting_decoder_seq_len_max)
            #predicting_decoder_state.cell_state:(batch_size, decoder_units)
            #predicting_decoder_seq_len:(batch_size)
            predicting_decoder_outputs,predicting_decoder_state,predicting_decoder_seq_len=tf.contrib.seq2seq.dynamic_decode(decoder=predicting_decoder,
                                                                                                                             output_time_major=False,
                                                                                                                             impute_finished=True,
                                                                                                                             maximum_iterations=self.maximum_iterations)

        #(batch_size, predicting_decoder_seq_len_max)
        self.predicting_ids=tf.identity(predicting_decoder_outputs.sample_id) #测试时，用于计算查询建议评估指标BLEU
        #(batch_size)
        self.predicting_len=predicting_decoder_seq_len #包含<eos>

        #(batch_size, candid_query_num, max_candid_next_q_length+1)
        self.candid_next_q=tf.placeholder(tf.int32,[None, self.candid_query_num, None]) #添加<sos>和<eos>
        #(batch_size, candid_query_num)
        self.candid_next_q_len=tf.placeholder(tf.int32,[None, self.candid_query_num]) #添加<eos>但不添加<sos>的候选查询长度

        #(batch_size, candid_query_num, decoder_units)
        candid_decoder_h0=tf.tile(tf.expand_dims(decoder_h0,1),[1,self.candid_query_num,1])
        #(batch_size x candid_query_num, decoder_units)
        candid_decoder_h0=tf.reshape(candid_decoder_h0,[-1,self.decoder_units])
        max_candid_next_q_length=tf.shape(self.candid_next_q)[2]-1 #添加<eos>但不添加<sos>的最大候选查询长度，为self.candid_next_q_len中的最大值

        b_c=tf.shape(candid_decoder_h0)[0]
        #(batch_size x candid_query_num, max_candid_next_q_length+1)
        candid_next_query=tf.reshape(self.candid_next_q,[-1,max_candid_next_q_length+1])
        #(batch_size x candid_query_num, max_candid_next_q_length)
        candid_decoder_input=tf.strided_slice(candid_next_query,[0,0],[b_c,-1],[1,1])
        #(batch_size x candid_query_num, max_candid_next_q_length)
        candid_decoder_output=tf.strided_slice(candid_next_query,[0,1],[b_c,max_candid_next_q_length+1],[1,1])

        #(batch_size x candid_query_num, max_candid_next_q_length, emb_dim)
        candid_decoder_input_emb=tf.nn.embedding_lookup(self.params['Wemb'],candid_decoder_input)

        #(batch_size, candid_query_num, max_query_len, query_encoder_units*2)
        candid_q_hiddens=tf.tile(tf.expand_dims(q_hiddens,1),[1,self.candid_query_num,1,1])
        #(batch_size x candid_query_num, max_query_len, query_encoder_units*2)
        candid_q_hiddens=tf.reshape(candid_q_hiddens,[-1,self.max_query_len,self.query_encoder_units*2])
        #(batch_size, candid_query_num)
        candid_q_len=tf.tile(tf.expand_dims(self.q_len,1),[1,self.candid_query_num])
        #(batch_size x candid_query_num)
        candid_q_len=tf.reshape(candid_q_len,[-1])

        with tf.variable_scope('decoder',reuse=True):
            testing_LuongAttention=tf.contrib.seq2seq.LuongAttention(num_units=self.decoder_units,memory=candid_q_hiddens,memory_sequence_length=candid_q_len)
            testing_atten_cell=tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell,attention_mechanism=testing_LuongAttention,attention_layer_size=self.decoder_units,alignment_history=False, output_attention=True)
            testing_atten_state=testing_atten_cell.zero_state(b_c,tf.float32).clone(cell_state=candid_decoder_h0)
            testing_helper=tf.contrib.seq2seq.TrainingHelper(inputs=candid_decoder_input_emb,sequence_length=tf.reshape(self.candid_next_q_len,[-1]),time_major=False) #sequence_length: int32
            testing_decoder=tf.contrib.seq2seq.BasicDecoder(cell=testing_atten_cell,helper=testing_helper,initial_state=testing_atten_state,output_layer=output_layer)
            #testing_decoder_outputs.rnn_output:(batch_size x candid_query_num, max_candid_next_q_length, vocab_size)
            #testing_decoder_outputs.sample_id:(batch_size x candid_query_num, max_candid_next_q_length)
            #testing_decoder_state.cell_state:(batch_size x candid_query_num, decoder_units)
            #testing_decoder_seq_len:(batch_size x candid_query_num)
            testing_decoder_outputs,testing_decoder_state,testing_decoder_seq_len=tf.contrib.seq2seq.dynamic_decode(decoder=testing_decoder,
                                                                                                                    output_time_major=False, #batch major
                                                                                                                    impute_finished=True,
                                                                                                                    maximum_iterations=max_candid_next_q_length) #maximum_iterations: int32

        #(batch_size x candid_query_num, max_candid_next_q_length, vocab_size)
        testing_logits=tf.identity(testing_decoder_outputs.rnn_output)
        #(batch_size x candid_query_num, max_candid_next_q_length, vocab_size)
        word_prob=tf.nn.softmax(logits=testing_logits,axis=2)

        #(batch_size x candid_query_num, max_candid_next_q_length, 2)
        self.idx=tf.placeholder(tf.int32,[None, None, 2])
        #(batch_size x candid_query_num, max_candid_next_q_length, 3)
        target_word_indices=tf.concat((self.idx,tf.expand_dims(candid_decoder_output,2)),axis=2)
        #(batch_size x candid_query_num, max_candid_next_q_length)
        target_word_prob=tf.gather_nd(params=word_prob,indices=target_word_indices)
        #(batch_size, candid_query_num, max_candid_next_q_length)
        target_word_prob=tf.reshape(target_word_prob,[-1,self.candid_query_num,max_candid_next_q_length])

        #(batch_size, candid_query_num, max_candid_next_q_length)
        target_word_mask=tf.sequence_mask(self.candid_next_q_len,max_candid_next_q_length,dtype=tf.float32)
        #(batch_size, candid_query_num)
        self.candid_query_score=tf.reduce_sum(target_word_prob*target_word_mask,axis=2)/tf.cast(self.candid_next_q_len,dtype=tf.float32) #测试时，用于计算查询建议评估指标MRR


    def init_weights(self,i_name,shape):
        sigma=np.sqrt(2./shape[0])
        return tf.get_variable(name=i_name,dtype=tf.float32,initializer=tf.random_normal(shape)*sigma) #正态分布初始化

    def init_params(self):
        params=dict()
        #word embedding
        const_init=tf.constant_initializer(self.pretrained_weight)
        params['Wemb']=tf.get_variable(name='Wemb',shape=[self.vocab_size,self.emb_dim],dtype=tf.float32,initializer=const_init)
        #transformation matrix
        params['Rmatrix']=self.init_weights('Rmatrix',(self.memory_H-1,self.hidden_units,self.hidden_units))
        #attention vector
        params['alpha']=self.init_weights('alpha',(1,self.memory_H))
        #fusion weights
        params['fusionW']=self.init_weights('fusionW',(self.hidden_units,self.hidden_units))
        params['fusionV']=self.init_weights('fusionV',(self.hidden_units,self.hidden_units))
        return params

    def get_memory_input(self,session_num):
        batch_size=session_num.shape[0]
        indices=np.zeros((batch_size,self.memory_k,2),dtype=np.int32)
        slots_num=np.zeros((batch_size),dtype=np.int32)
        for i in range(batch_size):
            indices[i,:,0]=i
            if session_num[i]>=self.memory_k:
                indices[i,:,1]=np.arange(session_num[i]+1-self.memory_k,session_num[i]+1)
                slots_num[i]=self.memory_k
            else:
                indices[i,:session_num[i],1]=np.arange(1,session_num[i]+1)
                slots_num[i]=session_num[i]
        return indices,slots_num

    def get_test_input(self,candid_next_q):
        batch_size=candid_next_q.shape[0]
        max_candid_next_q_length=candid_next_q.shape[2]-1
        idx=np.zeros((batch_size*self.candid_query_num,max_candid_next_q_length,2),dtype=np.int32)
        for i in range(batch_size*self.candid_query_num):
            idx[i,:,0]=i
        for j in range(max_candid_next_q_length):
            idx[:,j,1]=j
        return idx


