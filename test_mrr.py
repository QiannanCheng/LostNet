#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import json, random, os
from model import LostNet
import tensorflow as tf
import dataload, main, helper
from collections import Counter
from rank_metrics import MRR

args=main.get_args()

def test_mrr(model,filter_test_dataset,anchor_candidates,sess):
    batches_idx=helper.get_batches_idx(len(filter_test_dataset),args.batch_size)
    print('number of mrr test batches = ', len(batches_idx))

    num_batches=len(batches_idx)
    mrr=0
    for batch_no in range(1,num_batches+1):  #1,...,num_batches
        batch_idx=batches_idx[batch_no-1]
        batch_data=[filter_test_dataset.dataset[i] for i in batch_idx]

        #将一批数据转换为模型输入的格式
        (hist_query_input, hist_doc_input, session_num, hist_query_num, hist_query_len, hist_click_num, hist_doc_len,
        cur_query_input, cur_doc_input, cur_query_num, cur_query_len, cur_click_num, cur_doc_len,
        query, q_len, doc, d_len, y, next_q, next_q_len, maximum_iterations)=helper.batch_to_tensor(batch_data,args.max_query_len,args.max_doc_len)

        indices,slots_num=model.get_memory_input(session_num)

        candid_next_q,candid_next_q_len,label=helper.get_candid_tensor(batch_data,anchor_candidates,args.candid_query_num)

        idx=model.get_test_input(candid_next_q)

        feed_dict={
            model.hist_query_input: hist_query_input,
            model.hist_doc_input: hist_doc_input,
            model.session_num: session_num,
            model.hist_query_num: hist_query_num,
            model.hist_query_len: hist_query_len,
            model.hist_click_num: hist_click_num,
            model.hist_doc_len: hist_doc_len,
            model.cur_query_input: cur_query_input,
            model.cur_doc_input: cur_doc_input,
            model.cur_query_num: cur_query_num,
            model.cur_query_len: cur_query_len,
            model.cur_click_num: cur_click_num,
            model.cur_doc_len: cur_doc_len,
            model.q: query,
            model.q_len: q_len,
            model.d: doc,
            model.d_len: d_len,
            model.indices: indices,
            model.slots_num: slots_num,
            model.candid_next_q: candid_next_q,
            model.candid_next_q_len: candid_next_q_len,
            model.idx: idx}

        candid_query_score_=sess.run(model.candid_query_score,feed_dict=feed_dict)
        mrr += MRR(candid_query_score_, label)

    mrr = mrr / num_batches
    print('Query Suggestion MRR - ', mrr)


if __name__=="__main__":

    #load dictionary
    dictionary=helper.load_object(args.data_path+args.dataset+'/dictionary.p')
    print('vocabulary size = ', len(dictionary))

    if not os.path.exists(args.data_path+args.dataset+'/anchor_candidates.p'):
        #build anchor candidates
        anchor_queries={} #{'anchor_query':set(target_next_queries),...}
        with open(args.corpus_path+args.dataset+'/test.txt','r') as f:
            for line in f:
                user=json.loads(line.strip()) #一个用户的数据
                for session in user: #一个用户的每一个session
                    anchor_query=session['query'][-2] #一个session的倒数第二个query
                    target_next_query=session['query'][-1] #一个session的倒数第一个query
                    if anchor_query not in anchor_queries:
                        anchor_queries[anchor_query]=set()
                    anchor_queries[anchor_query].add(target_next_query)

        anchor_queries_count={} #{'anchor_query':Counter(),...}
        all_queries=set()
        with open(args.corpus_path+args.dataset+'/train.txt','r') as f:
            for line in f:
                user=json.loads(line.strip()) #一个用户的数据
                for session in user: #一个用户的每一个session
                    queries=session['query']
                    all_queries.update(queries)
                    for i in range(len(queries)-1): #对于一个session中的前n-1个query
                        if queries[i] in anchor_queries:
                            if queries[i] not in anchor_queries_count:
                                anchor_queries_count[queries[i]]=Counter()
                            anchor_queries_count[queries[i]].update([queries[i+1]])

        anchor_candidates={} #{'anchor_query':[candid_next_query1,candid_next_query2,...],...} candid_next_query -> Sentence(tag=True)
        for anchor_query,target_next_queries in anchor_queries.items():
            if anchor_query in anchor_queries_count:
                count=anchor_queries_count[anchor_query]
                for target_next_query in target_next_queries:
                    if target_next_query in count:
                        count.pop(target_next_query)
                most_common=count.most_common(args.candid_query_num-1) #[('next_query1',count1),('next_query2',count2),...]
                most_common_next_queries=[tup[0] for tup in most_common]
                if len(most_common_next_queries)<args.candid_query_num-1:
                    #随机samaple候选查询建议
                    valid_queries=all_queries.difference(set(most_common_next_queries)).difference(set(target_next_queries)) #set
                    extract_next_queries=random.sample(valid_queries,args.candid_query_num-1-len(most_common_next_queries)) #list
                    candid_next_queries=most_common_next_queries+extract_next_queries
                else:
                    candid_next_queries=most_common_next_queries
            else:
                valid_queries=all_queries.difference(set(target_next_queries)) #set
                extract_next_queries=random.sample(valid_queries,args.candid_query_num-1) #list
                candid_next_queries=extract_next_queries

            if anchor_query not in anchor_candidates:
                anchor_candidates[anchor_query]=[]
            random.shuffle(candid_next_queries)
            for candid_next_query in candid_next_queries:
                candid_sen=dataload.Sentence(tag=True)
                candid_sen.sen2seq(candid_next_query,dictionary,args.max_query_len)
                anchor_candidates[anchor_query].append(candid_sen)

        print('anchor queries size = ', len(anchor_candidates)) #输出测试集中anchor queries的数目
        #save anchor candidates object
        helper.save_object(anchor_candidates,args.data_path+args.dataset+'/anchor_candidates.p')
    else:
        #load anchor candidates
        anchor_candidates=helper.load_object(args.data_path+args.dataset+'/anchor_candidates.p')
        print('anchor queries size = ', len(anchor_candidates))

    if not os.path.exists(args.data_path+args.dataset+'/test_dataset.p'):
        #build test dataset
        test_dataset=dataload.Dataset(args.max_query_len,args.max_doc_len,args.hist_session_num_limit,args.click_num_limit)
        test_dataset.parse(args.corpus_path+args.dataset+'/test.txt',dictionary,args.max_example)
        print('test set size = ',len(test_dataset))
        #save the test_dataset object
        helper.save_object(test_dataset,args.data_path+args.dataset+'/test_dataset.p')
    else:
        #load test dataset
        test_dataset=helper.load_object(args.data_path+args.dataset+'/test_dataset.p')
        print('test set size = ',len(test_dataset))

    #build filter test dataset
    filter_test_dataset=dataload.Dataset(args.max_query_len,args.max_doc_len,args.hist_session_num_limit,args.click_num_limit)
    for data in test_dataset.dataset:
        if data.anchor_tag: #如果当前查询为当前session的倒数第二个查询
            filter_test_dataset.dataset.append(data)
    print('filter test set size = ', len(filter_test_dataset))

    #build pretrained weight
    pretrained_weight=helper.init_embedding_weights(dictionary,args.emb_dim)  #不利用预训练参数进行测试，使用载入的训练参数进行测试

    #build model
    model=LostNet(
        vocab_size=len(dictionary),
        emb_dim=args.emb_dim,
        max_query_len=args.max_query_len,
        max_doc_len=args.max_doc_len,
        query_encoder_units=args.query_encoder_units,
        query_atten_units=args.query_atten_units,
        doc_encoder_units=args.doc_encoder_units,
        doc_atten_units=args.doc_atten_units,
        hidden_units=args.hidden_units,
        atten_hidden_units=args.atten_hidden_units,
        memory_k=args.memory_k,
        memory_H=args.memory_H,
        decoder_units=args.decoder_units,
        vocab=dictionary.word2idx,
        learning_rate=args.learning_rate,
        candid_query_num=args.candid_query_num,
        pretrained_weight=pretrained_weight,
        re_lambda=args.re_lambda
    )

    #os.environ['CUDA_VISIBLE_DEVICES']='0' #指定gpu运行
    config=tf.ConfigProto(allow_soft_placement=True)  #允许tf自动选择一个存在并且可用的设备来运行操作
    config.gpu_options.allow_growth=True  #tf在运行过程中动态申请显存

    with tf.Session(config=config) as sess:
        #load model params
        model_save_path=args.save_path+args.dataset+'/model/'
        model.saver.restore(sess,tf.train.latest_checkpoint(model_save_path))

        #test model
        test_mrr(model, filter_test_dataset, anchor_candidates, sess)

