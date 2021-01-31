#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import main, helper, os, dataload
from model import LostNet
import tensorflow as tf
import multi_bleu
from rank_metrics import mean_average_precision, NDCG, MRR

args=main.get_args()

def test(model,test_dataset,dictionary,sess):
    batches_idx=helper.get_batches_idx(len(test_dataset),args.batch_size)
    print('number of test batches = ',len(batches_idx))

    num_batches=len(batches_idx)
    predicts, targets = [], []
    map, mrr, ndcg_1, ndcg_3, ndcg_5, ndcg_10 = 0, 0, 0, 0, 0, 0
    for batch_no in range(1,num_batches+1): #1,...,num_batches
        batch_idx=batches_idx[batch_no-1]
        batch_data=[test_dataset.dataset[i] for i in batch_idx]

        #将一批数据转换为模型输入的格式
        (hist_query_input, hist_doc_input, session_num, hist_query_num, hist_query_len, hist_click_num, hist_doc_len,
        cur_query_input, cur_doc_input, cur_query_num, cur_query_len, cur_click_num, cur_doc_len,
        query, q_len, doc, d_len, y, next_q, next_q_len, maximum_iterations)=helper.batch_to_tensor(batch_data, args.max_query_len, args.max_doc_len)

        indices,slots_num=model.get_memory_input(session_num)

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
            model.maximum_iterations: maximum_iterations}

        click_prob_, predicting_ids_, predicting_len_=sess.run([model.click_prob,model.predicting_ids,model.predicting_len],feed_dict=feed_dict)

        map += mean_average_precision(click_prob_,y)
        mrr += MRR(click_prob_,y)
        ndcg_1 += NDCG(click_prob_,y,1)
        ndcg_3 += NDCG(click_prob_,y,3)
        ndcg_5 += NDCG(click_prob_,y,5)
        ndcg_10 += NDCG(click_prob_,y,10)

        batch_predicting_text = helper.generate_predicting_text(predicting_ids_,predicting_len_,dictionary)
        batch_target_text, batch_query_text = helper.generate_target_text(batch_data,dictionary,args.max_query_len)
        predicts += batch_predicting_text
        targets += batch_target_text

    map = map / num_batches
    mrr = mrr / num_batches
    ndcg_1 = ndcg_1 / num_batches
    ndcg_3 = ndcg_3 / num_batches
    ndcg_5 = ndcg_5 / num_batches
    ndcg_10 = ndcg_10 / num_batches

    print('MAP - ', map)
    print('MRR - ', mrr)
    print('NDCG@1 - ', ndcg_1)
    print('NDCG@3 - ', ndcg_3)
    print('NDCG@5 - ', ndcg_5)
    print('NDCG@10 - ', ndcg_10)

    print("targets size = ", len(targets))
    print("predicts size = ", len(predicts))

    multi_bleu.print_multi_bleu(predicts, targets)


if __name__=="__main__":

    #load dictionary
    dictionary=helper.load_object(args.data_path+args.dataset+'/dictionary.p')
    print('vocabulary size = ',len(dictionary))

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

    #build pretrained weight
    pretrained_weight=helper.init_embedding_weights(dictionary,args.emb_dim) #不利用预训练参数进行测试，使用载入的训练参数进行测试

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
    config=tf.ConfigProto(allow_soft_placement=True) #允许tf自动选择一个存在并且可用的设备来运行操作
    config.gpu_options.allow_growth = True #tf在运行过程中动态申请显存

    with tf.Session(config=config) as sess:
        #load model params
        model_save_path=args.save_path+args.dataset+'/model/'
        model.saver.restore(sess,tf.train.latest_checkpoint(model_save_path))

        #test model
        test(model,test_dataset,dictionary, sess)

