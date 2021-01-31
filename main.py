#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os,dataload,helper,train_dev
import tensorflow as tf
from model import LostNet
from argparse import ArgumentParser

def get_args():
    parser=ArgumentParser(description='Long short-term session search, Network (LostNet)')

    parser.add_argument('--max_words', type=int, default=100000,
                        help='maximum number of words in vocabulary')
    parser.add_argument('--filter_frequency', action='store_true',
                        help='filter words by frequency')  #按照单词出现的频次进行过滤
    parser.add_argument('--frequency_limit', type=int, default=10,
                        help='limit of words frequency in vocabulary')  #保留频次>frequency_limit的单词（大于但不等于）
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimension of word embeddings')
    parser.add_argument('--max_query_len', type=int, default=10,
                        help='maximum length of a query')
    parser.add_argument('--max_doc_len', type=int, default=20,
                        help='maximum length of a document')
    parser.add_argument('--hidden_units', type=int, default=512,
                        help='number of gru hidden units')
    parser.add_argument('--atten_hidden_units', type=int, default=256,
                        help='number of hidden units for attention layer')
    parser.add_argument('--memory_k', type=int, default=32,
                        help='number of slots in personalized user memory')
    parser.add_argument('--memory_H', type=int, default=3,
                        help='number of reading hops in personalized user memory')
    parser.add_argument('--decoder_units', type=int, default=512,
                        help='number of gru hidden units for next query decoder')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--re_lambda', type=float, default=1e-4,
                        help='L2 regularization coefficient')
    parser.add_argument('--candid_query_num', type=int, default=20,
                        help='number of candidate query suggestions')
    parser.add_argument('--dataset', type=str, default='aol',
                        help='type of the experimental dataset')
    parser.add_argument('--corpus_path', type=str, default='./corpus/',
                        help='location of the original corpus')
    parser.add_argument('--data_path', type=str, default='./data/',
                        help='location of the loaded data')
    parser.add_argument('--save_path', type=str, default='./outputs/',
                        help='path to save the final model and results')
    parser.add_argument('--hist_session_num_limit', type=int, default=32,
                        help='limit the number of historical sessions')
    parser.add_argument('--click_num_limit', type=int, default=5,
                        help='limit the number of clicked documents for a query')
    parser.add_argument('--max_example', type=int, default=-1,
                        help='number of training examples (-1 = all examples)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='upper limit of epoch')
    parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                        help='batch size')
    parser.add_argument('--print_every', type=int, default=1000,
                        help='training report interval')
    parser.add_argument('--plot_every', type=int, default=500,
                        help='plotting interval')
    parser.add_argument('--early_stop', type=int, default=3,
                        help='early stopping criterion')

    parser.add_argument('--query_encoder_units', type=int, default=256,
                        help='number of hidden units for query encoder')
    parser.add_argument('--query_atten_units', type=int, default=256,
                        help='number of hidden units for query inner attention')
    parser.add_argument('--doc_encoder_units', type=int, default=256,
                        help='number of hidden units for document encoder')
    parser.add_argument('--doc_atten_units', type=int, default=256,
                        help='number of hidden units for document inner attention')

    args=parser.parse_args()
    return args

def main():
    args=get_args()
    if args.dataset=='sogou':
        if args.frequency_limit == 10: args.frequency_limit = 5
        if args.memory_k == 32: args.memory_k = 16
        if args.hist_session_num_limit == 32: args.hist_session_num_limit = 16

    #create directories
    if not os.path.exists(args.data_path + args.dataset):
        os.makedirs(args.data_path + args.dataset)
    if not os.path.exists(args.save_path + args.dataset):
        os.makedirs(args.save_path + args.dataset)

    if not os.path.exists(args.data_path+args.dataset+'/dictionary.p'):
        #build dictionary
        dictionary=dataload.Dictionary()
        dictionary.build_dict(args.corpus_path+args.dataset+'/train.txt',args.max_words,args.filter_frequency,args.frequency_limit)
        dictionary.load_word_embedding(args.corpus_path+args.dataset+'/glove.300d.txt')
        print('vocabulary size = ',len(dictionary))
        #save the dictionary object
        helper.save_object(dictionary, args.data_path+args.dataset+'/dictionary.p')
    else:
        #load dictionary
        dictionary=helper.load_object(args.data_path+args.dataset+'/dictionary.p')
        print('vocabulary size = ', len(dictionary))

    if not os.path.exists(args.data_path+args.dataset+'/train_dataset.p'):
        #build train dataset
        train_dataset=dataload.Dataset(args.max_query_len,args.max_doc_len,args.hist_session_num_limit,args.click_num_limit)
        train_dataset.parse(args.corpus_path+args.dataset+'/train.txt',dictionary,args.max_example)
        print('train set size = ',len(train_dataset))
        #save the train_dataset object
        helper.save_object(train_dataset,args.data_path+args.dataset+'/train_dataset.p')
    else:
        #load train dataset
        train_dataset=helper.load_object(args.data_path+args.dataset+'/train_dataset.p')
        print('train set size = ',len(train_dataset))

    if not os.path.exists(args.data_path+args.dataset+'/dev_dataset.p'):
        #build dev dataset
        dev_dataset=dataload.Dataset(args.max_query_len,args.max_doc_len,args.hist_session_num_limit,args.click_num_limit)
        dev_dataset.parse(args.corpus_path+args.dataset+'/dev.txt',dictionary,args.max_example)
        print('development set size = ',len(dev_dataset))
        #save the dev_dataset object
        helper.save_object(dev_dataset,args.data_path+args.dataset+'/dev_dataset.p')
    else:
        #load dev dataset
        dev_dataset=helper.load_object(args.data_path+args.dataset+'/dev_dataset.p')
        print('development set size = ',len(dev_dataset))

    #build pretrained weight
    pretrained_weight=helper.init_embedding_weights(dictionary,args.emb_dim)

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
    config.gpu_options.allow_growth=True #tf在运行过程中动态申请显存

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        #train model
        train_obj=train_dev.Train(model, args, sess, dictionary)
        train_obj.train_epochs(train_dataset, dev_dataset)


if __name__ == '__main__':
    main()