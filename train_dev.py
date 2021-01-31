#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import time, helper, sys, os
import multi_bleu
from rank_metrics import mean_average_precision, NDCG, MRR
import math

class Train:
    def __init__(self,model,args,sess,dictionary):
        self.model=model
        self.args=args
        self.best_dev_metrics=0
        self.times_no_improvement=0
        self.stop=False
        self.train_losses=[]
        self.dev_metrics=[]
        self.sess=sess
        self.dictionary=dictionary

    def train_epochs(self,train_dataset,dev_dataset):
        for epoch in range(self.args.epochs): #0,...,epochs-1
            if not self.stop:
                print('\nTraining: Epoch '+str(epoch+1))
                self.train(train_dataset)
                #training epoch completes, now do validation
                print('\nValidating: Epoch '+str(epoch + 1))
                metrics_sum=self.validate(dev_dataset)
                self.dev_metrics.append(metrics_sum)
                #save model if metrics sum goes up
                if self.best_dev_metrics < metrics_sum:
                    self.best_dev_metrics=metrics_sum
                    #save model
                    model_save_path=self.args.save_path+self.args.dataset+'/model/'
                    if not os.path.exists(model_save_path):
                        os.makedirs(model_save_path)
                    self.model.saver.save(self.sess,model_save_path+'lostnet.ckpt')
                    self.times_no_improvement=0
                else:
                    self.times_no_improvement+=1
                    #no improvement in validation metrics for last n iterations, so stop training
                    if self.times_no_improvement==self.args.early_stop:
                        self.stop=True
                #save the train loss plot
                helper.save_plot(self.train_losses, self.args.save_path+self.args.dataset+'/', 'train_loss', epoch + 1)
                #save the dev metrics plot
                helper.save_plot(self.dev_metrics, self.args.save_path+self.args.dataset+'/', 'dev_metrics', epoch + 1)
            else:
                break

    def train(self,train_dataset):
        batches_idx=helper.get_batches_idx(len(train_dataset),self.args.batch_size)
        print('number of train batches = ',len(batches_idx))

        start=time.time()
        print_loss_total=0
        plot_loss_total=0

        num_batches=len(batches_idx)
        for batch_no in range(1,num_batches+1): #1,...num_batches
            batch_idx=batches_idx[batch_no-1]
            batch_data=[train_dataset.dataset[i] for i in batch_idx]

            #将一批数据转换为模型输入的格式
            (hist_query_input, hist_doc_input, session_num, hist_query_num, hist_query_len, hist_click_num, hist_doc_len,
            cur_query_input, cur_doc_input, cur_query_num, cur_query_len, cur_click_num, cur_doc_len,
            query, q_len, doc, d_len, y, next_q, next_q_len, _)=helper.batch_to_tensor(batch_data,self.args.max_query_len,self.args.max_doc_len)

            indices,slots_num=self.model.get_memory_input(session_num)
            feed_dict={
                self.model.hist_query_input: hist_query_input,
                self.model.hist_doc_input: hist_doc_input,
                self.model.session_num: session_num,
                self.model.hist_query_num: hist_query_num,
                self.model.hist_query_len: hist_query_len,
                self.model.hist_click_num: hist_click_num,
                self.model.hist_doc_len: hist_doc_len,
                self.model.cur_query_input: cur_query_input,
                self.model.cur_doc_input: cur_doc_input,
                self.model.cur_query_num: cur_query_num,
                self.model.cur_query_len: cur_query_len,
                self.model.cur_click_num: cur_click_num,
                self.model.cur_doc_len: cur_doc_len,
                self.model.q: query,
                self.model.q_len: q_len,
                self.model.d: doc,
                self.model.d_len: d_len,
                self.model.y: y,  # 0/1
                self.model.indices: indices,
                self.model.slots_num: slots_num,
                self.model.next_q: next_q,
                self.model.next_q_len: next_q_len}

            #计算loss + 优化参数
            loss_=self.sess.run(self.model.loss,feed_dict=feed_dict)
            train_op_=self.sess.run(self.model.train_op,feed_dict=feed_dict)

            print_loss_total+=loss_
            plot_loss_total+=loss_

            if batch_no % self.args.print_every==0:
                print_loss_avg=print_loss_total/self.args.print_every
                print_loss_total=0
                print('%s (%d %d%%) %.4f' % (
                    helper.show_progress(start, batch_no / num_batches), batch_no,
                    batch_no / num_batches * 100, print_loss_avg))

            if batch_no % self.args.plot_every==0:
                plot_loss_avg=plot_loss_total/self.args.plot_every
                self.train_losses.append(plot_loss_avg)
                plot_loss_total=0

    def validate(self, dev_dataset):
        batches_idx=helper.get_batches_idx(len(dev_dataset), self.args.batch_size)
        print('number of dev batches = ', len(batches_idx))

        num_batches=len(batches_idx)
        predicts, targets = [], []
        map, mrr, ndcg_1, ndcg_3, ndcg_5, ndcg_10 = 0, 0, 0, 0, 0, 0
        for batch_no in range(1,num_batches + 1):  #1,...,num_batches
            batch_idx=batches_idx[batch_no-1]
            batch_data=[dev_dataset.dataset[i] for i in batch_idx]

            #将一批数据转换为模型输入的格式
            (hist_query_input, hist_doc_input, session_num, hist_query_num, hist_query_len, hist_click_num, hist_doc_len,
            cur_query_input, cur_doc_input, cur_query_num, cur_query_len, cur_click_num, cur_doc_len,
            query, q_len, doc, d_len, y, next_q, next_q_len, maximum_iterations)=helper.batch_to_tensor(batch_data,self.args.max_query_len,self.args.max_doc_len)

            indices, slots_num=self.model.get_memory_input(session_num)

            feed_dict={
                self.model.hist_query_input: hist_query_input,
                self.model.hist_doc_input: hist_doc_input,
                self.model.session_num: session_num,
                self.model.hist_query_num: hist_query_num,
                self.model.hist_query_len: hist_query_len,
                self.model.hist_click_num: hist_click_num,
                self.model.hist_doc_len: hist_doc_len,
                self.model.cur_query_input: cur_query_input,
                self.model.cur_doc_input: cur_doc_input,
                self.model.cur_query_num: cur_query_num,
                self.model.cur_query_len: cur_query_len,
                self.model.cur_click_num: cur_click_num,
                self.model.cur_doc_len: cur_doc_len,
                self.model.q: query,
                self.model.q_len: q_len,
                self.model.d: doc,
                self.model.d_len: d_len,
                self.model.indices: indices,
                self.model.slots_num: slots_num,
                self.model.maximum_iterations: maximum_iterations}

            click_prob_, predicting_ids_, predicting_len_ = self.sess.run([self.model.click_prob, self.model.predicting_ids, self.model.predicting_len], feed_dict=feed_dict)

            map += mean_average_precision(click_prob_, y)
            mrr += MRR(click_prob_, y)
            ndcg_1 += NDCG(click_prob_, y, 1)
            ndcg_3 += NDCG(click_prob_, y, 3)
            ndcg_5 += NDCG(click_prob_, y, 5)
            ndcg_10 += NDCG(click_prob_, y, 10)

            batch_predicting_text = helper.generate_predicting_text(predicting_ids_, predicting_len_, self.dictionary)
            batch_target_text, batch_query_text = helper.generate_target_text(batch_data, self.dictionary, self.args.max_query_len)

            predicts += batch_predicting_text
            targets += batch_target_text

        map = map / num_batches
        mrr = mrr / num_batches
        ndcg_1 = ndcg_1 / num_batches
        ndcg_3 = ndcg_3 / num_batches
        ndcg_5 = ndcg_5 / num_batches
        ndcg_10 = ndcg_10 / num_batches

        score, precisions, brevity_penalty, cand_tot_length, ref_closest_length = multi_bleu.multi_bleu(predicts, targets)

        metrics_sum = map + mrr + ndcg_1 + ndcg_3 + ndcg_5 + ndcg_10 + (precisions[0] + precisions[1] + precisions[2] + precisions[3])*0.01

        print('validation metrics: ')
        print('MAP = %.4f' % map)
        print('MRR = %.4f' % mrr)
        print("NDCG = {:.4f}/{:.4f}/{:.4f}/{:.4f}".format(ndcg_1, ndcg_3, ndcg_5, ndcg_10))
        print("BLEU = {:.1f}/{:.1f}/{:.1f}/{:.1f}".format(precisions[0], precisions[1], precisions[2], precisions[3]))

        return metrics_sum

