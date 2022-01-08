#coding:utf-8
import argparse
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict

import numpy as np

import faiss
import tensorflow as tf
from data_iterator import DataIterator
from tensorboardX import SummaryWriter
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-p', type=str, default='train', help='train | test')
parser.add_argument('--dataset', type=str, default='book', help='book | taobao')
parser.add_argument('--random_seed', type=int, default=19)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--num_interest', type=int, default=4)
parser.add_argument('--model_type', type=str, default='none', help='DNN | GRU4REC | ..')
parser.add_argument('--learning_rate', type=float, default=0.001, help='')
parser.add_argument('--max_iter', type=int, default=1000, help='(k)')
parser.add_argument('--patience', type=int, default=50)
parser.add_argument('--coef', default=None)
parser.add_argument('--topN', type=int, default=50)
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--con_gt_loss_weight', type=float, default=0.1)
parser.add_argument('--con_lt_loss_weight', type=float, default=0.1)
parser.add_argument('--int_gt_loss_weight', type=float, default=0.1)
parser.add_argument('--int_lt_loss_weight', type=float, default=0.1)
parser.add_argument('--dis_loss_weight', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.9)
parser.add_argument('--decay_iter', type=int, default=1000)
parser.add_argument('--cate_prop', type=float, default=0)
parser.add_argument('--neg_num', type=int, default=10)
parser.add_argument('--dis_loss_type', type=str, default='cos')

best_metric = 0

def prepare_data(src, target):
    nick_id, item_id = src
    hist_item, hist_mask = target
    return nick_id, item_id, hist_item, hist_mask

def load_item_cate(source):
    item_cate = {}
    with open(source, 'r') as f:
        for line in f:
            conts = line.strip().split(',')
            item_id = int(conts[0])
            cate_id = int(conts[1])
            item_cate[item_id] = cate_id
    return item_cate

def get_item_cate_lst(cate_file, item_count):
    item_cate = [0] * item_count
    item_cate[0] = -1
    reader = XXX # 读取额外feature，item对应的类目信息

    # 返回全量数据
    source_df = reader.to_pandas()
    source_data = source_df.values

    for li in tqdm(range(len(source_data))):
        conts = source_data[li]
        item_id = int(conts[0])
        cate_id = int(conts[1])
        item_cate[item_id] = cate_id

    return item_cate


def get_item_freq_lst(train_file, item_count):
    
    item_freq = [0] * item_count
    reader = XXX    # 读取训练集数据，计算item频次

    # 返回全量数据
    source_df = reader.to_pandas()
    source_data = source_df.values

    for li in tqdm(range(len(source_data))):
        conts = source_data[li]
        item_id = int(conts[1])
        item_freq[item_id] += 1
        
    return item_freq


def odps_load_item_cate(source):
    reader = XXX  # 读取类目数据
    # 返回全量数据
    source_df = reader.to_pandas()
    source_data = source_df.values

    item_cate = {}
    for li in tqdm(range(len(source_data))):
        conts = source_data[li]
        item_id = int(conts[0])
        cate_id = int(conts[1])
        item_cate[item_id] = cate_id
    return item_cate


def compute_diversity(item_list, item_cate_map):
    n = len(item_list)
    diversity = 0.0
    for i in range(n):
        for j in range(i+1, n):
            diversity += item_cate_map[item_list[i]] != item_cate_map[item_list[j]]
    diversity /= ((n-1) * n / 2)
    return diversity


def evaluate_full(sess, test_data, model, model_path, batch_size, item_cate_map, save=True, coef=None):
    topN = args.topN
    print('start evaluating full...Top N:', topN)
    item_embs = model.output_item(sess)

    # int在前 con在后
    print('item embs shape:', item_embs.shape)

    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    # flat_config.device = 0
    flat_config.device = 1  # 可以成功运行

    # 测试gpu faiss是否可用
    # item_embs = item_embs.astype('float32')
    gpu_index = faiss.GpuIndexFlatIP(res, args.embedding_dim * 2, flat_config)
    gpu_index.add(item_embs)
    print('使用faiss gpu')

    total = 0
    total_recall = 0.0
    total_ndcg = 0.0
    total_hitrate = 0
    total_map = 0.0
    total_diversity = 0.0
    for nick_id, item_id, hist_item, hist_mask in test_data:
        # nick_id, item_id, hist_item, hist_mask = prepare_data(src, tgt)

        user_embs = model.output_user(sess, [hist_item, hist_mask])
        # print('model predict user emb shape:', user_embs.shape)

        # (batch_size, embedding_dim)
        user_con_embs = user_embs[:, 0, :]

        # (batch_size, 1, embedding_dim)
        user_con_embs = np.reshape(user_con_embs, [user_con_embs.shape[0], 1, user_con_embs.shape[-1]])

        # (batch_size, interest_num, embedding_dim)
        user_con_embs = np.tile(user_con_embs, (1, user_embs.shape[1] - 1, 1))

        # (batch_size, interest_num, embedding_dim)
        user_int_embs = user_embs[:, 1:, :]

        # (batch_size, interest_num, 2 * embedding_dim)
        user_embs = np.concatenate((user_int_embs, user_con_embs), axis=2)

        # print('user emb shape:', user_embs.shape)

        if len(user_embs.shape) == 2:
            # gpu
            # user_embs = user_embs.astype('float32')
            D, I = gpu_index.search(user_embs, topN)

            # cpu
            # D, I = cpu_index.search(user_embs, topN)
            for i, iid_list in enumerate(item_id):
                recall = 0
                dcg = 0.0
                item_list = set(I[i])
                for no, iid in enumerate(iid_list):
                    if iid in item_list:
                        recall += 1
                        dcg += 1.0 / math.log(no+2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no+2, 2)
                total_recall += recall * 1.0 / len(iid_list)
                if recall > 0:
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
                if not save:
                    total_diversity += compute_diversity(I[i], item_cate_map)
        else:
            ni = user_embs.shape[1]
            user_embs = np.reshape(user_embs, [-1, user_embs.shape[-1]])

            # gpu
            # user_embs = user_embs.astype('float32')
            D, I = gpu_index.search(user_embs, topN)

            # cpu
            # D, I = cpu_index.search(user_embs, topN)
            for i, iid_list in enumerate(item_id):
                recall = 0
                dcg = 0.0
                item_list_set = set()
                if coef is None:
                    item_list = list(zip(np.reshape(I[i*ni:(i+1)*ni], -1), np.reshape(D[i*ni:(i+1)*ni], -1)))
                    item_list.sort(key=lambda x: x[1], reverse=True)
                    for j in range(len(item_list)):
                        if item_list[j][0] not in item_list_set and item_list[j][0] != 0:
                            item_list_set.add(item_list[j][0])
                            if len(item_list_set) >= topN:
                                break
                else:
                    origin_item_list = list(zip(np.reshape(I[i*ni:(i+1)*ni], -1), np.reshape(D[i*ni:(i+1)*ni], -1)))
                    origin_item_list.sort(key=lambda x:x[1], reverse=True)
                    item_list = []
                    tmp_item_set = set()
                    for (x, y) in origin_item_list:
                        if x not in tmp_item_set and x in item_cate_map:
                            item_list.append((x, y, item_cate_map[x]))
                            tmp_item_set.add(x)
                    cate_dict = defaultdict(int)
                    for j in range(topN):
                        max_index = 0
                        max_score = item_list[0][1] - coef * cate_dict[item_list[0][2]]
                        for k in range(1, len(item_list)):
                            if item_list[k][1] - coef * cate_dict[item_list[k][2]] > max_score:
                                max_index = k
                                max_score = item_list[k][1] - coef * cate_dict[item_list[k][2]]
                            elif item_list[k][1] < max_score:
                                break
                        item_list_set.add(item_list[max_index][0])
                        cate_dict[item_list[max_index][2]] += 1
                        item_list.pop(max_index)

                for no, iid in enumerate(iid_list):
                    if iid in item_list_set:
                        recall += 1
                        dcg += 1.0 / math.log(no+2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no+2, 2)
                total_recall += recall * 1.0 / len(iid_list)
                if recall > 0:
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
                if not save:
                    total_diversity += compute_diversity(list(item_list_set), item_cate_map)
        
        total += len(item_id)
    
    recall = total_recall / total
    ndcg = total_ndcg / total
    hitrate = total_hitrate * 1.0 / total
    diversity = total_diversity * 1.0 / total

    if save:
        return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate}
    return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate, 'diversity': diversity}

def get_model(dataset, model_type, item_count, batch_size, maxlen, item_cate, item_freq):
    # if model_type == 'DNN': 
    #     model = Model_DNN(item_count, args.embedding_dim, args.hidden_size, batch_size, maxlen)
    # elif model_type == 'GRU4REC': 
    #     model = Model_GRU4REC(item_count, args.embedding_dim, args.hidden_size, batch_size, maxlen)
    # elif model_type == 'MIND':
    #     relu_layer = True if dataset == 'book' else False
    #     model = Model_MIND(item_count, args.embedding_dim, args.hidden_size, batch_size, args.num_interest, maxlen, relu_layer=relu_layer)
    # elif model_type == 'ComiRec-DR':
    #     model = Model_ComiRec_DR(item_count, args.embedding_dim, args.hidden_size, batch_size, args.num_interest, maxlen)
    # elif model_type == 'ComiRec-SA':
    #     model = Model_ComiRec_SA(item_count, args.embedding_dim, args.hidden_size, batch_size, args.num_interest, maxlen)
    if model_type == 'MiceRec-SA':
        model = Model_MiceRecSA(n_mid=item_count, embedding_dim=args.embedding_dim, hidden_size=args.hidden_size, batch_size=batch_size, num_interest=args.num_interest, seq_len=maxlen, neg_num=args.neg_num, con_gt_loss_weight=args.con_gt_loss_weight, con_lt_loss_weight=args.con_lt_loss_weight, int_gt_loss_weight=args.int_gt_loss_weight, int_lt_loss_weight=args.int_lt_loss_weight, dis_loss_weight=args.dis_loss_weight, weight_decay=args.weight_decay, item_cate=item_cate, item_freq=item_freq, dis_loss_type=args.dis_loss_type)
    elif model_type == 'MiceRec-DR':
        model = Model_MiceRecDR(n_mid=item_count, embedding_dim=args.embedding_dim, hidden_size=args.hidden_size, batch_size=batch_size, num_interest=args.num_interest, seq_len=maxlen, neg_num=args.neg_num, con_gt_loss_weight=args.con_gt_loss_weight, con_lt_loss_weight=args.con_lt_loss_weight, int_gt_loss_weight=args.int_gt_loss_weight, int_lt_loss_weight=args.int_lt_loss_weight, dis_loss_weight=args.dis_loss_weight, weight_decay=args.weight_decay, item_cate=item_cate, item_freq=item_freq, dis_loss_type=args.dis_loss_type)
    else:
        print ("Invalid model_type : %s", model_type)
        return
    return model

def get_exp_name(dataset, model_type, batch_size, lr, maxlen, save=True):
    # extr_name = input('Please input the experiment name: ')
    extr_name = 'my_test'
    para_name = '_'.join([dataset, model_type, 'b'+str(batch_size), 'lr'+str(lr), 'd'+str(args.embedding_dim), 'len'+str(maxlen)])
    exp_name = para_name + '_' + extr_name

    while os.path.exists('runs/' + exp_name) and save:
        # flag = input('The exp name already exists. Do you want to cover? (y/n)')
        flag = 'Y'
        if flag == 'y' or flag == 'Y':
            shutil.rmtree('runs/' + exp_name)
            break
        else:
            extr_name = input('Please input the experiment name: ')
            exp_name = para_name + '_' + extr_name

    return exp_name

def train(
        train_file,
        valid_file,
        test_file,
        cate_file,
        item_count,
        dataset = "book",
        batch_size = 128,
        maxlen = 100,
        test_iter = 50,
        model_type = 'DNN',
        lr = 0.001,
        max_iter = 100,
        patience = 20
):
    exp_name = get_exp_name(dataset, model_type, batch_size, lr, maxlen)

    best_model_path = "best_model/" + exp_name + '/'

    # gpu选项
    gpu_options = tf.GPUOptions(allow_growth=True) # 可以成功运行
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth=False)

    writer = SummaryWriter('runs/' + exp_name)

    # item_cate_map = load_item_cate(cate_file)
    item_cate_map = odps_load_item_cate(cate_file)

    # item_cate list
    item_cate = get_item_cate_lst(cate_file, item_count)
    print('item_cate初始化完成！')

    # item_freq list
    item_freq = get_item_freq_lst(train_file, item_count)
    print('item_freq初始化完成!')

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_data = DataIterator(train_file, batch_size, maxlen, train_flag=0)
        valid_data = DataIterator(valid_file, batch_size, maxlen, train_flag=1)
        
        model = get_model(dataset, model_type, item_count, batch_size, maxlen, item_cate=item_cate, item_freq=item_freq)
        
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        sess.run(tf.tables_initializer())

        print('training begin')
        sys.stdout.flush()

        start_time = time.time()
        iter = 0

        # 权重部分打印
        print('辅助loss权重打印')
        model.get_loss_weights()
        print('-*-' * 30)

        loss_sum = 0.0
        trials = 0

        # 可注释
        # print('训练前 验证集评估')
        # metrics = evaluate_full(sess, valid_data, model, best_model_path, batch_size, item_cate_map)
        # if metrics != {}:
        #     log_str = ', ' + ', '.join(['valid ' + key + ': %.6f' % value for key, value in metrics.items()])
        # print(exp_name)
        # print(log_str)
        # print('-*-' * 30)

        # prepare_start_time = time.perf_counter()        
        for user_id_lst, item_id_lst, hist_item_lst, hist_mask_list in train_data:
            data_iter = [user_id_lst, item_id_lst, hist_item_lst, hist_mask_list]
            # print('data_iter:', data_iter)
            # prepare_end_time = time.perf_counter()
            # print('data prepare time cost:', prepare_end_time - prepare_start_time)
            # train_start_time = time.perf_counter()

            if args.debug == 1:
                print('main loss及辅助loss打印')
                # l2_loss = model.get_l2_loss(sess, data_iter)
                main_loss = model.get_main_loss(sess, data_iter)
                dcor_loss = model.get_discrepancy_loss(sess, data_iter)
                pos_gt_neg_con_softmax_loss, pos_lt_neg_con_softmax_loss, pos_gt_neg_int_softmax_loss, pos_lt_neg_int_softmax_loss = model.get_assist_loss(sess, data_iter)

                print('main loss:{}, dcor loss:{}, pos_gt_neg_con_softmax_loss:{}, pos_lt_neg_con_softmax_loss:{}, pos_gt_neg_int_softmax_loss:{}, pos_lt_neg_int_softmax_loss:{}'.format(main_loss, dcor_loss, pos_gt_neg_con_softmax_loss, pos_lt_neg_con_softmax_loss, pos_gt_neg_int_softmax_loss, pos_lt_neg_int_softmax_loss))

            loss = model.train(sess, data_iter + [lr])

            if args.debug == 1:
                print('loss:', loss)
            
            loss_sum += loss
            iter += 1

            # print('iter:', iter)
            if iter % args.decay_iter == 0:
                print('iter:', iter)
                # weight衰减
                model.adapt()
            
            if args.dataset == 'taobao':
                test_iter_lst = [test_iter * 1, test_iter * 5, test_iter * 10] + [test_iter * x for x in range(20, 1000, 20)] + [test_iter * x for x in range(1000, 1600, 10)]
                start_iter = 1600
                iter_mod = 2 * test_iter
            elif args.dataset == 'book':
                test_iter_lst = [test_iter * 1, test_iter * 5, test_iter * 10, test_iter * 15, test_iter * 20, test_iter * 25, test_iter * 30, test_iter * 35, test_iter * 40]
                start_iter = 45
                iter_mod = 1 * test_iter
            else:
                print('dataset error')

            if iter in test_iter_lst or (iter % test_iter == 0 and iter >= start_iter * test_iter and iter % iter_mod == 0):
            # if iter % test_iter == 0 and iter >= 10 * test_iter:
                metrics = evaluate_full(sess, valid_data, model, best_model_path, batch_size, item_cate_map)
                log_str = 'iter: %d, train loss: %.4f' % (iter, loss_sum / test_iter)
                if metrics != {}:
                    log_str += ', ' + ', '.join(['valid ' + key + ': %.6f' % value for key, value in metrics.items()])
                print(exp_name)
                print(log_str)

                writer.add_scalar('train/loss', loss_sum / test_iter, iter)
                if metrics != {}:
                    for key, value in metrics.items():
                        writer.add_scalar('eval/' + key, value, iter)
                
                if 'recall' in metrics:
                    recall = metrics['recall']
                    global best_metric
                    if recall > best_metric:
                        best_metric = recall
                        model.save(sess, best_model_path)
                        trials = 0
                    else:
                        trials += 1
                        print('trials:', trials)
                        if trials > patience:
                            break

                loss_sum = 0.0
                test_time = time.time()
                print("time interval: %.4f min" % ((test_time-start_time)/60.0))
                print("curr time: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                sys.stdout.flush()

                print('main loss及辅助loss打印')
                # l2_loss = model.get_l2_loss(sess, data_iter)
                main_loss = model.get_main_loss(sess, data_iter)
                print('main loss:', main_loss)
                # pos_gt_neg_con_softmax_loss, pos_lt_neg_con_softmax_loss, pos_gt_neg_int_softmax_loss, pos_lt_neg_int_softmax_loss = model.get_assist_loss(sess, data_iter)
                # print('main softmax loss: {}, pos_gt_neg_con_softmax_loss: {}, pos_lt_neg_con_softmax_loss: {}, pos_gt_neg_int_softmax_loss: {}, pos_lt_neg_int_softmax_loss:{}, dis_loss:{}'.format(main_loss, pos_gt_neg_con_softmax_loss, pos_lt_neg_con_softmax_loss, pos_gt_neg_int_softmax_loss, pos_lt_neg_int_softmax_loss, l2_loss))
                print('辅助loss权重打印')
                model.get_loss_weights()
                
                print('-*-' * 30)
            
            if iter >= max_iter * 1000:
                print('reached max iter')
                break

        print('-*-' * 50)
        model.restore(sess, best_model_path)
        print('saved model, start validate and test')
        metrics = evaluate_full(sess, valid_data, model, best_model_path, batch_size, item_cate_map, save=False)
        print(', '.join(['valid ' + key + ': %.6f' % value for key, value in metrics.items()]))

        test_data = DataIterator(test_file, batch_size, maxlen, train_flag=2)
        metrics = evaluate_full(sess, test_data, model, best_model_path, batch_size, item_cate_map, save=False)
        print(', '.join(['test ' + key + ': %.6f' % value for key, value in metrics.items()]))

def test(
        test_file,
        cate_file,
        item_count,
        dataset = "book",
        batch_size = 128,
        maxlen = 100,
        model_type = 'DNN',
        lr = 0.001
):
    exp_name = get_exp_name(dataset, model_type, batch_size, lr, maxlen, save=False)
    best_model_path = "best_model/" + exp_name + '/'
    gpu_options = tf.GPUOptions(allow_growth=True)
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

    model = get_model(dataset, model_type, item_count, batch_size, maxlen)
    item_cate_map = odps_load_item_cate(cate_file)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.restore(sess, best_model_path)
        
        test_data = DataIterator(test_file, batch_size, maxlen, train_flag=2)
        metrics = evaluate_full(sess, test_data, model, best_model_path, batch_size, item_cate_map, save=False, coef=args.coef)
        print(', '.join(['test ' + key + ': %.6f' % value for key, value in metrics.items()]))

def output(
        item_count,
        dataset = "book",
        batch_size = 128,
        maxlen = 100,
        model_type = 'DNN',
        lr = 0.001
):
    exp_name = get_exp_name(dataset, model_type, batch_size, lr, maxlen, save=False)
    best_model_path = "best_model/" + exp_name + '/'
    gpu_options = tf.GPUOptions(allow_growth=True)
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

    model = get_model(dataset, model_type, item_count, batch_size, maxlen)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.restore(sess, best_model_path)
        item_embs = model.output_item(sess)
        np.save('output/' + exp_name + '_emb.npy', item_embs)

if __name__ == '__main__':
    print(sys.argv)
    args = parser.parse_args()
    SEED = args.random_seed

    tf.set_random_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # 检验gpu是否可用
    print('-*-' * 30)
    print('GPU可用标识：', tf.test.is_gpu_available())
    print('-*-' * 30)

    train_name = 'train'
    valid_name = 'valid'
    test_name = 'test'

    if args.dataset == 'taobao':
        path = '../../data/data/taobao_data/'
        item_count = 1708531
        batch_size = 256
        maxlen = 50
        test_iter = 500
    elif args.dataset == 'book':
        path = '../../data/data/book_data/'
        item_count = 367983
        batch_size = 128
        maxlen = 20
        test_iter = 1000

    # 读取本地文件
    train_file = path + args.dataset + '_train.txt'
    valid_file = path + args.dataset + '_valid.txt'
    test_file = path + args.dataset + '_test.txt'
    cate_file = path + args.dataset + '_item_cate.txt'

    dataset = args.dataset

    if args.p == 'train':
        train(train_file=train_file, valid_file=valid_file, test_file=test_file, cate_file=cate_file, 
              item_count=item_count, dataset=dataset, batch_size=batch_size, maxlen=maxlen, test_iter=test_iter, 
              model_type=args.model_type, lr=args.learning_rate, max_iter=args.max_iter, patience=args.patience)
    elif args.p == 'test':
        test(test_file=test_file, cate_file=cate_file, item_count=item_count, dataset=dataset, batch_size=batch_size, 
             maxlen=maxlen, model_type=args.model_type, lr=args.learning_rate)
    elif args.p == 'output':
        output(item_count=item_count, dataset=dataset, batch_size=batch_size, maxlen=maxlen, 
               model_type=args.model_type, lr=args.learning_rate)
    else:
        print('do nothing...')

    print('-*-' * 30)
    print('train finished')
