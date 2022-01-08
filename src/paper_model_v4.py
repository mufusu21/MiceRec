import os
import numpy as np
import tensorflow as tf
# import tensorflow.contrib.slim as slim
# from tensorflow.nn.rnn_cell import GRUCell
from tensorflow.python.ops.nn_impl import _compute_sampled_logits


class Basic_Model(object):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, seq_len, neg_num, item_cate,
                 item_freq, cate_prop=0, dis_loss_type='cos', flag='MiceRec'):
        self.reg = False
        self.batch_size = batch_size
        self.n_mid = n_mid  # item数量
        self.neg_num = neg_num
        self.embedding_dim = embedding_dim
        self.dis_loss_type = dis_loss_type

        # item_ids (n_mid, )
        self.item_ids = tf.constant(list(range(n_mid)))
        self.item_cate = tf.constant(item_cate)
        self.item_freq = tf.constant(item_freq)
        self.item_cate_dict = tf.contrib.lookup.HashTable(
            initializer=tf.contrib.lookup.KeyValueTensorInitializer(keys=self.item_ids, values=self.item_cate),
            default_value=-1, name="item_cate_dict"
        )
        # item_freq (n_mid, )
        self.item_freq_dict = tf.contrib.lookup.HashTable(
            initializer=tf.contrib.lookup.KeyValueTensorInitializer(keys=self.item_ids, values=self.item_freq),
            default_value=0, name="item_freq_dict"
        )
        self.cate_prop = cate_prop

        with tf.name_scope('Inputs'):
            # 序列输入 (batch_size, max_len)
            self.item_his_batch_input = tf.placeholder(tf.int32, [None, None], name='item_his_batch_input')
            # user侧输入 (batch_size, )
            self.user_batch_input = tf.placeholder(tf.int32, [None, ], name='user_batch_input')
            # 目标item侧输入 (batch_size, )
            self.item_batch_input = tf.placeholder(tf.int32, [None, ], name='item_batch_input')
            # 序列mask输入 (batch_size, max_len)
            self.mask = tf.placeholder(tf.float32, [None, None], name='mask_batch_input')
            # # 训练时没有用到的输入
            # self.target_ph = tf.placeholder(tf.float32, [None, 2], name='target_input')
            # 学习率 数值
            self.lr = tf.placeholder(tf.float64, [])

        self.mask_length = tf.cast(tf.reduce_sum(self.mask, -1), dtype=tf.int32)
        self.batch_pos_items = self.item_batch_input

        # Embedding layer
        with tf.name_scope('Embedding_layer'):
            # item的兴趣 embedding (item_num, embedding_dim)
            self.item_int_embeddings_var = tf.get_variable("item_int_embeddings_var", [n_mid, embedding_dim],
                                                           trainable=True)
            self.item_con_embeddings_var = tf.get_variable("item_con_embeddings_var", [n_mid, embedding_dim],
                                                           trainable=True)

            # item embedding偏置项/lookup用 初始化为0,trainable为False (item_num)
            self.item_embeddings_bias = tf.get_variable("bias_lookup_table", [n_mid],
                                                        initializer=tf.zeros_initializer(),
                                                        trainable=False)
            # 根据输入的一个batch的item_id从item_int_embeddings_var中lookup (batch_size, embedding_dim)
            self.batch_item_int_embedded = tf.nn.embedding_lookup(self.item_int_embeddings_var, self.item_batch_input)

            # 根据输入的一个batch的item_id从item_con_embeddings_var中lookup (batch_size, embedding_dim)
            self.batch_item_con_embedded = tf.nn.embedding_lookup(self.item_con_embeddings_var, self.item_batch_input)

            # 根据输入的一个batch的hist item id list从item_int_embeddings_var中lookup (batch_size, max_len, embedding_dim)
            self.batch_item_int_his_embedded = tf.nn.embedding_lookup(self.item_int_embeddings_var,
                                                                      self.item_his_batch_input)

            # 根据输入的一个batch的hist item id list从item_con_embeddings_var中lookup (batch_size, max_len, embedding_dim)
            self.batch_item_con_his_embedded = tf.nn.embedding_lookup(self.item_con_embeddings_var,
                                                                      self.item_his_batch_input)


        # 最终使用的batch item int embedding (batch_size, embedding_dim)
        self.batch_item_int_embeddings = self.batch_item_int_embedded

        # 最终使用的batch item con embedding (batch_size, embedding_dim)
        self.batch_item_con_embeddings = self.batch_item_con_embedded


        # 最终使用的batch item_embedding (batch_size, embedding_dim * 2)
        self.batch_item_embedding = tf.concat([self.batch_item_int_embeddings, self.batch_item_con_embeddings], axis=1)

        # batch hist item interest embedding (batch_size, max_len, embedding_dim)
        self.batch_item_hist_int_embeddings = self.batch_item_int_his_embedded * tf.reshape(self.mask, (-1, seq_len, 1))

        # batch hist item conformity embedding (batch_size, max_len, embedding_dim)
        self.batch_item_hist_con_embeddings = self.batch_item_con_his_embedded * tf.reshape(self.mask, (-1, seq_len, 1))

        # 所有的 item_embedding (item_num, embedding_dim * 2)
        self.item_embeddings_var = tf.concat([self.item_int_embeddings_var, self.item_con_embeddings_var], axis=1)

    # 普通的softmax loss
    def get_softmax_loss(self, scores, labels):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=scores))

        return cross_entropy

    # cos平方距离公式
    def cos_square_dis(self, x, y):
        # x: (batch_size, embedding_dim), y: (batch_size, embedding_dim)
        # 对于每一个样本 sqrt((xy)^2 / (x2y2))
        x_square = tf.reduce_sum(tf.square(x), axis=1)
        y_square = tf.reduce_sum(tf.square(y), axis=1)
        xy_square = tf.square(tf.reduce_sum(x * y, axis=1))
        cov = tf.reduce_mean(tf.sqrt(xy_square / (x_square * y_square) + 1e-8), axis=0)

        return cov

    # cos距离公式
    def cos_dis(self, x, y):
        # x: (batch_size, embedding_dim), y: (batch_size, embedding_dim)
        # 对于每一个样本 sqrt((xy)^2 / (x2y2))
        x_square_sqrt = tf.sqrt(tf.reduce_sum(tf.square(x), axis=1) + 1e-8)
        y_square_sqrt = tf.sqrt(tf.reduce_sum(tf.square(y), axis=1) + 1e-8)
        xy = tf.reduce_sum(x * y, axis=1)
        cov = tf.reduce_mean(xy / (x_square_sqrt * y_square_sqrt), axis=0)

        return cov

    # l2距离公式
    def l2_dis(self, x, y):
        # x: (batch_size, embedding_dim), y: (batch_size, embedding_dim)
        l2_loss = -tf.nn.l2_loss(x - y) / self.batch_size
        return l2_loss

    def normalized_l2_dis(self, x, y):
        x_norm = tf.math.l2_normalize(x)
        y_norm = tf.math.l2_normalize(y)
        l2_norm_loss = -tf.nn.l2_loss(x_norm - y_norm) / self.batch_size
        return l2_norm_loss

    def mean_l2_dis(self, x, y):
        # 先计算中心再计算norm 再计算l2
        x_mean = tf.reduce_mean(x, axis=0)
        y_mean = tf.reduce_mean(y, axis=0)

        x_norm = tf.math.l2_normalize(x_mean)
        y_norm = tf.math.l2_normalize(y_mean)

        l2_loss = -tf.nn.l2_loss(x_norm - y_norm)
        return l2_loss

    def mean_cos_dis(self, x, y):
        # 先计算中心再计算norm 再计算l2
        x_mean = tf.reduce_mean(x, axis=0)
        y_mean = tf.reduce_mean(y, axis=0)

        # (embedding_dim, )
        x_norm = tf.math.l2_normalize(x_mean)
        y_norm = tf.math.l2_normalize(y_mean)

        x_square_sqrt = tf.sqrt(tf.reduce_sum(tf.square(x_norm)))
        y_square_sqrt = tf.sqrt(tf.reduce_sum(tf.square(y_norm)))
        xy = tf.reduce_sum(x_norm * y_norm)
        cov = xy / (x_square_sqrt * y_square_sqrt + 1e-8)

        return cov

    def dcor_ori(self, x, y):

        x_norm = tf.norm(tf.expand_dims(x, axis=1) - x, axis=2)
        y_norm = tf.norm(tf.expand_dims(y, axis=1) - y, axis=2)

        x_ = x_norm - tf.expand_dims(tf.reduce_mean(x_norm, axis=0), axis=0) - tf.expand_dims(tf.reduce_mean(x_norm, axis=1), axis=1) + tf.reduce_mean(
            x_norm)
        y_ = y_norm - tf.expand_dims(tf.reduce_mean(y_norm, axis=0), axis=0) - tf.expand_dims(tf.reduce_mean(y_norm, axis=1), axis=1) + tf.reduce_mean(
            y_norm)
        
        dcov2_xy = tf.reduce_sum(x_ * y_)
        dcov2_xx = tf.reduce_sum(x_ * x_)
        dcov2_yy = tf.reduce_sum(y_ * y_)

        # dcor_square = dcov2_xy
        dcor_square = -(dcov2_xy * dcov2_xy) / (dcov2_xx * dcov2_yy)

        # tf.constant(0, dtype=tf.float32)
        return dcor_square

    def dcor(self, x, y):

        x = tf.nn.l2_normalize(x, axis=1)
        y = tf.nn.l2_normalize(y, axis=1)

        x_norm = tf.reduce_sum(tf.square(tf.expand_dims(x, axis=1) - x), axis=2)
        y_norm = tf.reduce_sum(tf.square(tf.expand_dims(y, axis=1) - y), axis=2)

        x_ = x_norm - tf.expand_dims(tf.reduce_mean(x_norm, axis=0), axis=0) - tf.expand_dims(tf.reduce_mean(x_norm, axis=1), axis=1) + tf.reduce_mean(
            x_norm)
        y_ = y_norm - tf.expand_dims(tf.reduce_mean(y_norm, axis=0), axis=0) - tf.expand_dims(tf.reduce_mean(y_norm, axis=1), axis=1) + tf.reduce_mean(
            y_norm)

        dcov2_xy = tf.reduce_sum(x_ * y_)
        dcov2_xx = tf.reduce_sum(x_ * x_)
        dcov2_yy = tf.reduce_sum(y_ * y_)

        # dcor_square = dcov2_xy
        dcor_square = -(dcov2_xy * dcov2_xy) / (dcov2_xx * dcov2_yy)

        # tf.constant(0, dtype=tf.float32)
        return dcor_square

    def sampled_softmax_loss(self, labels, logits):
        # exp_logits = tf.exp(logits)
        # sampled_softmax = exp_logits / tf.expand_dims(tf.reduce_sum(exp_logits, axis=1), axis=1)
        # loss = -tf.reduce_mean(tf.math.log(tf.reduce_sum(labels * sampled_softmax, axis=1)), axis=0)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))

        return loss

    # mask只能是0/1，scores需在进入此函数前处理完
    def masked_sampled_softmax_loss(self, labels, logits, mask):
        # labels: (batch_size, neg_num + 1), logits: (batch_size, neg_num + 1), mask: (batch_size, neg_num + 1)
        # mask_softmax: (batch_size, neg_num + 1)
        mask_exp_logits = tf.exp(logits) * mask
        mask_softmax = mask_exp_logits / tf.expand_dims(tf.reduce_sum(mask_exp_logits, axis=1), axis=1)
        loss = -tf.reduce_mean(tf.math.log(tf.reduce_sum(labels * mask_softmax, axis=1)), axis=0)

        return loss

    def build_loss(self):
        # 原始论文的sampled_softmax_loss
        self.main_loss = self.sampled_softmax_loss(labels=self.main_labels, logits=self.main_logits)

        self.pos_gt_neg_con_softmax_loss = self.masked_sampled_softmax_loss(labels=self.con_pos_labels,
                                                                            logits=self.con_pos_logits,
                                                                            mask=self.con_pos_mask)
        self.pos_lt_neg_con_softmax_loss = self.masked_sampled_softmax_loss(labels=self.con_neg_labels,
                                                                            logits=self.con_neg_logits,
                                                                            mask=self.con_neg_mask)

        self.pos_gt_neg_int_softmax_loss = self.masked_sampled_softmax_loss(labels=self.int_pos_labels,
                                                                            logits=self.int_pos_logits,
                                                                            mask=self.int_pos_mask)
        self.pos_lt_neg_int_softmax_loss = self.masked_sampled_softmax_loss(labels=self.int_neg_labels,
                                                                            logits=self.int_neg_logits,
                                                                            mask=self.int_neg_mask)
        # cos_dis normalized_l2_dis
        if self.dis_loss_type == 'cos':
            self.int_con_discrepancy_loss = self.cos_dis(self.user_int_att_emb, self.user_con_att_emb) + \
                                        self.cos_dis(self.batch_item_int, self.batch_item_con)
        if self.dis_loss_type == 'cos_square':
            self.int_con_discrepancy_loss = self.cos_square_dis(self.user_int_att_emb, self.user_con_att_emb) + \
                                        self.cos_square_dis(self.batch_item_int, self.batch_item_con)
        elif self.dis_loss_type == 'dcor':
            # self.int_con_discrepancy_loss = self.dcor(self.user_int_att_emb, self.user_con_att_emb) + \
            #                             self.dcor(self.batch_item_int, self.batch_item_con)
            self.int_con_discrepancy_loss = self.dcor(self.batch_item_int, self.batch_item_con)
        elif self.dis_loss_type == 'l2':
            self.int_con_discrepancy_loss = self.normalized_l2_dis(self.user_int_att_emb, self.user_con_att_emb) + \
                                        self.normalized_l2_dis(self.batch_item_int, self.batch_item_con)
        elif self.dis_loss_type == 'mean_cos':
            self.int_con_discrepancy_loss = self.mean_cos_dis(self.user_int_att_emb, self.user_con_att_emb) + \
                                        self.mean_cos_dis(self.batch_item_int, self.batch_item_con)
        elif self.dis_loss_type == 'mean_l2':
            self.int_con_discrepancy_loss = self.mean_l2_dis(self.user_int_att_emb, self.user_con_att_emb) + \
                                        self.mean_l2_dis(self.batch_item_int, self.batch_item_con)
        else:
            self.int_con_discrepancy_loss = self.cos_dis(self.user_int_att_emb, self.user_con_att_emb) + \
                                        self.cos_dis(self.batch_item_int, self.batch_item_con)                                        
        # 原始loss
        # self.loss = self.main_loss + self.con_gt_loss_weight * self.pos_gt_neg_con_softmax_loss + \
        #             + self.con_lt_loss_weight * self.pos_lt_neg_con_softmax_loss + \
        #             + self.int_gt_loss_weight * self.pos_gt_neg_int_softmax_loss + \
        #             + self.int_lt_loss_weight * self.pos_lt_neg_int_softmax_loss + \
        #             + self.dis_loss_weight * self.int_con_discrepancy_loss
        self.loss = self.main_loss
        if self.con_gt_loss_weight > 1e-6:
            self.loss += self.con_gt_loss_weight * self.pos_gt_neg_con_softmax_loss
        if self.con_lt_loss_weight > 1e-6:
            self.loss += self.con_lt_loss_weight * self.pos_lt_neg_con_softmax_loss
        if self.int_gt_loss_weight > 1e-6:
            self.loss += self.int_gt_loss_weight * self.pos_gt_neg_int_softmax_loss
        if self.int_lt_loss_weight > 1e-6:
            self.loss += self.int_lt_loss_weight * self.pos_lt_neg_int_softmax_loss
        if self.dis_loss_weight > 1e-6:
            self.loss += self.dis_loss_weight * self.int_con_discrepancy_loss

        # self.loss = self.main_loss

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def train(self, sess, inps):
        feed_dict = {
            # user_id batch (batch_size, )
            self.user_batch_input: inps[0],
            # item_id batch (batch_size, )
            self.item_batch_input: inps[1],
            # hist_item_id_lst (batch_size, hist_len)
            self.item_his_batch_input: inps[2],
            # mask (batch_size, hist_len)
            self.mask: inps[3],
            # 学习率
            self.lr: inps[4]
        }
        loss, _ = sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
        return loss

    def output_item(self, sess):
        # 输出item的embedding (batch_size, 2 * embedding_dim) int在前con在后
        item_embs = sess.run(self.item_embeddings_var)
        return item_embs

    def output_user(self, sess, inps):
        # 输出user embedding 输入是历史序列和mask user_embeddings由usr_int_embeddings和user_con_embeddings拼接得到
        # (batch_size, num_heads + 1, embedding_dim) con在前 int在后
        user_embs = sess.run(self.user_embeddings, feed_dict={
            self.item_his_batch_input: inps[0],
            self.mask: inps[1]
        })
        return user_embs

    def save(self, sess, path):
        if not os.path.exists(path):
            os.makedirs(path)
        saver = tf.train.Saver()
        saver.save(sess, path + 'model.ckpt')

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, path + 'model.ckpt')
        print('model restored from %s' % path)

    def adapt(self):
        self.con_gt_loss_weight = self.con_gt_loss_weight * self.weight_decay
        self.con_lt_loss_weight = self.con_lt_loss_weight * self.weight_decay
        self.int_gt_loss_weight = self.int_gt_loss_weight * self.weight_decay
        self.int_lt_loss_weight = self.int_lt_loss_weight * self.weight_decay
        self.dis_loss_weight = self.dis_loss_weight * self.weight_decay

    def get_neg_items(self, sess, inps):
        feed_dict = {
            # user_id batch (batch_size, )
            self.user_batch_input: inps[0],
            # item_id batch (batch_size, )
            self.item_batch_input: inps[1],
            # hist_item_id_lst (batch_size, hist_len)
            self.item_his_batch_input: inps[2],
            # mask (batch_size, hist_len)
            self.mask: inps[3],
        }
        neg_items = sess.run(self.sampled_batch_neg_items, feed_dict=feed_dict)
        return neg_items

    def get_assist_loss(self, sess, inps):
        feed_dict = {
            # user_id batch (batch_size, )
            self.user_batch_input: inps[0],
            # item_id batch (batch_size, )
            self.item_batch_input: inps[1],
            # hist_item_id_lst (batch_size, hist_len)
            self.item_his_batch_input: inps[2],
            # mask (batch_size, hist_len)
            self.mask: inps[3],
        }
        pos_gt_neg_con_softmax_loss, pos_lt_neg_con_softmax_loss, pos_gt_neg_int_softmax_loss, pos_lt_neg_int_softmax_loss = sess.run([self.pos_gt_neg_con_softmax_loss, self.pos_lt_neg_con_softmax_loss,
                            self.pos_gt_neg_int_softmax_loss, self.pos_lt_neg_int_softmax_loss], feed_dict=feed_dict)

        return pos_gt_neg_con_softmax_loss, pos_lt_neg_con_softmax_loss, pos_gt_neg_int_softmax_loss, pos_lt_neg_int_softmax_loss

    def get_masks(self, sess, inps):
        feed_dict = {
            # user_id batch (batch_size, )
            self.user_batch_input: inps[0],
            # item_id batch (batch_size, )
            self.item_batch_input: inps[1],
            # hist_item_id_lst (batch_size, hist_len)
            self.item_his_batch_input: inps[2],
            # mask (batch_size, hist_len)
            self.mask: inps[3],
        }
        con_pos_mask, con_neg_mask, int_pos_mask, int_neg_mask = sess.run([self.con_pos_mask, self.con_neg_mask,
                            self.int_pos_mask, self.int_neg_mask], feed_dict=feed_dict)
        return con_pos_mask, con_neg_mask, int_pos_mask, int_neg_mask

    def get_origin_masks(self, sess, inps):
        feed_dict = {
            # user_id batch (batch_size, )
            self.user_batch_input: inps[0],
            # item_id batch (batch_size, )
            self.item_batch_input: inps[1],
            # hist_item_id_lst (batch_size, hist_len)
            self.item_his_batch_input: inps[2],
            # mask (batch_size, hist_len)
            self.mask: inps[3],
        }
        conformity_mask, interest_mask = sess.run([self.conformity_mask, self.interest_mask],
                                                  feed_dict=feed_dict)

        return conformity_mask, interest_mask

    def get_cos_loss(self, sess, inps):
        feed_dict = {
            # user_id batch (batch_size, )
            self.user_batch_input: inps[0],
            # item_id batch (batch_size, )
            self.item_batch_input: inps[1],
            # hist_item_id_lst (batch_size, hist_len)
            self.item_his_batch_input: inps[2],
            # mask (batch_size, hist_len)
            self.mask: inps[3],
        }
        int_con_discrepancy_loss = sess.run([self.int_con_discrepancy_loss], feed_dict=feed_dict)
        return int_con_discrepancy_loss

    def get_l2_loss(self, sess, inps):
        feed_dict = {
            # user_id batch (batch_size, )
            self.user_batch_input: inps[0],
            # item_id batch (batch_size, )
            self.item_batch_input: inps[1],
            # hist_item_id_lst (batch_size, hist_len)
            self.item_his_batch_input: inps[2],
            # mask (batch_size, hist_len)
            self.mask: inps[3],
        }
        l2_loss = sess.run([self.int_con_discrepancy_loss], feed_dict=feed_dict)
        return l2_loss

    def get_discrepancy_loss(self, sess, inps):
        feed_dict = {
            # user_id batch (batch_size, )
            self.user_batch_input: inps[0],
            # item_id batch (batch_size, )
            self.item_batch_input: inps[1],
            # hist_item_id_lst (batch_size, hist_len)
            self.item_his_batch_input: inps[2],
            # mask (batch_size, hist_len)
            self.mask: inps[3],
        }
        discrepancy_loss = sess.run([self.int_con_discrepancy_loss], feed_dict=feed_dict)
        return discrepancy_loss

    def get_main_loss(self, sess, inps):
        feed_dict = {
            # user_id batch (batch_size, )
            self.user_batch_input: inps[0],
            # item_id batch (batch_size, )
            self.item_batch_input: inps[1],
            # hist_item_id_lst (batch_size, hist_len)
            self.item_his_batch_input: inps[2],
            # mask (batch_size, hist_len)
            self.mask: inps[3],
        }
        main_loss = sess.run([self.main_loss], feed_dict=feed_dict)
        return main_loss

    def get_loss_weights(self):
        print('con_gt_loss_weight:{}, con_lt_loss_weight:{}, con_gt_loss_weight:{}, con_lt_loss_weight:{}, '
              'dis_loss_weight:{}'.format(self.con_gt_loss_weight, self.con_lt_loss_weight,
                                                  self.con_gt_loss_weight, self.con_lt_loss_weight,
                                                  self.dis_loss_weight))

    # debug用 batch_pos_freq_tile
    def get_debug_res(self, sess, inps):
        feed_dict = {
            # user_id batch (batch_size, )
            self.user_batch_input: inps[0],
            # item_id batch (batch_size, )
            self.item_batch_input: inps[1],
            # hist_item_id_lst (batch_size, hist_len)
            self.item_his_batch_input: inps[2],
            # mask (batch_size, hist_len)
            self.mask: inps[3],
        }
        sampled_batch_neg_items, batch_pos_freq, batch_neg_freq, batch_pos_freq_tile, batch_neg_freq_tile, \
        conformity_mask, interest_mask, con_pos_mask, con_neg_mask, int_pos_mask, int_neg_mask = \
            sess.run([self.sampled_batch_neg_items, self.batch_pos_freq, self.batch_neg_freq, self.batch_pos_freq_tile,
                      self.batch_neg_freq_tile, self.conformity_mask, self.interest_mask,
                      self.con_pos_mask, self.con_neg_mask, self.int_pos_mask, self.int_neg_mask], feed_dict=feed_dict)
        return sampled_batch_neg_items, batch_pos_freq, batch_neg_freq, batch_pos_freq_tile, batch_neg_freq_tile, \
               conformity_mask, interest_mask, con_pos_mask, con_neg_mask, int_pos_mask, int_neg_mask


def get_shape(inputs):
    dynamic_shape = tf.shape(inputs)
    static_shape = inputs.get_shape().as_list()
    shape = []
    for i, dim in enumerate(static_shape):
        shape.append(dim if dim is not None else dynamic_shape[i])
    return shape


# neg_num = batch_size * neg_num
class Model_MiceRecSA(Basic_Model):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, num_interest, item_cate,
                 item_freq, cate_prop=0, seq_len=256, neg_num=10, add_pos=True,
                 con_att=True, con_gt_loss_weight=0.1, con_lt_loss_weight=0.1, int_gt_loss_weight=0.1,
                 int_lt_loss_weight=0.1, dis_loss_weight=0.01, weight_decay=0.9, dis_loss_type='cos'):
        super(Model_MiceRecSA, self).__init__(n_mid, embedding_dim, hidden_size, batch_size, seq_len, neg_num, item_cate,
                 item_freq, cate_prop, dis_loss_type, flag="MiceRec-SA")
        # embedding维度
        self.dim = embedding_dim
        self.con_att = con_att
        self.seq_len = seq_len
        self.con_gt_loss_weight = con_gt_loss_weight
        self.con_lt_loss_weight = con_lt_loss_weight
        self.int_gt_loss_weight = int_gt_loss_weight
        self.int_lt_loss_weight = int_lt_loss_weight
        self.dis_loss_weight = dis_loss_weight
        self.weight_decay = weight_decay

        # (batch_size, max_len, embedding_dim)
        batch_item_list_int_emb = tf.reshape(self.batch_item_hist_int_embeddings, [-1, seq_len, embedding_dim])
        batch_item_list_con_emb = tf.reshape(self.batch_item_hist_con_embeddings, [-1, seq_len, embedding_dim])

        if add_pos:
            # (1, max_len, embedding_dim)
            self.position_embedding = \
                tf.get_variable(
                    shape=[1, seq_len, embedding_dim],
                    name='position_embedding')
            # (batch_size, max_len, embedding_dim)
            batch_item_list_int_add_pos = batch_item_list_int_emb + tf.tile(self.position_embedding,
                                                                            [tf.shape(batch_item_list_int_emb)[0], 1, 1])
            batch_item_list_con_add_pos = batch_item_list_con_emb + tf.tile(self.position_embedding,
                                                                            [tf.shape(batch_item_list_con_emb)[0], 1, 1])
        else:
            # (batch_size, max_len, embedding_dim)
            batch_item_list_int_add_pos = batch_item_list_int_emb
            batch_item_list_con_add_pos = batch_item_list_con_emb

        # 兴趣数量
        num_heads = num_interest

        # user兴趣emb侧的self-attention
        with tf.variable_scope("self_atten", reuse=tf.AUTO_REUSE) as scope:
            # (batch_size, max_len, 4 * hidden_size)
            user_int_item_hidden = tf.layers.dense(batch_item_list_int_add_pos, hidden_size * 4, activation=tf.nn.tanh)

            # (batch_size, max_len, num_heads)
            user_int_item_att_w = tf.layers.dense(user_int_item_hidden, num_heads, activation=None)

            # (batch_size, num_heads, max_len)
            user_int_item_att_w = tf.transpose(user_int_item_att_w, [0, 2, 1])

            # tf.expand_dims(self.mask, axis=1) -> (batch_size, 1, max_len)
            # atten_mask -> (batch_size, num_heads, max_len)
            user_int_atten_mask = tf.tile(tf.expand_dims(self.mask, axis=1), [1, num_heads, 1])

            # paddings: (batch_size, num_heads, max_len)
            user_int_paddings = tf.ones_like(user_int_atten_mask) * (-2 ** 32 + 1)

            # item_att_w: (batch_size, num_heads, max_len)
            user_int_item_att_w = tf.where(tf.equal(user_int_atten_mask, 0), user_int_paddings, user_int_item_att_w)

            # item_att_w: (batch_size, num_heads, max_len)
            user_int_item_att_w = tf.nn.softmax(user_int_item_att_w)

            # interest_emb: (batch_size, num_heads, embedding_dim)
            user_interest_emb = tf.matmul(user_int_item_att_w, batch_item_list_int_emb)

        # user_eb: (batch_size, num_heads, embedding_dim)
        self.user_int_embeddings = user_interest_emb

        # user从众侧的embedding
        if con_att:
            # user从众侧添加self-attention进行降维 (batch_size, max_len, 4 * hidden_size)
            user_con_item_hidden = tf.layers.dense(batch_item_list_con_add_pos, hidden_size * 4, activation=tf.nn.tanh)

            # (batch_size, max_len, 1)
            user_con_item_att_w = tf.layers.dense(user_con_item_hidden, 1, activation=None)
            # print('user_con_item_att_w shape:', user_con_item_att_w.shape)

            # (batch_size, 1, max_len)
            user_con_item_att_w = tf.transpose(user_con_item_att_w, [0, 2, 1])
            # print('user_con_item_att_w shape:', user_con_item_att_w.shape)

            # tf.expand_dims(self.mask, axis=1) -> (batch_size, 1, max_len)
            # atten_mask -> (batch_size, 1, max_len)
            user_con_atten_mask = tf.tile(tf.expand_dims(self.mask, axis=1), [1, 1, 1])
            # print('user_con_atten_mask shape:', user_con_atten_mask.shape)

            # paddings: (batch_size, 1, max_len)
            user_con_paddings = tf.ones_like(user_con_atten_mask) * (-2 ** 32 + 1)
            # print('user_con_paddings shape:', user_con_paddings.shape)

            # item_att_w: (batch_size, 1, max_len)
            user_con_item_att_w = tf.where(tf.equal(user_con_atten_mask, 0), user_con_paddings, user_con_item_att_w)
            # print('user_con_item_att_w shape:', user_con_item_att_w.shape)

            # item_att_w: (batch_size, 1, max_len)
            user_con_item_att_w = tf.nn.softmax(user_con_item_att_w)

            # interest_emb: (batch_size, 1, embedding_dim)
            # print('user_con_item_att_w shape:', user_con_item_att_w.shape)
            # print('batch_item_list_con_emb:', batch_item_list_con_emb.shape)
            user_conformity_emb = tf.matmul(user_con_item_att_w, batch_item_list_con_emb)
            # print('user_conformity_emb:', user_conformity_emb.shape)

        else:
            # user从众直接添加全连接进行降维 先把user_con_item摊平 (batch_size, max_len * embedding_dim)
            batch_item_list_con_add_pos_flatten = tf.reshape(batch_item_list_con_add_pos,
                                                             [batch_size, self.seq_len * embedding_dim])
            # (batch_size, embedding_dim)
            user_con_item_hidden = tf.layers.dense(batch_item_list_con_add_pos_flatten, embedding_dim, activation=tf.nn.relu)

            # (batch_size, 1, embedding_dim)
            user_conformity_emb = tf.reshape(user_con_item_hidden, [batch_size, 1, embedding_dim])

        # (batch_size, 1, embedding_dim)
        self.user_con_embeddings = user_conformity_emb

        # 得到serving阶段的user embedding (batch_size, num_heads + 1, embedding_dim)
        self.user_embeddings = tf.concat([self.user_con_embeddings, self.user_int_embeddings], axis=1)

        # user兴趣侧 target-aware attention key,value为user_int_embeddings, query为target item 的兴趣embedding
        # self.user_int_embeddings (batch_size, num_heads, embedding_dim)

        # user_int_embeddings: (batch_size, num_heads, embedding_dim), batch_item_int_embeddings: (batch_size, embedding_dim)
        # batch_item_list_int_emb: (batch_size, max_len, embedding_dim), get_shape(item_list_emb)[0] = batch_size
        # tf.reshape(self.batch_item_int_embeddings, [get_shape(batch_item_list_int_emb)[0], self.dim, 1]): (batch_size, embedding_dim, 1)
        # item_int_target_atten: (batch_size, num_heads, 1)
        item_int_target_atten = tf.matmul(self.user_int_embeddings, tf.reshape(self.batch_item_int_embeddings,
                                                                               [get_shape(batch_item_list_int_emb)[0], self.dim, 1]))

        # tf.reshape(item_int_target_atten, [get_shape(batch_item_list_int_emb)[0], num_heads]): (batch_size, num_heads)
        # tf.pow(tf.reshape(item_int_target_atten, [get_shape(batch_item_list_int_emb)[0], num_heads]), 1): (batch_size, num_heads)
        # item_int_target_atten: (batch_size, num_heads)
        item_int_target_atten = tf.nn.softmax(tf.pow(tf.reshape(item_int_target_atten,
                                                                [get_shape(batch_item_list_int_emb)[0], num_heads]), 1))

        # tf.reshape(self.user_eb, [-1, self.dim]): (batch_size * num_heads, embedding_dim)
        # tf.argmax(atten, axis=1, output_type=tf.int32): (batch_size, )
        # tf.range(tf.shape(item_list_emb)[0]) (batch_size, )
        # user_int_att_emb: (batch_size, embedding_dim)
        self.user_int_att_emb = tf.gather(tf.reshape(self.user_int_embeddings, [-1, self.dim]),
                            tf.argmax(item_int_target_atten, axis=1, output_type=tf.int32) + tf.range(
                                tf.shape(batch_item_list_int_emb)[0]) * num_heads)

        # (batch_size, embedding_dim)
        self.user_con_att_emb = tf.reshape(self.user_con_embeddings, [batch_size, embedding_dim])

        # 最终的user_emb (batch_size, 2 * embedding_dim) 用来计算最后的点积softmax loss不用作serving
        self.user_embeddings_var = tf.concat([self.user_int_att_emb,
                                              self.user_con_att_emb], axis=1)

        # 以下为负采样部分
        batch_neg_num = neg_num * batch_size
        self.sampled_batch_neg_items, self.pos_q, self.neg_q = tf.nn.log_uniform_candidate_sampler(true_classes=tf.cast(tf.reshape(self.item_batch_input, [-1, 1]), tf.int64),
                                            num_true=1, num_sampled=batch_neg_num, unique=True,
                                            range_max=n_mid)

        # 计算整体的logits和labels
        self.main_logits, self.main_labels = _compute_sampled_logits(self.item_embeddings_var, self.item_embeddings_bias,
                        tf.reshape(self.item_batch_input, [-1, 1]), self.user_embeddings_var, batch_neg_num, n_mid,
                        sampled_values=(self.sampled_batch_neg_items, self.pos_q, self.neg_q))
        self.main_labels = tf.stop_gradient(self.main_labels, name="main_labels_stop_gradient")

        # 计算从众部分logits和labels
        self.con_pos_logits, self.con_pos_labels = _compute_sampled_logits(self.item_con_embeddings_var,
                                                                     self.item_embeddings_bias,
                                                                     tf.reshape(self.item_batch_input, [-1, 1]),
                                                                     self.user_con_att_emb, batch_neg_num, n_mid,
                                                                     sampled_values=(
                                                                     self.sampled_batch_neg_items, self.pos_q,
                                                                     self.neg_q))
        self.con_neg_logits, self.con_neg_labels = _compute_sampled_logits(-self.item_con_embeddings_var,
                                                                           self.item_embeddings_bias,
                                                                           tf.reshape(self.item_batch_input, [-1, 1]),
                                                                           self.user_con_att_emb, batch_neg_num, n_mid,
                                                                           sampled_values=(
                                                                               self.sampled_batch_neg_items, self.pos_q,
                                                                               self.neg_q))
        # 计算兴趣部分logits和labels
        self.int_pos_logits, self.int_pos_labels = _compute_sampled_logits(self.item_int_embeddings_var,
                                                                           self.item_embeddings_bias,
                                                                           tf.reshape(self.item_batch_input, [-1, 1]),
                                                                           self.user_int_att_emb, batch_neg_num, n_mid,
                                                                           sampled_values=(
                                                                               self.sampled_batch_neg_items, self.pos_q,
                                                                               self.neg_q))
        self.int_neg_logits, self.int_neg_labels = _compute_sampled_logits(-self.item_int_embeddings_var,
                                                                           self.item_embeddings_bias,
                                                                           tf.reshape(self.item_batch_input, [-1, 1]),
                                                                           self.user_int_att_emb, batch_neg_num, n_mid,
                                                                           sampled_values=(
                                                                               self.sampled_batch_neg_items, self.pos_q,
                                                                               self.neg_q))

        # ----辅助loss部分-----
        # 根据输入的一个batch的item_id从item_int_embeddings_var中lookup (batch_size * neg_num, embedding_dim)
        self.batch_neg_item_int_embeddings = tf.nn.embedding_lookup(self.item_int_embeddings_var, self.sampled_batch_neg_items)

        # 根据输入的一个batch的item_id从item_con_embeddings_var中lookup (batch_size * neg_num, embedding_dim)
        self.batch_neg_item_con_embeddings = tf.nn.embedding_lookup(self.item_con_embeddings_var, self.sampled_batch_neg_items)

        # 正样本的频率 (batch_size, )
        self.batch_pos_freq = self.item_freq_dict.lookup(self.batch_pos_items)

        # 负样本的频率 (batch_size * neg_num, )
        self.batch_neg_freq = self.item_freq_dict.lookup(tf.cast(self.sampled_batch_neg_items, tf.int32))

        # 历史序列的类别矩阵 (batch_size, max_len)
        self.batch_hist_item_cate = self.item_cate_dict.lookup(self.item_his_batch_input)

        # 正样本的类别矩阵 (batch_size, )
        self.batch_pos_item_cate = self.item_cate_dict.lookup(self.batch_pos_items)

        # 负样本的类别矩阵 (neg_num * batch_size, )
        self.batch_neg_item_cate = self.item_cate_dict.lookup(tf.cast(self.sampled_batch_neg_items, tf.int32))

        # 拆分因果数据集用mask表示 (batch_size, batch_size * neg_num)
        self.neg_all_one_masks = tf.ones((self.batch_size, self.batch_size * self.neg_num))
        self.neg_all_zero_masks = tf.zeros((self.batch_size, self.batch_size * self.neg_num))
        self.neg_all_neg_ones_masks = -tf.ones((self.batch_size, self.batch_size * self.neg_num))

        # 频率部分
        # self.batch_pos_freq_tile: (batch_size, batch_size * neg_num)
        self.batch_pos_freq_tile = tf.tile(tf.reshape(self.batch_pos_freq, (self.batch_size, 1)),
                                           (1, self.batch_size * self.neg_num))
        # self.batch_neg_freq_tile: (batch_size, batch_size * neg_num)
        self.batch_neg_freq_tile = tf.tile(tf.reshape(self.batch_neg_freq, (1, self.batch_size * self.neg_num)),
                                           (self.batch_size, 1))

        # 类别部分
        # batch_pos_item_cate_tile: (batch_size, seq_len)
        self.batch_pos_item_cate_tile = tf.tile(tf.reshape(self.batch_pos_item_cate, (self.batch_size, 1)),
                                                (1, self.seq_len))
        # batch_neg_item_cate_tile: (neg_num * batch_size, batch_size, seq_len)
        self.batch_neg_item_cate_tile = tf.tile(tf.reshape(self.batch_neg_item_cate, (self.batch_size * self.neg_num, 1, 1)),
                                                (1, self.batch_size, self.seq_len))

        # batch_hist_item_tile: (neg_num * batch_size, batch_size, seq_len)
        self.batch_hist_item_cate_tile = tf.tile(tf.reshape(self.batch_hist_item_cate, (1, self.batch_size, self.seq_len)),
                                            (self.neg_num * self.batch_size, 1, 1))

        # pos_cate_diff (batch_size, seq_len) 如果有为0说明是
        self.batch_pos_hist_item_cate_sub = self.batch_pos_item_cate_tile - self.batch_hist_item_cate

        # neg_cate_diff (neg_num * batch_size, batch_size, seq_len)
        self.batch_neg_hist_item_cate_sub = self.batch_neg_item_cate_tile - self.batch_hist_item_cate_tile

        # neg_cate_equal_-1 如果类别是-1说明负采样到0，会和历史序列中补齐的0冲突，导致属于兴趣，需要做特判, 我们认为该负样本不属于用户兴趣，mask为False
        self.batch_neg_item_cate_tile_only_batch = tf.tile(tf.reshape(self.batch_neg_item_cate,
                                                                      (self.batch_size * self.neg_num, 1)),
                                                           (1, self.batch_size))
        # neg_cate_equal_neg_one: (neg_num * batch_size, batch_size)
        self.neg_cate_equal_neg_one = tf.where(
            tf.equal(self.batch_neg_item_cate_tile_only_batch, -tf.ones_like(self.batch_neg_item_cate_tile_only_batch)),
            tf.fill(self.batch_neg_item_cate_tile_only_batch.shape, False),
            tf.fill(self.batch_neg_item_cate_tile_only_batch.shape, True))

        # 正样本类别和历史序列类别重复mask矩阵 1为相同 0为不相同 (batch_size, seq_len)
        self.pos_hist_same_mask = tf.where(tf.equal(self.batch_pos_hist_item_cate_sub, tf.zeros_like(self.batch_pos_hist_item_cate_sub)),
                                           tf.ones_like(self.batch_pos_hist_item_cate_sub), tf.zeros_like(self.batch_pos_hist_item_cate_sub))

        # 负样本和历史序列类别重复mask矩阵 1为相同 0为不相同 (neg_num * batch_size, batch_size, seq_len)
        self.neg_hist_same_mask = tf.where(tf.equal(self.batch_neg_hist_item_cate_sub, tf.zeros_like(self.batch_neg_hist_item_cate_sub)),
                                           tf.ones_like(self.batch_neg_hist_item_cate_sub), tf.zeros_like(self.batch_neg_hist_item_cate_sub))

        self.mask_sum = tf.reduce_sum(self.mask, axis=1)
        # 防止mask中出现0
        self.mask_sum = tf.where(tf.equal(self.mask_sum, tf.zeros_like(self.mask_sum)), tf.ones_like(self.mask_sum), self.mask_sum)

        # 正样本相同占比概率 (batch_size, )
        self.pos_hist_same_mask = tf.cast(self.pos_hist_same_mask, tf.float32)

        self.pos_hist_same_prop = tf.math.divide(tf.reduce_sum(self.pos_hist_same_mask, axis=1), self.mask_sum)
        # print(self.pos_hist_same_prop)

        # 负样本相同占比概率 (batch_size * neg_num, batch_size)
        self.neg_hist_same_mask = tf.cast(self.neg_hist_same_mask, tf.float32)
        self.neg_hist_same_prop = tf.math.divide(tf.reduce_sum(self.neg_hist_same_mask, axis=2),
                                                        tf.cast(tf.tile(tf.reshape(self.mask_sum, (1, self.batch_size))
                                                                , (self.batch_size * self.neg_num, 1)), tf.float32))
        # 正样本属于用户兴趣mask (batch_size, )
        self.pos_belong_user_int_mask = tf.where(tf.greater(self.pos_hist_same_prop, self.cate_prop * tf.ones_like(self.pos_hist_same_prop)),
                                                 tf.fill(self.pos_hist_same_prop.shape, True),
                                                 tf.fill(self.pos_hist_same_prop.shape, False))
        # (self.neg_num * self.batch_size, batch_size)
        self.pos_belong_user_int_mask_tile = tf.tile(tf.reshape(self.pos_belong_user_int_mask, (1, self.batch_size)),
                                                     (self.neg_num * self.batch_size, 1))


        # 负样本属于用户兴趣mask (batch_size * neg_num, batch_size) prop > cate_prop
        self.neg_belong_user_int_mask = tf.where(tf.greater(self.neg_hist_same_prop, self.cate_prop * tf.ones_like(self.neg_hist_same_prop)),
                                                 tf.fill(self.neg_hist_same_prop.shape, True),
                                                 tf.fill(self.neg_hist_same_prop.shape, False))
        self.neg_belong_user_int_mask = self.neg_belong_user_int_mask & self.neg_cate_equal_neg_one

        self.pos_belong_user_int_mask_tile = tf.transpose(self.pos_belong_user_int_mask_tile, [1, 0])
        self.neg_belong_user_int_mask = tf.transpose(self.neg_belong_user_int_mask, [1, 0])

        # 第一层频数判断 1为正样本大于负样本 -1为正样本小于等于负样本 (batch_size, batch_size * neg_num)
        self.conformity_mask = tf.where(tf.less_equal(self.batch_pos_freq_tile, self.batch_neg_freq_tile),
                                        -tf.ones_like(self.batch_pos_freq_tile),
                                        tf.ones_like(self.batch_pos_freq_tile))

        # 第二层兴趣判断 0为无法判断 1为正样本大于负样本 -1为正样本小于负样本 (batch_size, batch_size * neg_num)

        self.interest_mask = tf.where(tf.equal(self.conformity_mask, -tf.ones_like(self.conformity_mask)),
                                      tf.fill(self.conformity_mask.shape, 1),
                                      tf.where(
                                          tf.equal(self.pos_belong_user_int_mask_tile, self.neg_belong_user_int_mask),
                                          tf.fill(self.pos_belong_user_int_mask_tile.shape, 0),
                                          tf.where(self.pos_belong_user_int_mask_tile,
                                                   tf.fill(self.pos_belong_user_int_mask_tile.shape, 1),
                                                   tf.fill(self.pos_belong_user_int_mask_tile.shape, -1))
                                      ))

        self.conformity_mask = tf.cast(self.conformity_mask, tf.float32)
        self.interest_mask = tf.cast(self.interest_mask, tf.float32)

        # 正样本的mask 和输入的mask进行拼接
        self.mask_pos_gt_neg = tf.ones((batch_size, 1))
        self.mask_pos_lt_neg = -tf.ones((batch_size, 1))

        self.con_pos_mask = tf.concat([self.mask_pos_gt_neg, self.conformity_mask], axis=1)
        self.con_neg_mask = tf.concat([self.mask_pos_lt_neg, self.conformity_mask], axis=1)

        self.con_pos_mask = tf.where(tf.equal(self.con_pos_mask, tf.ones_like(self.con_pos_mask)),
                                     tf.ones_like(self.con_pos_mask),
                                     tf.zeros_like(self.con_pos_mask))
        self.con_neg_mask = tf.where(tf.equal(self.con_neg_mask, -tf.ones_like(self.con_neg_mask)),
                                     tf.ones_like(self.con_neg_mask),
                                     tf.zeros_like(self.con_neg_mask))

        self.int_pos_mask = tf.concat([self.mask_pos_gt_neg, self.interest_mask], axis=1)
        self.int_neg_mask = tf.concat([self.mask_pos_lt_neg, self.interest_mask], axis=1)

        self.int_pos_mask = tf.where(tf.equal(self.int_pos_mask, tf.ones_like(self.int_pos_mask)),
                                     tf.ones_like(self.int_pos_mask),
                                     tf.zeros_like(self.int_pos_mask))
        self.int_neg_mask = tf.where(tf.equal(self.int_neg_mask, -tf.ones_like(self.int_neg_mask)),
                                     tf.ones_like(self.int_neg_mask),
                                     tf.zeros_like(self.int_neg_mask))

        # int和con 区分loss
        self.batch_pos_items = tf.reshape(self.item_batch_input, [-1, ])
        self.batch_neg_items = tf.cast(tf.reshape(self.sampled_batch_neg_items, [-1, ]), tf.int32)
        self.batch_item_all = tf.unique(tf.concat([self.batch_pos_items, self.batch_neg_items], axis=0))[0]

        self.batch_item_int = tf.gather(self.item_int_embeddings_var, self.batch_item_all)
        self.batch_item_con = tf.gather(self.item_con_embeddings_var, self.batch_item_all)

        self.build_loss()


class InterestCapsuleNetwork(tf.layers.Layer):
    def __init__(self, dim, seq_len, bilinear_type=2, num_interest=4, hard_readout=True, relu_layer=False):
        super(InterestCapsuleNetwork, self).__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.bilinear_type = bilinear_type
        self.num_interest = num_interest
        self.hard_readout = hard_readout
        self.relu_layer = relu_layer
        self.stop_grad = True

    def call(self, item_his_emb, mask):
        with tf.variable_scope('bilinear'):
            if self.bilinear_type == 0:
                item_emb_hat = tf.layers.dense(item_his_emb, self.dim, activation=None, bias_initializer=None)
                item_emb_hat = tf.tile(item_emb_hat, [1, 1, self.num_interest])
            elif self.bilinear_type == 1:
                item_emb_hat = tf.layers.dense(item_his_emb, self.dim * self.num_interest, activation=None,
                                               bias_initializer=None)
            else:
                w = tf.get_variable(
                    'weights', shape=[1, self.seq_len, self.num_interest * self.dim, self.dim],
                    initializer=tf.random_normal_initializer())
                # [N, T, 1, C]
                u = tf.expand_dims(item_his_emb, axis=2)
                # [N, T, num_caps * dim_caps]
                item_emb_hat = tf.reduce_sum(w[:, :self.seq_len, :, :] * u, axis=3)

        item_emb_hat = tf.reshape(item_emb_hat, [-1, self.seq_len, self.num_interest, self.dim])
        item_emb_hat = tf.transpose(item_emb_hat, [0, 2, 1, 3])
        item_emb_hat = tf.reshape(item_emb_hat, [-1, self.num_interest, self.seq_len, self.dim])

        if self.stop_grad:
            item_emb_hat_iter = tf.stop_gradient(item_emb_hat, name='item_emb_hat_iter')
        else:
            item_emb_hat_iter = item_emb_hat

        if self.bilinear_type > 0:
            capsule_weight = tf.stop_gradient(tf.zeros([get_shape(item_his_emb)[0], self.num_interest, self.seq_len]))
        else:
            capsule_weight = tf.stop_gradient(
                tf.truncated_normal([get_shape(item_his_emb)[0], self.num_interest, self.seq_len], stddev=1.0))

        for i in range(3):
            atten_mask = tf.tile(tf.expand_dims(mask, axis=1), [1, self.num_interest, 1])
            paddings = tf.zeros_like(atten_mask)

            capsule_softmax_weight = tf.nn.softmax(capsule_weight, axis=1)
            capsule_softmax_weight = tf.where(tf.equal(atten_mask, 0), paddings, capsule_softmax_weight)
            capsule_softmax_weight = tf.expand_dims(capsule_softmax_weight, 2)

            if i < 2:
                interest_capsule = tf.matmul(capsule_softmax_weight, item_emb_hat_iter)
                cap_norm = tf.reduce_sum(tf.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / tf.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

                delta_weight = tf.matmul(item_emb_hat_iter, tf.transpose(interest_capsule, [0, 1, 3, 2]))
                delta_weight = tf.reshape(delta_weight, [-1, self.num_interest, self.seq_len])
                capsule_weight = capsule_weight + delta_weight
            else:
                interest_capsule = tf.matmul(capsule_softmax_weight, item_emb_hat)
                cap_norm = tf.reduce_sum(tf.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / tf.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

        interest_capsule = tf.reshape(interest_capsule, [-1, self.num_interest, self.dim])

        return interest_capsule


# class Model_MiceRecDR(Basic_Model):
    """
    Due to the time conflict of the company's patent application, 
    the details of the model part are temporarily invisible. 
    We will add all the codes after the patent application is successful. 
    It is expected to be March 2022.
    Thanks for your patience.
    """