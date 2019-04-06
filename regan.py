import nrekit
import numpy as np
import tensorflow as tf
import sys
import os
from distribution import *

dataset_name = 'nyt'
if len(sys.argv) > 1:
    dataset_name = sys.argv[1]
dataset_dir = os.path.join('./data', dataset_name)
if not os.path.isdir(dataset_dir):
    raise Exception("[ERROR] Dataset dir %s doesn't exist!" % (dataset_dir))

train_loader = nrekit.data_loader.json_file_data_loader(os.path.join(dataset_dir, 'train.json'),
                                                        os.path.join(dataset_dir, 'word_vec.json'),
                                                        os.path.join(dataset_dir, 'rel2id.json'),
                                                        mode=nrekit.data_loader.json_file_data_loader.MODE_RELFACT_BAG,
                                                        shuffle=True)
test_loader = nrekit.data_loader.json_file_data_loader(os.path.join(dataset_dir, 'test.json'),
                                                       os.path.join(dataset_dir, 'word_vec.json'),
                                                       os.path.join(dataset_dir, 'rel2id.json'),
                                                       mode=nrekit.data_loader.json_file_data_loader.MODE_ENTPAIR_BAG,
                                                       shuffle=False)

framework = nrekit.framework.gan_framework(train_loader, test_loader)


class Discriminator(nrekit.framework.re_model):
    encoder = "pcnn"
    selector = "att"

    def __init__(self, train_data_loader, batch_size, max_length=120, model_name='discriminator'):
        self.max_length = max_length
        with tf.variable_scope(model_name, reuse=tf.AUTO_REUSE):
            nrekit.framework.re_model.__init__(self, train_data_loader, batch_size, max_length=max_length,
                                               name=model_name)
            self.mask = tf.placeholder(dtype=tf.int32, shape=[None, self.max_length], name="mask")

    def __call__(self, rel, reuse=False):
        """

        :param rel: Integer for class or [Integer]
        :param reuse:
        :return:
        """
        if len(rel.shape) == 1:
            rel = tf.one_hot(rel, self.rel_tot)
        elif len(rel.shape) == 2:
            pass
        elif len(rel.shape) == 3:
            rel = tf.reduce_mean(rel, 0)
        else:
            raise NotImplementedError
        with tf.variable_scope(self.name, reuse=reuse):
            self.relation_embedding = nrekit.network.embedding.init_relation_embedding(self.train_data_loader)
            # Embedding
            x = nrekit.network.embedding.word_position_embedding(self.word, self.word_vec_mat, self.pos1, self.pos2)
            # bag level, Not implemented: consider the instance level
            rel_embedding = nrekit.network.embedding.rel_embedding(rel, self.relation_embedding)

            # Encoder
            if Discriminator.encoder == "pcnn":
                x_train = nrekit.network.encoder.pcnn(x, self.mask, keep_prob=0.5)
                x_test = nrekit.network.encoder.pcnn(x, self.mask, keep_prob=1.0)
            elif Discriminator.encoder == "cnn":
                x_train = nrekit.network.encoder.cnn(x, keep_prob=0.5)
                x_test = nrekit.network.encoder.cnn(x, keep_prob=1.0)
            elif Discriminator.encoder == "rnn":
                x_train = nrekit.network.encoder.rnn(x, self.length, keep_prob=0.5)
                x_test = nrekit.network.encoder.rnn(x, self.length, keep_prob=1.0)
            elif Discriminator.encoder == "birnn":
                x_train = nrekit.network.encoder.birnn(x, self.length, keep_prob=0.5)
                x_test = nrekit.network.encoder.birnn(x, self.length, keep_prob=1.0)
            else:
                raise NotImplementedError

            # Selector
            if Discriminator.selector == "att":
                self._train_logit, train_repre = nrekit.network.selector.bag_attention(x_train, self.scope,
                                                                                       self.ins_label,
                                                                                       self.rel_tot, True,
                                                                                       keep_prob=0.5)
                self._test_logit, test_repre = nrekit.network.selector.bag_attention(x_test, self.scope, self.ins_label,
                                                                                     self.rel_tot, False, keep_prob=1.0)
            elif Discriminator.selector == "ave":
                self._train_logit, train_repre = nrekit.network.selector.bag_average(x_train, self.scope, self.rel_tot,
                                                                                     keep_prob=0.5)
                self._test_logit, test_repre = nrekit.network.selector.bag_average(x_test, self.scope, self.rel_tot,
                                                                                   keep_prob=1.0)
                self._test_logit = tf.nn.softmax(self._test_logit)
            elif Discriminator.selector == "one":
                self._train_logit, train_repre = nrekit.network.selector.bag_one(x_train, self.scope, self.label,
                                                                                 self.rel_tot, True, keep_prob=0.5)
                self._test_logit, test_repre = nrekit.network.selector.bag_one(x_test, self.scope, self.label,
                                                                               self.rel_tot,
                                                                               False, keep_prob=1.0)
                self._test_logit = tf.nn.softmax(self._test_logit)
            elif Discriminator.selector == "cross_max":
                self._train_logit, train_repre = nrekit.network.selector.bag_cross_max(x_train, self.scope,
                                                                                       self.rel_tot,
                                                                                       keep_prob=0.5)
                self._test_logit, test_repre = nrekit.network.selector.bag_cross_max(x_test, self.scope, self.rel_tot,
                                                                                     keep_prob=1.0)
                self._test_logit = tf.nn.softmax(self._test_logit)
            else:
                raise NotImplementedError

            x_train = tf.concat([self._train_logit, rel_embedding], -1)
            x_test = tf.concat([self._test_logit, rel_embedding], -1)
            x_train = nrekit.network.encoder.__linear__(x_train, self.rel_tot, bias=True)
            x_test = nrekit.network.encoder.__linear__(x_test, self.rel_tot, bias=True)
            x_train = tf.nn.relu(x_train)
            x_test = tf.nn.relu(x_test)
            self._train_disc = nrekit.network.encoder.__linear__(x_train, 1, bias=False)
            self._test_disc = nrekit.network.encoder.__linear__(x_test, 1, bias=False)
            return self._train_disc, self._test_disc

    def pretrain_loss(self):
        _loss = nrekit.network.classifier.softmax_cross_entropy(self._train_logit, self.label, self.rel_tot,
                                                                weights_table=self.get_weights())
        return _loss

    def train_out(self):
        return self._train_disc

    def test_out(self):
        return self._test_disc

    def get_weights(self):
        with tf.variable_scope("weights_table", reuse=tf.AUTO_REUSE):
            print("Calculating weights_table...")
            _weights_table = np.zeros((self.rel_tot), dtype=np.float32)
            for i in range(len(self.train_data_loader.data_rel)):
                _weights_table[self.train_data_loader.data_rel[i]] += 1.0
            _weights_table = 1 / (_weights_table ** 0.05)
            weights_table = tf.get_variable(name='weights_table', dtype=tf.float32, trainable=False,
                                            initializer=_weights_table)
            print("Finish calculating")
        return weights_table


class Generator(nrekit.framework.re_model):
    encoder = "pcnn"
    selector = "att"

    def __init__(self, train_data_loader, batch_size, max_length=120, model_name='generator'):
        with tf.variable_scope(model_name, reuse=tf.AUTO_REUSE):
            nrekit.framework.re_model.__init__(self, train_data_loader, batch_size, max_length=max_length,
                                               name=model_name)
            self.mask = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name="mask")
            self.temperature = tf.placeholder(dtype=tf.float32, name="temperature")
            # Embedding
            x = nrekit.network.embedding.word_position_embedding(self.word, self.word_vec_mat, self.pos1, self.pos2)

            # Encoder
            if Generator.encoder == "pcnn":
                x_train = nrekit.network.encoder.pcnn(x, self.mask, keep_prob=0.5)
                x_test = nrekit.network.encoder.pcnn(x, self.mask, keep_prob=1.0)
            elif Generator.encoder == "cnn":
                x_train = nrekit.network.encoder.cnn(x, keep_prob=0.5)
                x_test = nrekit.network.encoder.cnn(x, keep_prob=1.0)
            elif Generator.encoder == "rnn":
                x_train = nrekit.network.encoder.rnn(x, self.length, keep_prob=0.5)
                x_test = nrekit.network.encoder.rnn(x, self.length, keep_prob=1.0)
            elif Generator.encoder == "birnn":
                x_train = nrekit.network.encoder.birnn(x, self.length, keep_prob=0.5)
                x_test = nrekit.network.encoder.birnn(x, self.length, keep_prob=1.0)
            else:
                raise NotImplementedError

            # Selector
            if Generator.selector == "att":
                self._train_logit, train_repre = nrekit.network.selector.bag_attention(x_train, self.scope,
                                                                                       self.ins_label,
                                                                                       self.rel_tot, True,
                                                                                       keep_prob=0.5)
                self._test_logit, test_repre = nrekit.network.selector.bag_attention(x_test, self.scope, self.ins_label,
                                                                                     self.rel_tot, False, keep_prob=1.0)
            elif Generator.selector == "ave":
                self._train_logit, train_repre = nrekit.network.selector.bag_average(x_train, self.scope, self.rel_tot,
                                                                                     keep_prob=0.5)
                self._test_logit, test_repre = nrekit.network.selector.bag_average(x_test, self.scope, self.rel_tot,
                                                                                   keep_prob=1.0)
                self._test_logit = tf.nn.softmax(self._test_logit)
            elif Generator.selector == "one":
                self._train_logit, train_repre = nrekit.network.selector.bag_one(x_train, self.scope, self.label,
                                                                                 self.rel_tot, True, keep_prob=0.5)
                self._test_logit, test_repre = nrekit.network.selector.bag_one(x_test, self.scope, self.label,
                                                                               self.rel_tot,
                                                                               False, keep_prob=1.0)
                self._test_logit = tf.nn.softmax(self._test_logit)
            elif Generator.selector == "cross_max":
                self._train_logit, train_repre = nrekit.network.selector.bag_cross_max(x_train, self.scope,
                                                                                       self.rel_tot,
                                                                                       keep_prob=0.5)
                self._test_logit, test_repre = nrekit.network.selector.bag_cross_max(x_test, self.scope, self.rel_tot,
                                                                                     keep_prob=1.0)
                self._test_logit = tf.nn.softmax(self._test_logit)
            else:
                raise NotImplementedError

    def loss(self):
        return self._loss

    def set_loss(self, loss):
        self._loss = loss

    def train_out(self):
        gumbel_dist = tf.contrib.distributions.RelaxedOneHotCategorical(self.temperature,
                                                                        logits=self._train_logit)
        _train_out = gumbel_dist.sample(1)  # 这里不能reshape，会失去概率意义
        return _train_out

    def test_out(self):
        gumbel_dist = tf.contrib.distributions.RelaxedOneHotCategorical(0.001,
                                                                        logits=self._train_logit)
        _test_out = gumbel_dist.sample(1)
        _test_out = tf.squeeze(_test_out)
        return _test_out


model_file = tf.train.latest_checkpoint('./checkpoint')
framework.train(Generator, Discriminator, "gan_model", max_epoch=60, ckpt_dir="checkpoint", pretrain_model=model_file)

# framework.test(Generator, model_file)
