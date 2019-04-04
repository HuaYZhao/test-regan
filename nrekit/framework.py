import tensorflow as tf
import os
import sklearn.metrics
import numpy as np
import sys
import time
import logging

logging.basicConfig(filename='regan.log', level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(message)s', datefmt='%Y-%m-%d')


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
    Returns:
         List of pairs of (gradient, variable) where the gradient has been averaged
         across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #     ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


class re_model:
    def __init__(self, train_data_loader, batch_size, max_length=120, name=None):
        self.word = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name='word')
        self.pos1 = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name='pos1')
        self.pos2 = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name='pos2')
        self.label = tf.placeholder(dtype=tf.int32, shape=[batch_size], name='label')
        self.ins_label = tf.placeholder(dtype=tf.int32, shape=[None], name='ins_label')
        self.length = tf.placeholder(dtype=tf.int32, shape=[None], name='length')
        self.scope = tf.placeholder(dtype=tf.int32, shape=[batch_size, 2], name='scope')
        self.train_data_loader = train_data_loader
        self.rel_tot = train_data_loader.rel_tot
        self.word_vec_mat = train_data_loader.word_vec_mat
        self.name = name

    def loss(self):
        raise NotImplementedError

    def train_logit(self):
        raise NotImplementedError

    def test_logit(self):
        raise NotImplementedError

    def get_trainable_variables(self):
        t_vars = tf.trainable_variables()
        t_vars_model = {v.name: v for v in t_vars if self.name in v.name}
        return t_vars_model


class re_framework:
    MODE_BAG = 0  # Train and test the model at bag level.
    MODE_INS = 1  # Train and test the model at instance level

    def __init__(self, train_data_loader, test_data_loader, max_length=120, batch_size=160):
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.sess = None

    def one_step_multi_models(self, sess, models, batch_data_gen, run_array, return_label=True):
        feed_dict = {}
        batch_label = []
        for model in models:
            batch_data = batch_data_gen.next_batch(batch_data_gen.batch_size // len(models))
            feed_dict.update({
                model.word: batch_data['word'],
                model.pos1: batch_data['pos1'],
                model.pos2: batch_data['pos2'],
                model.label: batch_data['rel'],
                model.ins_label: batch_data['ins_rel'],
                model.scope: batch_data['scope'],
                model.length: batch_data['length'],
            })
            if 'mask' in batch_data and hasattr(model, "mask"):
                feed_dict.update({model.mask: batch_data['mask']})
            batch_label.append(batch_data['rel'])
        result = sess.run(run_array, feed_dict)
        batch_label = np.concatenate(batch_label)
        if return_label:
            result += [batch_label]
        return result

    def one_step(self, sess, model, batch_data, run_array):
        feed_dict = {
            model.word: batch_data['word'],
            model.pos1: batch_data['pos1'],
            model.pos2: batch_data['pos2'],
            model.label: batch_data['rel'],
            model.ins_label: batch_data['ins_rel'],
            model.scope: batch_data['scope'],
            model.length: batch_data['length'],
        }
        if 'mask' in batch_data and hasattr(model, "mask"):
            feed_dict.update({model.mask: batch_data['mask']})
        result = sess.run(run_array, feed_dict)
        return result

    def train(self,
              model,
              model_name,
              ckpt_dir='./checkpoint',
              summary_dir='./summary',
              test_result_dir='./test_result',
              learning_rate=0.5,
              max_epoch=60,
              pretrain_model=None,
              test_epoch=1,
              optimizer=tf.train.GradientDescentOptimizer,
              gpu_nums=1):

        assert (self.train_data_loader.batch_size % gpu_nums == 0)
        print("Start training...")

        # Init
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        optimizer = optimizer(learning_rate)

        # Multi GPUs
        tower_grads = []
        tower_models = []
        for gpu_id in range(gpu_nums):
            with tf.device("/gpu:%d" % gpu_id):
                with tf.name_scope("gpu_%d" % gpu_id):
                    cur_model = model(self.train_data_loader, self.train_data_loader.batch_size // gpu_nums,
                                      self.train_data_loader.max_length)
                    tower_grads.append(optimizer.compute_gradients(cur_model.loss()))
                    tower_models.append(cur_model)
                    tf.add_to_collection("loss", cur_model.loss())
                    tf.add_to_collection("train_logit", cur_model.train_logit())

        loss_collection = tf.get_collection("loss")
        loss = tf.add_n(loss_collection) / len(loss_collection)
        train_logit_collection = tf.get_collection("train_logit")
        train_logit = tf.concat(train_logit_collection, 0)

        grads = average_gradients(tower_grads)
        train_op = optimizer.apply_gradients(grads)
        summary_writer = tf.summary.FileWriter(summary_dir, self.sess.graph)

        # Saver
        saver = tf.train.Saver(max_to_keep=None)
        if pretrain_model is None:
            self.sess.run(tf.global_variables_initializer())
        else:
            saver.restore(self.sess, pretrain_model)

        # Training
        best_metric = 0
        best_prec = None
        best_recall = None
        not_best_count = 0  # Stop training after several epochs without improvement.
        for epoch in range(max_epoch):
            print('###### Epoch ' + str(epoch) + ' ######')
            tot_correct = 0
            tot_not_na_correct = 0
            tot = 0
            tot_not_na = 0
            i = 0
            time_sum = 0
            while True:
                time_start = time.time()
                try:
                    iter_loss, iter_logit, _train_op, iter_label = self.one_step_multi_models(self.sess, tower_models,
                                                                                              self.train_data_loader,
                                                                                              [loss, train_logit,
                                                                                               train_op])
                except StopIteration:
                    break
                time_end = time.time()
                t = time_end - time_start
                time_sum += t
                iter_output = iter_logit.argmax(-1)
                iter_correct = (iter_output == iter_label).sum()
                iter_not_na_correct = np.logical_and(iter_output == iter_label, iter_label != 0).sum()
                tot_correct += iter_correct
                tot_not_na_correct += iter_not_na_correct
                tot += iter_label.shape[0]
                tot_not_na += (iter_label != 0).sum()
                if tot_not_na > 0:
                    sys.stdout.write("epoch %d step %d time %.2f | loss: %f, not NA accuracy: %f, accuracy: %f\r" % (
                        epoch, i, t, iter_loss, float(tot_not_na_correct) / tot_not_na, float(tot_correct) / tot))
                    sys.stdout.flush()
                i += 1
            print("\nAverage iteration time: %f" % (time_sum / i))

            if (epoch + 1) % test_epoch == 0:
                metric = self.test(model)
                if metric > best_metric:
                    best_metric = metric
                    best_prec = self.cur_prec
                    best_recall = self.cur_recall
                    print("Best model, storing...")
                    if not os.path.isdir(ckpt_dir):
                        os.mkdir(ckpt_dir)
                    path = saver.save(self.sess, os.path.join(ckpt_dir, model_name))
                    print("Finish storing")
                    not_best_count = 0
                else:
                    not_best_count += 1

            if not_best_count >= 20:
                break

        print("######")
        print("Finish training " + model_name)
        print("Best epoch auc = %f" % (best_metric))
        if (not best_prec is None) and (not best_recall is None):
            if not os.path.isdir(test_result_dir):
                os.mkdir(test_result_dir)
            np.save(os.path.join(test_result_dir, model_name + "_x.npy"), best_recall)
            np.save(os.path.join(test_result_dir, model_name + "_y.npy"), best_prec)

    def test(self,
             model,
             ckpt=None,
             return_result=False,
             mode=MODE_BAG):
        if mode == re_framework.MODE_BAG:
            return self.__test_bag__(model, ckpt=ckpt, return_result=return_result)
        elif mode == re_framework.MODE_INS:
            raise NotImplementedError
        else:
            raise NotImplementedError

    def __test_bag__(self, model, ckpt=None, return_result=False):
        print("Testing...")
        if self.sess == None:
            self.sess = tf.Session()
        model = model(self.test_data_loader, self.test_data_loader.batch_size, self.test_data_loader.max_length)
        if not ckpt is None:
            saver = tf.train.Saver()
            saver.restore(self.sess, ckpt)
        tot_correct = 0
        tot_not_na_correct = 0
        tot = 0
        tot_not_na = 0
        entpair_tot = 0
        test_result = []
        pred_result = []

        for i, batch_data in enumerate(self.test_data_loader):
            iter_logit = self.one_step(self.sess, model, batch_data, [model.test_logit()])[0]
            iter_output = iter_logit.argmax(-1)
            iter_correct = (iter_output == batch_data['rel']).sum()
            iter_not_na_correct = np.logical_and(iter_output == batch_data['rel'], batch_data['rel'] != 0).sum()
            tot_correct += iter_correct
            tot_not_na_correct += iter_not_na_correct
            tot += batch_data['rel'].shape[0]
            tot_not_na += (batch_data['rel'] != 0).sum()
            if tot_not_na > 0:
                sys.stdout.write("[TEST] step %d | not NA accuracy: %f, accuracy: %f\r" % (
                    i, float(tot_not_na_correct) / tot_not_na, float(tot_correct) / tot))
                sys.stdout.flush()
            for idx in range(len(iter_logit)):
                for rel in range(1, self.test_data_loader.rel_tot):
                    test_result.append({'score': iter_logit[idx][rel], 'flag': batch_data['multi_rel'][idx][rel]})
                    if batch_data['entpair'][idx] != "None#None":
                        pred_result.append({'score': float(iter_logit[idx][rel]),
                                            'entpair': batch_data['entpair'][idx].encode('utf-8'), 'relation': rel})
                entpair_tot += 1
        sorted_test_result = sorted(test_result, key=lambda x: x['score'])
        prec = []
        recall = []
        correct = 0
        for i, item in enumerate(sorted_test_result[::-1]):
            correct += item['flag']
            prec.append(float(correct) / (i + 1))
            recall.append(float(correct) / self.test_data_loader.relfact_tot)
        auc = sklearn.metrics.auc(x=recall, y=prec)
        print("\n[TEST] auc: {}".format(auc))
        print("Finish testing")
        self.cur_prec = prec
        self.cur_recall = recall

        if not return_result:
            return auc
        else:
            return (auc, pred_result)


class gan_framework:
    MODE_BAG = 0  # Train and test the model at bag level.
    MODE_INS = 1  # Train and test the model at instance level

    def __init__(self, train_data_loader, test_data_loader, max_length=120, batch_size=160):
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.sess = None

    @staticmethod
    def add_gradient_summary(list_gradvar):
        # Add summary for gradients
        for g, v in list_gradvar:
            if g is not None:
                tf.summary.histogram(v.name + "/gradient", g)

    def feed_data(self, models, batch_data_gen):
        feed_dict = {}
        batch_data = batch_data_gen.next_batch(batch_data_gen.batch_size)
        for model in models:
            feed_dict.update({
                model.word: batch_data['word'],
                model.pos1: batch_data['pos1'],
                model.pos2: batch_data['pos2'],
                model.label: batch_data['rel'],
                model.ins_label: batch_data['ins_rel'],
                model.scope: batch_data['scope'],
                model.length: batch_data['length'],
            })
            if 'mask' in batch_data and hasattr(model, "mask"):
                feed_dict.update({model.mask: batch_data['mask']})
        return feed_dict

    def one_step(self, sess, model, batch_data, run_array):
        feed_dict = {
            model.word: batch_data['word'],
            model.pos1: batch_data['pos1'],
            model.pos2: batch_data['pos2'],
            model.label: batch_data['rel'],
            model.ins_label: batch_data['ins_rel'],
            model.scope: batch_data['scope'],
            model.length: batch_data['length'],
        }
        if 'mask' in batch_data and hasattr(model, "mask"):
            feed_dict.update({model.mask: batch_data['mask']})
        result = sess.run(run_array, feed_dict)
        return result

    def pretrain(self):
        pass

    def train(self,
              gen_model,
              disc_model,
              model_name,
              ckpt_dir='./checkpoint',
              summary_dir='./summary',
              test_result_dir='./test_result',
              max_epoch=60,
              pretrain_model=None,
              test_epoch=1,
              gpu_nums=1):

        assert (self.train_data_loader.batch_size % gpu_nums == 0)
        print("Start training...")

        # Init
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.7
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)

        #########################
        # Instantiate generator and discriminator
        #########################

        G = gen_model(self.train_data_loader, self.train_data_loader.batch_size,
                      self.train_data_loader.max_length)
        D = disc_model(self.train_data_loader, self.train_data_loader.batch_size,
                       self.train_data_loader.max_length)

        #########################
        # Instantiate optimizers
        #########################

        G_opt = tf.train.RMSPropOptimizer(learning_rate=2E-4)
        D_opt = tf.train.RMSPropOptimizer(learning_rate=2E-4)
        CLIP = [-0.01, 0.01]

        # #######################
        # Instantiate model outputs
        #########################

        REL_fake = G.train_out()
        D_real, _ = D(D.label)
        D_fake, _ = D(REL_fake, reuse=True)

        ##########################
        # Instantiate losses
        ##########################
        G_loss = - tf.reduce_mean(D_fake)
        D_loss = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real)

        ##########################
        # Compute gradient updates
        ##########################

        dict_G_vars = G.get_trainable_variables()
        G_vars = [dict_G_vars[k] for k in dict_G_vars.keys()]

        dict_D_vars = D.get_trainable_variables()
        D_vars = [dict_D_vars[k] for k in dict_D_vars.keys()]

        G_gradvar = G_opt.compute_gradients(G_loss, var_list=G_vars, colocate_gradients_with_ops=True)
        G_update = G_opt.apply_gradients(G_gradvar, name='G_loss_minimize')

        D_gradvar = D_opt.compute_gradients(D_loss, var_list=D_vars, colocate_gradients_with_ops=True)
        D_update = D_opt.apply_gradients(D_gradvar, name='D_loss_minimize')
        clip_d_op = [var.assign(tf.clip_by_value(var, CLIP[0], CLIP[1])) for var in D_vars]

        #############################
        # Group training ops
        ##############################
        loss_ops = [G_loss, D_loss]

        ##############################
        # Summary ops
        ##############################
        gan_framework.add_gradient_summary(G_gradvar)
        gan_framework.add_gradient_summary(D_gradvar)

        # Add scalar summaries
        tf.summary.scalar("G_loss", G_loss)
        tf.summary.scalar("D_loss", D_loss)

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(summary_dir, self.sess.graph)

        # Saver
        saver = tf.train.Saver(max_to_keep=None)
        if pretrain_model is None:
            self.sess.run(tf.global_variables_initializer())
        else:
            saver.restore(self.sess, pretrain_model)

        # Training
        global_step = tf.Variable(0, trainable=False)
        temp = tf.train.exponential_decay(1.0, global_step, 200, 0.8, staircase=True)
        temp = tf.clip_by_value(temp, 0.001, 1.0)
        global_steps = 0
        for epoch in range(max_epoch):
            logging.info('###### Epoch ' + str(epoch) + ' ######')
            i = 0
            time_sum = 0
            while True:
                time_start = time.time()
                try:
                    feed_data = self.feed_data([G, D], self.train_data_loader)
                    critic_num = 10
                    if i < 25 or i % 500 == 0:
                        critic_num = 25
                    temperature = self.sess.run(temp, feed_dict={global_step: global_steps})
                    feed_data.update({G.temperature: temperature})
                    for di in range(critic_num):
                        _, d_loss = self.sess.run([D_update, D_loss], feed_dict=feed_data)
                        logging.info("d_loss: %f" % d_loss)
                        self.sess.run(clip_d_op)

                    _, g_loss, d_loss, _, d_real, d_fake = self.sess.run(
                        [G_update] + loss_ops + [summary_op] + [D_real, D_fake],
                        feed_dict=feed_data)
                    logging.info("g_loss: %f, d_loss: %f" % (g_loss, d_loss))
                    print("d_real", d_real)
                except StopIteration:
                    break
                time_end = time.time()
                t = time_end - time_start
                time_sum += t
                logging.info("epoch %d step %d time %.2f | loss: \r" % (epoch, i, t))
                global_steps += 1
                i += 1
            logging.info("\nAverage iteration time: %f" % (time_sum / i))

            if (epoch + 1) % test_epoch == 0:
                logging.info("storing...")
                if not os.path.isdir(ckpt_dir):
                    os.mkdir(ckpt_dir)
                path = saver.save(self.sess, os.path.join(ckpt_dir, model_name))
                logging.info("Finish storing")

    def test(self,
             model,
             ckpt=None,
             return_result=False,
             mode=MODE_BAG):
        if mode == re_framework.MODE_BAG:
            return self.__test_bag__(model, ckpt=ckpt, return_result=return_result)
        elif mode == re_framework.MODE_INS:
            raise NotImplementedError
        else:
            raise NotImplementedError

    def __test_bag__(self, model, ckpt=None, return_result=False):
        print("Testing...")
        if self.sess == None:
            self.sess = tf.Session()
        model = model(self.test_data_loader, self.test_data_loader.batch_size, self.test_data_loader.max_length)
        if not ckpt is None:
            saver = tf.train.Saver()
            saver.restore(self.sess, ckpt)
        tot_correct = 0
        tot_not_na_correct = 0
        tot = 0
        tot_not_na = 0
        entpair_tot = 0
        test_result = []
        pred_result = []

        for i, batch_data in enumerate(self.test_data_loader):
            iter_logit = self.one_step(self.sess, model, batch_data, [model.test_out()])[0]
            iter_output = iter_logit.argmax(-1)
            iter_correct = (iter_output == batch_data['rel']).sum()
            iter_not_na_correct = np.logical_and(iter_output == batch_data['rel'], batch_data['rel'] != 0).sum()
            tot_correct += iter_correct
            tot_not_na_correct += iter_not_na_correct
            tot += batch_data['rel'].shape[0]
            tot_not_na += (batch_data['rel'] != 0).sum()
            if tot_not_na > 0:
                sys.stdout.write("[TEST] step %d | not NA accuracy: %f, accuracy: %f\r" % (
                    i, float(tot_not_na_correct) / tot_not_na, float(tot_correct) / tot))
                sys.stdout.flush()
            for idx in range(len(iter_logit)):
                for rel in range(1, self.test_data_loader.rel_tot):
                    test_result.append({'score': iter_logit[idx][rel], 'flag': batch_data['multi_rel'][idx][rel]})
                    if batch_data['entpair'][idx] != "None#None":
                        pred_result.append({'score': float(iter_logit[idx][rel]),
                                            'entpair': batch_data['entpair'][idx].encode('utf-8'), 'relation': rel})
                entpair_tot += 1
        sorted_test_result = sorted(test_result, key=lambda x: x['score'])
        prec = []
        recall = []
        correct = 0
        for i, item in enumerate(sorted_test_result[::-1]):
            correct += item['flag']
            prec.append(float(correct) / (i + 1))
            recall.append(float(correct) / self.test_data_loader.relfact_tot)
        auc = sklearn.metrics.auc(x=recall, y=prec)
        print("\n[TEST] auc: {}".format(auc))
        print("Finish testing")
        self.cur_prec = prec
        self.cur_recall = recall

        if not return_result:
            return auc
        else:
            return (auc, pred_result)
