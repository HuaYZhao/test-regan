import tensorflow as tf
import numpy as np


def relation_embedding(bag_rel, rel_tot, embedding_dim=5, var_scope=None):
    with tf.variable_scope(var_scope or 'relation_embedding', reuse=tf.AUTO_REUSE):
        relation_matrix = tf.get_variable('relation_matrix', shape=[rel_tot, embedding_dim], dtype=tf.float32,
                                          initializer=tf.contrib.layers.xavier_initializer())
        bag_rel = tf.cast(bag_rel, dtype=tf.float32)
        x = tf.matmul(bag_rel, relation_matrix)
    return x


def init_relation_embedding(train_data_loader, var_scope=None, relation_embedding_dim=50, level=1):
    rel2id = train_data_loader.rel2id
    word2id = train_data_loader.word2id
    word_vec_mat = train_data_loader.word_vec_mat
    with tf.variable_scope(var_scope or 'relation_embedding', reuse=tf.AUTO_REUSE):
        rel_tol = len(rel2id)
        relation_embedding = np.zeros(shape=[rel_tol, relation_embedding_dim], dtype=np.float)

        for rel, idx in rel2id.items():
            rel_key = rel.split('/')[-1]
            if rel_key in word2id:
                key_id = word2id[rel_key]
                relation_embedding[idx, :] = word_vec_mat[key_id, :]
            else:
                relation_embedding[idx, :] = np.random.normal(scale=0.1, size=relation_embedding_dim)
        x = tf.get_variable('relation_embedding', initializer=relation_embedding.astype(np.float32), dtype=tf.float32)
    return x


def word_embedding(word, word_vec_mat, var_scope=None, word_embedding_dim=50, add_unk_and_blank=True):
    with tf.variable_scope(var_scope or 'word_embedding', reuse=tf.AUTO_REUSE):
        word_embedding = tf.get_variable('word_embedding', initializer=word_vec_mat, dtype=tf.float32)
        if add_unk_and_blank:
            word_embedding = tf.concat([word_embedding,
                                        tf.get_variable("unk_word_embedding", [1, word_embedding_dim], dtype=tf.float32,
                                                        initializer=tf.contrib.layers.xavier_initializer()),
                                        tf.constant(np.zeros((1, word_embedding_dim), dtype=np.float32))], 0)
        x = tf.nn.embedding_lookup(word_embedding, word)
        return x


def pos_embedding(pos1, pos2, var_scope=None, pos_embedding_dim=5, max_length=120):
    with tf.variable_scope(var_scope or 'pos_embedding', reuse=tf.AUTO_REUSE):
        pos_tot = max_length * 2

        pos1_embedding = tf.get_variable('real_pos1_embedding', [pos_tot, pos_embedding_dim], dtype=tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer())
        # pos1_embedding = tf.concat([tf.zeros((1, pos_embedding_dim), dtype=tf.float32), real_pos1_embedding], 0)
        pos2_embedding = tf.get_variable('real_pos2_embedding', [pos_tot, pos_embedding_dim], dtype=tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer())
        # pos2_embedding = tf.concat([tf.zeros((1, pos_embedding_dim), dtype=tf.float32), real_pos2_embedding], 0)

        input_pos1 = tf.nn.embedding_lookup(pos1_embedding, pos1)
        input_pos2 = tf.nn.embedding_lookup(pos2_embedding, pos2)
        x = tf.concat([input_pos1, input_pos2], -1)
        return x


def rel_embedding(rel, relation_embedding, var_scope=None, embedding_dim=50, level=1):
    if len(rel.shape) != 2:
        raise NotImplementedError
    with tf.variable_scope(var_scope or 'relation_embedding', reuse=tf.AUTO_REUSE):
        relation_embedding = tf.get_variable('relation_embedding', initializer=relation_embedding, dtype=tf.float32)
        x = tf.matmul(rel, relation_embedding)
        # x = tf.expand_dims(rel, 1)
        # x = tf.tile(x, [1, embedding_dim, 1])
        # x = tf.map_fn(lambda x_: tf.reduce_sum(relation_embedding * tf.transpose(x_), 0), x)
        return x


def word_position_embedding(word, word_vec_mat, pos1, pos2, var_scope=None, word_embedding_dim=50, pos_embedding_dim=5,
                            max_length=120, add_unk_and_blank=True):
    w_embedding = word_embedding(word, word_vec_mat, var_scope=var_scope, word_embedding_dim=word_embedding_dim,
                                 add_unk_and_blank=add_unk_and_blank)
    p_embedding = pos_embedding(pos1, pos2, var_scope=var_scope, pos_embedding_dim=pos_embedding_dim,
                                max_length=max_length)
    return tf.concat([w_embedding, p_embedding], -1)
