import sys
sys.path.append('..')
import tensorflow as tf
import pickle
import random
from numpy import linalg
import numpy as np
import math
import os
import utils


if __name__ == '__main__':

    # ################ Prepare Data ###################
    # basic config
    dim = 200                                   # dimension of embedding
    lr_decay_rate = 0.99                        # learning rate decay rate
    batch_size = 1                              # batch size, set to 1 because we use SGD
    learning_rate = 0.01                        # initial learning rate
    labda = 0.0001                              # regularization term labmda
    total_epoch = 50                            # total training epoches
    hownet_filename = 'dataset/hownet.txt'
    comp_filename = 'dataset/all.bin'
    train_filename = 'dataset/train.bin'
    test_filename = 'dataset/test.bin'
    dev_filename = 'dataset/dev.bin'
    embedding_filename = 'dataset/word_embedding.txt'
    sem_embed_filename = 'dataset/sememe_vector.txt'
    logdir_name = 'phrase_sim/SCMSA'

    # load hownet，并把hownet.comp分成test_set和train_set
    hownet = utils.Hownet(hownet_file=hownet_filename, comp_file=comp_filename)
    hownet.build_hownet()
    hownet.token2id()
    hownet.load_split_dataset(train_filename=train_filename, test_filename=test_filename, dev_filename=dev_filename)
    word_embedding_np, hownet = utils.load_word_embedding(embedding_filename, hownet, scale=False)  # load word embedding
    sememe_embedding_np = utils.load_sememe_embedding(sem_embed_filename, hownet, scale=True)  # load sememe embedding
    hownet, wordsim_words = utils.fliter_wordsim_all(hownet)  # remove MWEs in testset
    train_num = len(hownet.comp_train)
    pos_dict, word_remove = utils.load_hownet_pos()
    hownet, cls_dict = utils.divide_data_with_pos(pos_dict, hownet)
    print("number of dataset in training set:{}".format(len(hownet.comp_train)))
    print("number of dataset in test set:{}".format(len(hownet.comp_test)))
    print("number of dataset in dev set:{}".format(len(hownet.comp_dev)))

    if not os.path.exists(logdir_name):
        os.makedirs(logdir_name)
        os.makedirs(os.path.join(logdir_name, 'print_files'))
        os.makedirs(os.path.join(logdir_name, 'model_file'))
        os.makedirs(os.path.join(logdir_name, 'tensorboard_logs'))
        os.makedirs(os.path.join(logdir_name, 'example_files'))
    # ################ Prepare Data ###################

    # ################ Model and Run ###################
    input_word_l = tf.placeholder(tf.int32, shape=[1], name='word_left')
    input_word_r = tf.placeholder(tf.int32, shape=[1], name='word_right')
    input_word_t = tf.placeholder(tf.int32, shape=[1], name='word_truth')
    input_sememes_l = tf.placeholder(tf.int32, shape=[None], name='sememes_left')
    input_sememes_r = tf.placeholder(tf.int32, shape=[None], name='sememes_right')

    word_embedding = tf.Variable(tf.constant(0.0, shape=[word_embedding_np.shape[0], word_embedding_np.shape[1]]),trainable=False,name='word_embed')
    embedding_placeholder = tf.placeholder(tf.float32, [word_embedding_np.shape[0], word_embedding_np.shape[1]])
    embedding_init = word_embedding.assign(embedding_placeholder)

    W_a = tf.Variable(tf.truncated_normal([dim, dim], stddev=0.5), tf.float32, name='W_a')
    b_a = tf.Variable(tf.zeros([1, dim]), tf.float32, name='b_a')
    W_c = tf.Variable(tf.truncated_normal([2 * dim, dim], stddev=1.0), tf.float32, name='W_c')
    b_c = tf.Variable(tf.zeros([1, dim]), tf.float32, name='b_c')
    global_step = tf.Variable(0, trainable=False)
    # regularizer
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(labda)(W_a))
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(labda)(W_c))

    with tf.name_scope('word_embedding'):
        embed_word_r = tf.nn.embedding_lookup(word_embedding, input_word_r)
        embed_word_l = tf.nn.embedding_lookup(word_embedding, input_word_l)
        embed_truth = tf.nn.embedding_lookup(word_embedding, input_word_t)
    with tf.name_scope('sememe_embedding'):
        sememe_embedding = tf.Variable(tf.constant(0.0, shape=[hownet.sem_num, dim]),trainable=True,name='Sememe_embeddings')
        sememe_placeholder = tf.placeholder(tf.float32, [hownet.sem_num, dim])
        sememe_init = sememe_embedding.assign(sememe_placeholder)
        embed_sememe_l = tf.nn.embedding_lookup(sememe_embedding, input_sememes_l)
        embed_sememe_r = tf.nn.embedding_lookup(sememe_embedding, input_sememes_r)
        embed_sememe_r = utils.norm(embed_sememe_r)
        embed_sememe_l = utils.norm(embed_sememe_l)
    # 在att_no_wordvec基础上，把att_no_wordvec的聚合得到的词向量变成预训练的词向量，同时与word embedding一起使用
    with tf.name_scope('attention_left'):
        embed_word_align_r = tf.nn.tanh(tf.matmul(embed_word_r, W_a)+b_a, name='embed_align_r')  # 1 * dim
        att_l = tf.nn.softmax(tf.matmul(embed_sememe_l, tf.transpose(embed_word_align_r)), axis=0)
        embed_sememe_l = att_l * embed_sememe_l
        embed_aggre_word_l_pure = tf.reduce_sum(embed_sememe_l, axis=0, keepdims=True, name='embed_aggre_word_l_pure')
    with tf.name_scope('attention_right'):
        embed_word_align_l = tf.nn.tanh(tf.matmul(embed_word_l, W_a)+b_a, name='embed_align_l')
        att_r = tf.nn.softmax(tf.matmul(embed_sememe_r, tf.transpose(embed_word_align_l)), axis=0)
        embed_sememe_r = att_r * embed_sememe_r
        embed_aggre_word_r_pure = tf.reduce_sum(embed_sememe_r, axis=0, keepdims=True, name='embed_aggre_word_r_pure')
    with tf.name_scope('phrase_embedding'):
        embed_word_whole = embed_word_r + embed_word_l
        embed_sememe_whole = embed_aggre_word_r_pure + embed_aggre_word_l_pure
        phrase_vec = tf.nn.tanh(tf.matmul(tf.concat([embed_word_whole, embed_sememe_whole], 1), W_c)+b_c, name="phrase_vec")
    with tf.name_scope('loss'):
        loss_pure = tf.reduce_mean((phrase_vec - embed_truth) ** 2)
        tf.summary.scalar("loss pure", loss_pure)
        tf.add_to_collection("losses", loss_pure)
        cross_entropy_mean = tf.add_n(tf.get_collection("losses"))  # 只是变量名叫cross_entropy_mean, 实际上是loss_all
        tf.summary.scalar('loss_all', cross_entropy_mean)

    lr = tf.train.exponential_decay(learning_rate, global_step=global_step, decay_steps=train_num/batch_size,decay_rate=lr_decay_rate, staircase=True)
    tf.summary.scalar("learning_rate", lr)
    opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
    grads_vars = opt.compute_gradients(cross_entropy_mean)
    capped_grads_vars = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in grads_vars]
    train_one_example = opt.apply_gradients(capped_grads_vars)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(embedding_init, feed_dict={embedding_placeholder: word_embedding_np})  # initialize word embedding
        sess.run(sememe_init, feed_dict={sememe_placeholder: sememe_embedding_np})  # initialize sememe embedding

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(logdir_name + '/tensorboard_logs', sess.graph)  # writer of tensorboard
        print_writer_filename = logdir_name + '/print_files/print.txt'  # saver for printing
        saver = tf.train.Saver(max_to_keep=1)  # saver for saving model

        # training
        random.shuffle(hownet.comp_train)
        for epoch in range(total_epoch):
            example_writer_filename = logdir_name + '/example_files/epoch' + str(epoch + 1) + '.txt'  # file for writing examples
            loss_train = 0
            loss_this_epoch = 0
            print("\nEpoch:" + str(epoch + 1))
            for current_num, train_tup in enumerate(hownet.comp_train):
                total_num = epoch * train_num + current_num
                batch_dict = utils.generate_one_example(hownet, train_tup)
                _, summary_i, step_num, loss_i = sess.run([train_one_example, merged, global_step, cross_entropy_mean],
                                                          feed_dict={input_word_l: [batch_dict['wl']],
                                                                     input_word_r: [batch_dict['wr']],
                                                                     input_word_t: [batch_dict['lb']],
                                                                     input_sememes_l: batch_dict['sl'],
                                                                     input_sememes_r: batch_dict['sr'],})
                loss_this_epoch += loss_i
                if current_num % 100 == 0:
                    sys.stdout.flush()
                    sys.stdout.write('\rTraining num: ' + str(current_num) + ' of ' + str(train_num) + ' loss:' + str(loss_this_epoch / (0.1 + current_num)))
                train_writer.add_summary(summary=summary_i, global_step=step_num)
            with open(print_writer_filename, 'a', encoding='utf-8') as fprint:
                fprint.write('epoch: '+str(epoch+1)+' loss:'+str(loss_this_epoch/(0.1+len(hownet.comp_train)))+'\n')
            saver.save(sess, logdir_name + '/model_file/model_ckpt', global_step=epoch + 1)

        # dev
        loss_dev = 0
        for current_num, dev_tup in enumerate(hownet.comp_dev):
            batch_dict = utils.generate_one_example(hownet, dev_tup)
            loss_i = sess.run(cross_entropy_mean,
                              feed_dict={input_word_l: [batch_dict['wl']],
                                         input_word_r: [batch_dict['wr']],
                                         input_word_t: [batch_dict['lb']],
                                         input_sememes_l: batch_dict['sl'],
                                         input_sememes_r: batch_dict['sr'], })
            loss_dev += loss_i
        sys.stdout.flush()
        sys.stdout.write('\nDev set loss:' + str(loss_dev / (0.1 + len(hownet.comp_dev))) + '\n')
        with open(print_writer_filename, 'a', encoding='utf-8') as fprint:
            fprint.write('Dev set loss:' + str(loss_dev / (0.1 + len(hownet.comp_dev))) + '\n')

        # test MWE similarity: write embedding to phrase_vec_file
        number = 0
        phrase_vec_file = os.path.join(logdir_name, 'example_files', 'phrase_vector_epoch.txt')
        for current_num, test_tup in enumerate(hownet.comp_test):
            if test_tup[4] in wordsim_words:
                batch_test = utils.generate_one_example(hownet, test_tup)
                phrase_vector = sess.run(phrase_vec, feed_dict={input_word_l: [batch_test['wl']],
                                                                input_word_r: [batch_test['wr']],
                                                                input_word_t: [batch_test['lb']],
                                                                input_sememes_l: batch_test['sl'],
                                                                input_sememes_r: batch_test['sr'], })
                with open(phrase_vec_file, 'a', encoding='utf-8') as f_phrase_embed:
                    f_phrase_embed.write(test_tup[4] + ' ')
                    phrase_vector = phrase_vector.tolist()[0]
                    phrase_vector = [str(vec) for vec in phrase_vector]
                    f_phrase_embed.write(' '.join(phrase_vector))
                    f_phrase_embed.write('\n')
                number += 1
        print('Have written {} words to phrase_vector.txt'.format(number))
        with open(print_writer_filename, 'a', encoding='utf-8') as fprint:
            fprint.write('Have written {} words to phrase_vector.txt'.format(number))

    train_writer.close()

