import sys
sys.path.append('..')
import tensorflow as tf
import pickle
import random
import numpy as np
import math
import os
import re
import utils


if __name__ == '__main__':

    # ################ Prepare Data ###################
    # basic config
    dim = 200                                   # dimension of embedding
    lr_decay_rate = 0.99                        # learning rate decay rate
    batch_size = 1                              # batch size, set to 1 because we use SGD
    learning_rate = 0.2          # initial learning rate  0.2
    total_epoch = 40             # total training epoches  40
    k = 100
    trunc_num = 5
    # initial_seed = 2018
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'

    hownet_filename = 'dataset/hownet.txt'
    comp_filename = 'dataset/all.bin'
    train_filename = 'dataset/train.bin'
    test_filename = 'dataset/test.bin'
    dev_filename = 'dataset/dev.bin'
    embedding_filename = 'dataset/word_embedding.txt'
    sem_embed_filename = 'dataset/sememe_vector.txt'
    logdir_name = 'sememe_prediction/SCAS_R'

    # load hownet，并把hownet.comp分成test_set和train_set
    hownet = utils.Hownet(hownet_file=hownet_filename, comp_file=comp_filename)
    hownet.build_hownet()
    hownet.token2id()
    hownet.load_split_dataset(train_filename=train_filename, test_filename=test_filename, dev_filename=dev_filename)
    word_embedding_np, hownet = utils.load_word_embedding(embedding_filename, hownet, scale=True)  # load word embedding
    sememe_embedding_np = utils.load_sememe_embedding(sem_embed_filename, hownet, scale=True)  # load sememe embedding
    train_num = len(hownet.comp_train)
    pos_dict, word_remove = utils.load_hownet_pos()
    hownet, cls_dict = utils.divide_data_with_pos(pos_dict, hownet)
    print("number of dataset in training set:{}".format(len(hownet.comp_train)))
    print("number of dataset in test set:{}".format(len(hownet.comp_test)))
    print("number of dataset in dev set:{}".format(len(hownet.comp_dev)))

    # # # basic config
    # initial_seed = 2018
    # dim = 200
    # k = 100
    # initial_learning_rate = 0.2
    # lr_decay_rate = 0.99
    # batch_size = 1  # batch_size越小，则α变化越慢，5-100
    # total_epoch = 40
    # dim = int(sys.arg v[1])
    # initial_learning_rate  = float(sys.argv[2])
    # lr_decay_rate = float(sys.argv[3])
    # batch_size = int(sys.argv[4])
    # # cpu_num = int(sys.argv[5])
    # logdir_name = sys.argv[5]

    if not os.path.exists(logdir_name):
        os.makedirs(logdir_name)
        os.makedirs(os.path.join(logdir_name, 'print_files'))
        os.makedirs(os.path.join(logdir_name, 'model_file'))
        os.makedirs(os.path.join(logdir_name, 'tensorboard_logs'))
        os.makedirs(os.path.join(logdir_name, 'example_files'))

    input_word_l = tf.placeholder(tf.int32, shape=[1], name='word_left')
    input_word_r = tf.placeholder(tf.int32, shape=[1], name='word_right')
    input_pos = tf.placeholder(tf.int32, shape=[1], name='pos_tag')
    input_sememes_l = tf.placeholder(tf.int32, shape=[None], name='sememes_left')
    input_sememes_r = tf.placeholder(tf.int32, shape=[None], name='sememes_right')
    answer_sememes = tf.placeholder(tf.int32, shape=[1, None], name='sememes_answer')
    labels = tf.placeholder(tf.float32, shape=[1, hownet.sem_num], name='labels')
    ones = tf.ones([1, hownet.sem_num], dtype=tf.float32)
    tf_dim = tf.constant(math.log(dim), dtype=tf.float32, name='tf_dim')

    word_embedding = tf.Variable(tf.constant(0.0, shape=[word_embedding_np.shape[0], word_embedding_np.shape[1]]),trainable=False,name='word_embed')
    embedding_placeholder = tf.placeholder(tf.float32, [word_embedding_np.shape[0], word_embedding_np.shape[1]])
    embedding_init = word_embedding.assign(embedding_placeholder)
    sememe_embedding = tf.Variable(tf.constant(0.0, shape=[hownet.sem_num, dim]), trainable=True, name='Sememe_embeddings')
    sememe_placeholder = tf.placeholder(tf.float32, [hownet.sem_num, dim])
    sememe_init = sememe_embedding.assign(sememe_placeholder)

    W_a = tf.Variable(tf.truncated_normal([dim, dim], stddev=0.5), tf.float32, name='W_a')
    b_a = tf.Variable(tf.zeros([1, dim]), tf.float32, name='b_a')

    U = tf.Variable(tf.truncated_normal([4, 2 * dim, trunc_num], stddev=0.5), tf.float32, name='U')
    V = tf.Variable(tf.truncated_normal([4, dim, trunc_num], stddev=0.5), tf.float32, name='V')
    U_i = tf.reshape(tf.nn.embedding_lookup(U, input_pos), [2 * dim, trunc_num])
    V_i = tf.reshape(tf.nn.embedding_lookup(V, input_pos), [dim, trunc_num])
    W_c_i = tf.matmul(U_i, V_i, transpose_b=True)
    W_c_base = tf.Variable(tf.truncated_normal([2 * dim, dim], stddev=1.0), tf.float32, name='W_c_base')
    b_c = tf.Variable(tf.zeros([1, dim]), tf.float32, name='b_c')
    global_step = tf.Variable(0, trainable=False)

    # w_k_init_eye = tf.tile(tf.reshape(tf.eye(dim), [1,dim,dim]), [4,2,1])
    # w_k_init_gaussian = tf.reshape(tf.truncated_normal([8*dim,dim], stddev=0.1), [4, 2*dim, dim])
    # W_c = tf.Variable(w_k_init_eye+w_k_init_gaussian, tf.float32, name='W_c')
    # W_c_i = tf.reshape(tf.nn.embedding_lookup(W_c, input_pos), [2*dim, dim])
    # b_c = tf.Variable(tf.zeros([1, dim]), tf.float32, name='b_c')
    # global_step = tf.Variable(0, trainable=False)

    with tf.name_scope('word_embedding'):
        embed_word_r = tf.nn.embedding_lookup(word_embedding, input_word_r)
        embed_word_l = tf.nn.embedding_lookup(word_embedding, input_word_l)
    with tf.name_scope('sememe_embedding'):
        embed_sememe_l = tf.nn.embedding_lookup(sememe_embedding, input_sememes_l)
        embed_sememe_r = tf.nn.embedding_lookup(sememe_embedding, input_sememes_r)
        embed_sememe_r = utils.norm(embed_sememe_r)
        embed_sememe_l = utils.norm(embed_sememe_l)
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
        phrase_vec = tf.nn.tanh(tf.matmul(tf.concat([embed_word_whole, embed_sememe_whole], 1), W_c_i)+b_c, name="phrase_vec")
    with tf.name_scope('output_layer'):
        y_hat = tf.matmul(phrase_vec, tf.transpose(sememe_embedding))
    with tf.name_scope('cross_entropy_loss'):
        cross_entropy = tf.nn.weighted_cross_entropy_with_logits(logits=y_hat, targets=labels, pos_weight=k, name='cross_entropy')
        losses_pure = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
        tf.summary.scalar("loss pure", losses_pure)
        tf.add_to_collection("losses", losses_pure)
        cross_entropy_mean = tf.add_n(tf.get_collection("losses"))  # 只是变量名叫cross_entropy_mean, 实际上是loss_all
        tf.summary.scalar('loss_all', cross_entropy_mean)

    rank = tf.nn.top_k(tf.nn.sigmoid(y_hat), k=hownet.sem_num, name='rank')
    lr = tf.train.exponential_decay(learning_rate, global_step=global_step, decay_steps=train_num / batch_size,decay_rate=lr_decay_rate, staircase=True)
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
        saver = tf.train.Saver(max_to_keep=3)  # saver for saving model

        # 这四个参数用来判断是否终止训练
        # these 4 params. are used for deciding whether to stop training
        last_map = 0
        last_last_map = 10
        now_map = 100
        jump2test = False

        random.shuffle(hownet.comp_train)
        for epoch in range(total_epoch):
            example_writer_filename = logdir_name + '/example_files/epoch' + str(epoch + 1) + '.txt'  # file for writing examples
            # train process
            maps_train = []
            loss_train = 0
            for current_num, train_tup in enumerate(hownet.comp_train):
                total_num = epoch * train_num + current_num
                batch_dict = utils.generate_one_example4sememe_prediction(hownet, train_tup)
                _, summary_i, step_num, loss_i, rank_res = sess.run(
                    [train_one_example, merged, global_step, cross_entropy_mean, rank],
                    feed_dict={input_word_l: [batch_dict['wl']],
                               input_word_r: [batch_dict['wr']],
                               input_pos:[batch_dict['pos']],
                               input_sememes_l: batch_dict['sl'],
                               input_sememes_r: batch_dict['sr'],
                               labels: batch_dict['lb'],
                               answer_sememes: batch_dict['al']})
                map_score = utils.cal_map_one(batch_dict['al'], rank_res[1])
                maps_train.append(map_score)
                loss_train += loss_i
                if current_num % 100 == 0:
                    sys.stdout.flush()
                    sys.stdout.write('\rTraining num: ' + str(current_num) + ' of ' + str(train_num) + '.Epoch:' + str(epoch + 1))
                    loss_train = 0
                train_writer.add_summary(summary=summary_i, global_step=step_num)
            saver.save(sess, logdir_name + '/model_file/model_ckpt', global_step=epoch + 1)

            # dev set test
            maps_dev = []
            maps_dev_class = {0:[],1:[],2:[],3:[]}
            loss_dev = 0
            for current_num, dev_tup in enumerate(hownet.comp_dev):
                batch_dev = utils.generate_one_example4sememe_prediction(hownet, dev_tup)
                rank_res, loss_i = sess.run([rank, cross_entropy_mean],
                                            feed_dict={input_word_l: [batch_dev['wl']],
                                                       input_word_r: [batch_dev['wr']],
                                                       input_pos: [batch_dev['pos']],
                                                       input_sememes_l: batch_dev['sl'],
                                                       input_sememes_r: batch_dev['sr'],
                                                       labels: batch_dev['lb'],
                                                       answer_sememes: batch_dev['al']})
                map_score = utils.cal_map_one(batch_dev['al'], rank_res[1])
                maps_dev.append(map_score)
                maps_dev_class[dev_tup[6]].append(map_score)
                loss_dev += loss_i
            print('Loss(dev )in epoch %d : %f' % (epoch + 1, loss_dev / len(hownet.comp_dev)))
            print('MAP(dev) in epoch %d : %f' % (epoch + 1, sum(maps_dev) / float(len(hownet.comp_dev))))
            for clss, maps in maps_dev_class.items():
                print('MAP(dev) classfied:{}:{}'.format(cls_dict[clss], sum(maps)/float(len(maps))))
            print("*************DEV END*************\n")
            # write log file

            with open(print_writer_filename, 'a', encoding='utf-8') as fp:
                fp.write('\nLoss(dev)in epoch %d:\t%f' % (epoch + 1, loss_dev / len(hownet.comp_dev)))
                fp.write('\nMAP(dev ) in epoch %d:\t%f' % (epoch + 1, sum(maps_dev) / float(len(hownet.comp_dev))))
                for clss, maps in maps_dev_class.items():
                    fp.write('\nMAP(dev) classfied:{}:{}'.format(cls_dict[clss], sum(maps) / float(len(maps))))
                fp.write("**************DEV END*************\n")

            # 判断终止条件：至少训练了20个epoch后，若在开发集上，连续两次map值上升，则终止；
            # Deciding the stop condition: After training at least 20 epochs,
            # if the MAP value rises twice in the development set, it stops;
            if epoch+1 == 20:
                last_map = sum(maps_dev) / float(len(hownet.comp_dev))
            if epoch+1 == 21:
                last_last_map = last_map
                last_map = sum(maps_dev) / float(len(hownet.comp_dev))
            elif epoch+1 >= 22:
                now_map = sum(maps_dev) / float(len(hownet.comp_dev))
                if now_map <= last_map <= last_last_map:
                    jump2test = True
                else:
                    last_last_map = last_map
                    last_map = now_map
            if epoch+1 >= 40:
                jump2test = True

            if jump2test:
                model_file = os.path.join(logdir_name, 'model_file')
                if not os.path.exists(model_file):
                    print("WARNING: path doesn't exist!")
                    sys.exit(0)
                files = os.listdir(model_file)
                third_last = 'model_ckpt-99'
                for _model in files:
                    if _model != 'checkpoint':
                        _model = _model[:13]
                        if _model < third_last:  # 加载倒数第二次map最大的那个文件
                            third_last = _model
                epoch = int(third_last[11:13]) - 1
                meta_file = os.path.join(model_file, third_last + '.meta')
                data_file = os.path.join(model_file, third_last + '.data-00000-of-00001')
                phrase_vec_file = os.path.join(logdir_name, 'example_files', 'phrase_vector.txt')
                third_last = os.path.join(model_file, third_last)

                saver.restore(sess, third_last)

                # train set test
                maps_test = []
                maps_train_class = {0: [], 1: [], 2: [], 3: []}
                loss_test = 0
                for current_num, train_tup in enumerate(hownet.comp_train):
                    batch_train = utils.generate_one_example4sememe_prediction(hownet, train_tup)

                    rank_res, loss_i, att_left, att_right = sess.run([rank, cross_entropy_mean, att_l, att_r],
                                                                     feed_dict={input_word_l: [batch_train['wl']],
                                                                                input_word_r: [batch_train['wr']],
                                                                                input_pos: [batch_train['pos']],
                                                                                input_sememes_l: batch_train['sl'],
                                                                                input_sememes_r: batch_train['sr'],
                                                                                labels: batch_train['lb'],
                                                                                answer_sememes: batch_train['al']})
                    map_score = utils.cal_map_one(batch_train['al'], rank_res[1])
                    maps_test.append(map_score)
                    maps_train_class[train_tup[6]].append(map_score)
                    loss_test += loss_i

                print("************TRAIN START*************")
                print('Loss(train )in epoch %d : %f' % (epoch + 1, loss_test / len(hownet.comp_train)))
                print('MAP(train) in epoch %d : %f' % (epoch + 1, sum(maps_test) / float(len(hownet.comp_train))))
                for clss, maps in maps_train_class.items():
                    print('MAP(train) classfied:{}:{}'.format(cls_dict[clss], sum(maps) / float(len(maps))))
                print("************TRAIN END***************\n")
                # write log file
                with open(print_writer_filename, 'a', encoding='utf-8') as fp:
                    fp.write("\n************TRAIN START*************\n")
                    fp.write('Loss(train)in epoch %d : %f' % (epoch+1, loss_test/len(hownet.comp_train)))
                    fp.write('\nMAP(train) in epoch %d : %f' % (epoch+1,sum(maps_test)/float(len(hownet.comp_train))))
                    for clss, maps in maps_train_class.items():
                        fp.write('MAP(train) classfied:{}:{}'.format(cls_dict[clss], sum(maps) / float(len(maps))))
                    fp.write("************TRAIN END***************\n")

                # test set test
                maps_test = []
                maps_test_class = {0: [], 1: [], 2: [], 3: []}
                loss_test = 0
                for current_num, test_tup in enumerate(hownet.comp_test):
                    batch_test = utils.generate_one_example4sememe_prediction(hownet, test_tup)
                    rank_res, loss_i, att_left, att_right = sess.run([rank, cross_entropy_mean, att_l, att_r],
                                                                     feed_dict={input_word_l: [batch_test['wl']],
                                                                                input_word_r: [batch_test['wr']],
                                                                                input_pos: [batch_test['pos']],
                                                                                input_sememes_l: batch_test['sl'],
                                                                                input_sememes_r: batch_test['sr'],
                                                                                labels: batch_test['lb'],
                                                                                answer_sememes: batch_test['al']})
                    map_score = utils.cal_map_one(batch_test['al'], rank_res[1])
                    maps_test.append(map_score)
                    maps_test_class[test_tup[6]].append(map_score)
                    _, test_predict = hamming_loss(batch_test['al'], rank_res[1], get_answer=True, predict_num=hownet.sem_num)
                    loss_test += loss_i
                    if len(test_predict) != 0:
                        test_predict_str = predictlabel2char(hownet.id2sememe, test_predict)
                        with open(example_writer_filename, 'a', encoding='utf-8') as ex:
                            ex.write(test_tup[4] + '\n\t')
                            for s in test_predict_str['truth']:
                                ex.write(s + ' ')
                            ex.write('\n\t')
                            for s in test_predict_str['predict']:
                                ex.write(s + ' ')
                            ex.write('\n')

                print("************TEST START*************")
                print('Loss(test )in epoch %d : %f'%(epoch+1,loss_test/len(hownet.comp_test)))
                print('MAP(test) in epoch %d : %f'%(epoch+1,sum(maps_test)/float(len(hownet.comp_test))))
                for clss, maps in maps_test_class.items():
                    print('MAP(test) classfied:{}:{}'.format(cls_dict[clss], sum(maps) / float(len(maps))))
                print("************TEST END***************\n")
                # write log file
                with open(print_writer_filename, 'a', encoding='utf-8') as fp:
                    fp.write("\n************TEST START*************\n")
                    fp.write('Loss(test )in epoch %d : %f' % (epoch + 1, loss_test / len(hownet.comp_test)))
                    fp.write('\nMAP(test) in epoch %d : %f' % (
                    epoch + 1, sum(maps_test) / float(len(hownet.comp_test))))
                    for clss, maps in maps_test_class.items():
                        fp.write('MAP(test) classfied:{}:{}'.format(cls_dict[clss], sum(maps) / float(len(maps))))
                    fp.write("************TEST END***************\n")
                break

    train_writer.close()

