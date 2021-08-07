import os 
import argparse
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pickle

import tensorflow as tf


# precoessing of the image 
def preprocess( img_data ):
    
    # converting unit8s to float
    img_data = img_data.astype( float )

    # scaling
    img_data = img_data / 255

    # normalize: subtracting the mean 
    img_data = img_data - np.mean( img_data )

    return img_data 

# specific conv2d: This is the specific conv2d function. with stride as 1
def conv2d( x, W):
    # stride 1: [1, x_move, y_move, 1] ==> x_move, y_move = 1, 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], 
                              padding='VALID' )

# specific max pooling: 
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], 
                           strides=[1, 2, 2, 1], padding='VALID')

def cal_accuracy( y_pred, y_label):
    acc = sum(y_pred==y_label)/len(y_pred)
    return acc

# plot acc for result
def plot_acc_result( train_acc_history, test_acc_history  ):
    plt.style.use('ggplot')  
    fig = plt.figure()
    plt.plot( train_acc_history) 
    plt.plot( test_acc_history) 
    plt.xlabel( 'iterations' )
    plt.ylabel( 'accuracy')
    plt.legend( [' train', 'test'], loc=4 )
    plt.title( 'accuracy train vs. test')
    return fig 

# plot loss
def plot_loss_result( train_acc_history, test_acc_history  ):
    plt.style.use('ggplot')  
    fig = plt.figure()
    plt.plot( train_acc_history) 
    plt.plot( test_acc_history) 
    plt.xlabel( 'iterations' )
    plt.ylabel( 'loss')
    plt.legend( [' train', 'test'], loc=4 )
    plt.title( 'loss train vs. test')
    return fig 

def plot_twin_result( train_loss_history, train_acc_history, test_acc_history):
    fig = plt.figure(figsize = [7,5])
    plt.style.use('seaborn-bright')
    x = range( max_epoch)

    ax1 = fig.add_subplot(111)
    ax1.plot(x, train_loss_history,'b')
    ax1.set_ylabel('loss')
    ax1.set_title("Overall training details")
    ax1.set_xlabel('epoch')
    ax1.legend( ['train_loss'], loc=[.75,.6])

    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(x, train_acc_history, 'r')
    ax2.plot(x, test_acc_history, 'g')
    ax2.set_ylabel('accuracy')
    ax2.legend( ['train_acc', 'test_acc'], loc=[.75,.49] )
    return fig

# compute confusion matrix         
def confusion_matrix( y_true, y_pred_cls ):
    cls_num = np.max( y_true ) + 1
    cm = np.empty( [cls_num, cls_num] )
    for i in range( cls_num ):
        for j in range( cls_num ):
            yi_num = len(y_true[y_true==i])
            yi_pred_num = len( y_true[ (y_pred_cls == j) * (y_true==i) ])
            cm[i, j] = yi_pred_num / yi_num
    return cm

# plot confusion matrix 
def plot_confusion_matrix( cm ):
    num_classes = cm.shape[0]
    fig = plt.figure( figsize = [9,9])
    plt.imshow( cm )
    plt.grid(False)
    plt.colorbar( )
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for i in range( num_classes ):
        for j in range( num_classes):
            plt.text(i-.25,j, np.round(cm[i, j],2))
    return fig

# plot the weight
def plot_filter( w ):
    w = w / (np.max(w) - np.min(w))
    fig = plt.figure(figsize=(12,8))
    num = w.shape[3]
    for i in range(num):
        plt.subplot(4, 8, i+1 )
        plt.imshow(w[:,:,:,i])
        plt.axis('off')
        plt.title('filter'+str(i))
    return fig 
if __name__ == '__main__':

    ## 0. HYPERPARAMETER TUNING
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--lr', help='learning rate', type=float, default=0.0001)

    # args = parser.parse_args()

    ## 1. LOAD THE DATA
    with open('cifar_10_tf_train_test.pkl', 'rb') as handle:
        x_train, y_train, x_test, y_test = pickle.load(handle, encoding = 'latin1')

    # precoess
    x_train = preprocess(x_train)
    x_test  = preprocess(x_test)
    y_train = np.array(y_train)

    ## 2. PREPARE THE COMPUTATIONAL GRAPH

    # hyper paramters
    lr          = 0.00075  # learning rate
    batch_size  = 2500      # number fo the sample with a batch  

    # fix some useful values
    n_class       = len(np.unique(y_train)) # number of the class
    save_steps    = 1                       # save the sess every n epoch 
    display_steps = 10                      # print epoch loss 
    max_epoch     = 300                     # max epoch 
    total_samples = x_train.shape[0]        # total sample number
    img_width     = x_train.shape[1]        # width of the image
    img_height    = x_train.shape[2]        # height of the image
    img_channel   = x_train.shape[3]        # num of channel of the image
    batch_num     = total_samples // batch_size # num of mini batches
    train_indices = np.arange(0, total_samples) # indices to help shuffle
    rng = np.random.RandomState( 2020 ) # to help generate random things

    # storages
    train_loss_history = []  # record the train loss
    test_loss_history  = []  # record the test loss
    train_acc_history  = []  # record the train acc
    test_acc_history   = []  # record the test acc 
    test_pred_log = dict()  # save the y_pred for confusion matrix 

    # the method to init the weight
    weight_init = tf.contrib.layers.xavier_initializer()
    bias_init = tf.constant_initializer( 0.)

    # reset the defualt graph and hence
    tf.reset_default_graph()

    # use a graph collection
    tf.get_collection( 'validation_nodes' ) 

    # define placehodler 
    x_input = tf.placeholder( tf.float32, [None, img_width, img_height, img_channel], name='input_img')
    y_target = tf.placeholder( tf.int64, [None,], name='y_target')

    # Forward
    # weight: conv1
    weight_x1 = tf.get_variable( 'weight_x1', 
                                shape = [5, 5, img_channel, 32 ],
                                initializer = weight_init)
    bias_x1 = tf.get_variable( 'bias_x1', shape = [32,],
                                initializer = bias_init ) 
    # h_conv1 = conv2d( X,W) +b
    c1 = conv2d( x_input, weight_x1) + bias_x1 #c1: 28 x 28 x 32
    p1 = max_pool( c1 ) #p1: 14 x 14 x 32

    # weight: conv2
    weight_x2 = tf.get_variable( 'weight_x2', 
                                shape = [5, 5, 32, 32 ],
                                initializer = weight_init)
    bias_x2 = tf.get_variable( 'bias_x2', shape = [32,],
                                initializer = bias_init ) 
    # h_conv1 = conv2d( X,W) +b
    c2 = conv2d( p1, weight_x2) + bias_x2 #c2 no padding: 10 x 10 x 32
    p2 = max_pool( c2 ) #p1: 5 x 5 x 32

    # weight: conv3
    weight_x3 = tf.get_variable( 'weight_x3', 
                                shape = [3, 3, 32, 64 ],
                                initializer = weight_init)
    bias_x3 = tf.get_variable( 'bias_x3', shape = [64,],
                                initializer = bias_init ) 
    # h_conv1 = conv2d( X,W) +b
    c3 = conv2d( p2, weight_x3) + bias_x3 #c3 no padding: 3 x 3 x 64

    # flatten 
    p_vec = tf.reshape( c3, [-1, 3 * 3 * 64]) # batch_size x (5 * 5 * 64)

    # fully connect: 
    weight = tf.get_variable( 'weight', 
                                shape = [3 * 3 * 64, n_class],
                                initializer = weight_init)
    bias = tf.get_variable( 'bias', shape = [n_class,],
                                initializer = bias_init ) 
    z = tf.matmul(p_vec, weight) + bias
    y_pred = tf.nn.softmax(z, axis=1)
    y_pred_cls = tf.argmax(y_pred, axis=1, name='predict_lbl')#after argmax, is a list : be careful, y_target is from 0:9

    # calc cross entropy loss using fancy tf function
    ce_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                logits=z, labels=y_target), name='celoss')

    #optimizer
    optim_step = tf.train.AdamOptimizer(lr).minimize(ce_loss)

    # add to the collection 
    tf.add_to_collection( 'validation_nodes', x_input)
    tf.add_to_collection( 'validation_nodes', y_pred_cls)

    # prepare the saver
    saver = tf.train.Saver()

    ## 3. START TRAINING

    # init operation
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:

        # init parameters
        sess.run( init_op )

        # training cycle
        for epoch in range( max_epoch ):

            # shuffle idx
            rng.shuffle( train_indices )

            # record the average loss 
            train_avg_loss = 0.
            train_avg_acc  = 0.

            for idx in range( batch_num ):

                # divide the shuffled dataset into mini-batches. 
                # drop the samples that not in the range of batch_size * batch_num
                idx_start = idx * batch_size 
                idx_end = (idx + 1) * batch_size

                # chosse the suffled indices
                shuffle_indices = train_indices[ idx_start:idx_end ]

                # obtain the training data: X_batch, and training label: y_batch,
                x_batch, y_batch = x_train[shuffle_indices, :, :, :], y_train[shuffle_indices, ].tolist()

                # train the weight 
                _, loss_train, y_train_pred = sess.run( [optim_step, ce_loss, y_pred_cls], 
                                                                    feed_dict = { x_input: x_batch, 
                                                                                y_target: y_batch })

                # calculate the test accuracy
                acc_train = sum( y_train_pred==y_batch) / len(y_train_pred)

                # computer average_loss, note: loss_train has been divded by batch_size
                train_avg_loss += loss_train / batch_num
                train_avg_acc  += acc_train / batch_num

            # run the test data to show the performance of the model        
            y_test_pred = sess.run( y_pred_cls, feed_dict = { x_input: x_test })

            # calculate the test accuracy
            acc_test = sum( y_test_pred==y_test) / len(y_test)

            # record the train loss
            train_loss_history.append( train_avg_loss )
            train_acc_history.append( train_avg_acc )
            test_acc_history.append( acc_test )

            # record the prediction for confusion matrix 
            test_pred_log[epoch] = y_test_pred

            if epoch % display_steps == 0:
                print( 'epoch: {}, train_acc: {}%, test_acc: {} %'\
                    .format(epoch, np.round(train_avg_acc, 4)*100, np.round(acc_test, 4)*100 ))

        # save session 
        save_path = saver.save(sess, 'my_model')

    ## 4. PLOT RESULTS

    # plot the confusion matrix for the optimal model
    y_pred_op = test_pred_log[max_epoch-1]
    y_test = np.array(y_test)
    y_pred_op = np.array(y_pred_op)
    cm = confusion_matrix( y_test, y_pred_op )
    fig1 = plot_confusion_matrix( cm )
    plt.savefig( 'figure/confusion_matrix-lr={}.png'.format( lr ) )

    # plot the overall training and testing error
    fig2 = plot_twin_result( train_loss_history, 
                                train_acc_history, test_acc_history)
    plt.savefig( 'figure/overall_acc-lr={}.png'.format( lr ) )

    # obtain the 1st convolution filter    
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('my_model.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('./'))
        graph = tf.get_default_graph()
        wx1 =graph.get_tensor_by_name( 'weight_x1:0')
        weight1 = sess.run(wx1)

    # print 1st convolution filter
    fig3 = plot_filter( weight1 )
    plt.savefig( 'figure/filters-lr={}.png'.format( lr ) )





    