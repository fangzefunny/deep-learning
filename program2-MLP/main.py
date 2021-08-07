import os 
import argparse
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle

import tensorflow as tf

## utils
# one hot encoding
def onehotencode( y_train, num_classes ):
    '''
    Input: 1.data need to be encoded; 2. number of the classes
    output: one hot code of the data 
    '''
    num_samples = len( y_train )
    y_train_onehot = np.zeros( [num_samples, num_classes] )
    for i in range( len(y_train) ):
        y_train_onehot[i, int(y_train[i]-1)] = 1. 
    return  y_train_onehot

# load the images
def img_loader( dir_path, start_idx, end_idx, img_size, pad_size=5 ):
    num_samples = end_idx - start_idx 
    flatten_img = img_size[0] * img_size[1]
    X =np.empty( [num_samples, flatten_img] )
    for i in range(num_samples):
        imgname = dir_path + str(i+1).zfill(pad_size) + '.jpg' 
        img = mpimg.imread( imgname )
        img = img.reshape(-1) / 255
        X[i, :] = img
    return X 

# load the labels
def label_loader( dir_path, start_idx, end_idx, num_classes ):
    num_samples = end_idx - start_idx 
    y = np.empty( [num_samples,] )
    rb = open( dir_path )
    lines = rb.readlines()
    for i in range( num_samples ):
        y[i] = lines[i]
    y = onehotencode( y, num_classes ) # one hot encode of  the label
    return y 

## activation function

# softmax function
def softmax( logits ):
    r'''
    Input: the output of regression
    output: softmax probability distribution
    '''
    y_pred = tf.exp(logits) / tf.reduce_sum( tf.exp(logits), axis = 1, keepdims=True )
    y_pred = tf.clip_by_value( y_pred, 1e-10, 1. )
    return y_pred

# relu function
def ReLU( x ):
    r'''
    z = max( 0, x )
    '''
    return tf.maximum( 0., x )

## loss function 

# cross entropy
def cross_entropy( y_pred, y_target ):
    '''
    loss = - 1/m * sum_k(  y_target[m][k] * log y_pred[m][k] )
    '''
    loss = - tf.reduce_mean( tf.reduce_sum( y_target * tf.log( y_pred ), axis = 1) )
    return loss 

# l2_norm
def l2_norm( weight ):
    r'''
    loss = .5 * w^2
    '''
    loss = .5 * tf.reduce_sum( weight ** 2 )
    return  loss

## plot tools

# plot weights
def plot_weights( W, num_classes ):
    fig = plt.figure( figsize = [12, 2] )
    for class_idx in range( num_classes ):
        W_k = W[0:784, class_idx] + W[ 784, class_idx]
        img = W_k.reshape(28,28)
        plt.subplot( 1,5, class_idx+1)
        plt.imshow(img)
        plt.colorbar()
        plt.title( 'weight{}'.format( class_idx ), fontsize = 15)
        plt.axis('off')
    return fig
    
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

# plot tool for result
def plot_digit_error( digit_table  ):
    plt.style.use('ggplot')
    x = np.arange( 1, 6 )
    fig = plt.figure()
    plt.plot( x, digit_table ) 
    plt.xlabel( 'digits' )
    plt.ylabel( 'accuracy')
    plt.ylim( [.5, 1.])
    plt.title( 'prediction error for each digits')
    return fig 

# compute confusion matrix         
def confusion_matrix( y_true, y_pred_cls ):
    cls_num = np.max( y_true ) + 1
    cm = np.empty( [cls_num, cls_num] )
    for i in range( cls_num ):
        for j in range( cls_num ):
            yi_num = y_true[y_true==i].shape[0]
            yi_pred_num = y_true[ (y_pred_cls == j) * (y_true==i) ].shape[0]
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
 
if __name__ == '__main__':

    ## 0. HYPERPARAMETER TUNING
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', help='learning rate', type=float, default=0.05)

    args = parser.parse_args()

    ## 1. LOAD THE DATA 

    # get the current dir path
    dir_path = os.getcwd()

    # load the training data into a np array: X_train 
    train_img_path = dir_path + '/data_prog2/train_data/'    
    X_train = img_loader( train_img_path, 1, 50000, [ 28, 28], 5 )

    # load the training label into a np array: y_train 
    train_label_path = dir_path + '/data_prog2/labels/train_label.txt'
    y_train = label_loader( train_label_path, 1, 50000, 10 )

    # load the test data into a np array: X_test
    test_img_path = dir_path + '/data_prog2/test_data/'    
    X_test = img_loader( test_img_path, 1, 5000, [ 28, 28], 5 )

    # load the test label into a np array: y_test 
    test_label_path = dir_path + '/data_prog2/labels/test_label.txt'
    y_test = label_loader( test_label_path, 1, 5000, 10 )

    ## 2. PREPARE THE  COMPUTATIONAL GRAPH

    # hyper paramters
    lr            = args.lr     # learning rate 
    batch_size    = 50          # number of samples within a mini-batch

    # other useful values
    max_epoch     = 100                # number of iterations 
    diplay_steps  = 1                  # display the loss every n steps
    total_samples = y_train.shape[0]   # total number of training samples
    test_num      = y_test.shape[0]    # total number of testing samples
    input_shape   = X_train.shape[1]   # dimension of input stim
    output_shape  = y_train.shape[1]   # dimenions of output classes
    hidden_shape  = [100, 100]
    batch_num     = total_samples // batch_size + 1 # num of mini batches
    train_indices = np.arange( 0, total_samples )   # train_data indices
    rng = np.random.RandomState( 1234 ) # to help generate random things

    # storages
    train_loss_history = np.zeros( [ max_epoch, ] ) # record the train loss
    test_loss_history  = np.zeros( [ max_epoch, ] ) # record the test loss
    train_acc_history  = np.zeros( [ max_epoch, ] ) # record the train acc
    test_acc_history   = np.zeros( [ max_epoch, ] ) # record the test acc 
    param_dict = dict() # save the weights, so we can choose the optimal one later
    test_pred_log = dict() # save the y_pred for confusion matrix 

    # layer initialization weight very small random value, bias, 0
    norm_init = tf.random_normal_initializer( mean=0.0, stddev=.05 )
    zero_init = tf.constant_initializer( 0.)

    # placeholders
    input_x  = tf.placeholder( tf.float32, shape = [None, input_shape], name = 'x')
    y_target = tf.placeholder( tf.float32, shape = [None, output_shape], name= 'y') 
    sample_num = tf.placeholder(dtype=tf.int32, name ='sample_num') 

    # forward 
    # input --> hidden1 
    with tf.variable_scope( 'W1', reuse=tf.AUTO_REUSE ):
        w1  = tf.get_variable( 'weight1', [input_shape, hidden_shape[0]], 
                         initializer = norm_init )
        w10 = tf.get_variable( 'bias1', [hidden_shape[0],], 
                         initializer = zero_init )
    z1 = tf.matmul( input_x, w1 ) + w10
    h1 = ReLU( z1 )

    # hidden1 --> hidden2
    with tf.variable_scope( 'W2', reuse=tf.AUTO_REUSE ):
        w2  = tf.get_variable( 'weight2', [hidden_shape[0], hidden_shape[1]],
                          initializer = norm_init )
        w20 = tf.get_variable( 'bias2', [hidden_shape[1],], 
                         initializer = zero_init )
    z2 = tf.matmul( h1, w2 ) + w20 
    h2 = ReLU( z2 )

    # hidden2 --> output
    with tf.variable_scope( 'W3', reuse=tf.AUTO_REUSE ):
        w3  = tf.get_variable( 'weight3', [hidden_shape[1], output_shape ],
                          initializer = norm_init )
        w30 = tf.get_variable( 'bias3', [output_shape,], 
                         initializer = zero_init )
    z3 = tf.matmul( h2, w3 ) + w30
    y_pred = softmax( z3 ) 

    # calculate loss 
    loss = cross_entropy( y_pred, y_target ) 

    # calculate acc
    y_pred_cls = tf.argmax(y_pred, 1)
    correct_pred = tf.equal(y_pred_cls, tf.argmax(y_target, 1))
    accuracy = tf.reduce_mean( tf.cast(correct_pred, tf.float32) )

    # backward
    # from output --> hidden2 
    dL_dz = (y_pred - y_target) / tf.cast( sample_num, tf.float32 )
    gradw30 = tf.reshape(tf.matmul( tf.transpose(tf.ones([sample_num , 1])),  dL_dz), [-1]) # grad bias3
    gradw3 = tf.matmul( tf.transpose(h2), dL_dz ) # grad weight3
    gradh2 = tf.matmul( dL_dz, tf.transpose(w3) ) # grad h2

    # from hidden2 --> hidden1
    dh2_dz = tf.cast((z2 > 0), tf.float32) * gradh2 
    gradw20 = tf.reshape(tf.matmul( tf.transpose(tf.ones([sample_num , 1])), dh2_dz ), [-1]) # grad bias2
    gradw2 = tf.matmul( tf.transpose(h1), dh2_dz ) # grad weight2
    gradh1 = tf.matmul( dh2_dz, tf.transpose(w2) ) # grad h1

    # from hidden1 --> input x
    dh1_dz = tf.cast((z1 > 0), tf.float32) * gradh1
    gradw10 =  tf.reshape(tf.matmul( tf.transpose(tf.ones([sample_num , 1])), dh1_dz ), [-1]) # grad bias1
    gradw1 = tf.matmul( tf.transpose(input_x),  dh1_dz ) # grad weight1

    # update weights
    neww1  = w1.assign( w1 - lr * gradw1 )
    neww10 = w10.assign( w10 - lr * gradw10 )
    neww2  = w2.assign( w2 - lr * gradw2 )
    neww20 = w20.assign( w20 - lr * gradw20 )
    neww3  = w3.assign( w3 - lr * gradw3 )
    neww30 = w30.assign( w30 - lr * gradw30 )

    # init operation
    init_op = tf.global_variables_initializer()

   ## 3. START TRAINING 

    # construct graph session
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

                # divde the shuffled dataset into mini-batches each with 100 samples,
                idx_start = idx * batch_size 
                idx_end = np.min( [(idx + 1) * batch_size, total_samples] )

                # chosse the suffled indices
                shuffle_indices = train_indices[ idx_start:idx_end ]
                actual_batch = int( idx_end - idx_start )

                # obtain the training data: X_batch, and training label: y_batch,
                X_batch, y_batch = X_train[shuffle_indices, : ], y_train[shuffle_indices, : ]

                # train the weight 
                w1, w10,\
                w2, w20,\
                w3, w30,\
                loss_train, acc_train = sess.run( [ neww1, neww10,
                                                    neww2, neww20,
                                                    neww3, neww30, loss, accuracy], 
                                                    feed_dict = { input_x: X_batch, 
                                                                 y_target: y_batch, 
                                                               sample_num: actual_batch }
                                                    )

               # computer average_loss, note: loss_train has been divded by batch_size
                train_avg_loss += loss_train / batch_num
                train_avg_acc  += acc_train / batch_num

                # record the train loss
                train_loss_history[epoch] =  train_avg_loss 
                train_acc_history[epoch] = train_avg_acc
        
            # record the test acc 
            loss_test, acc_test, y_test_pred = sess.run(  [loss, accuracy, y_pred_cls], 
                                        feed_dict = { input_x: X_test, 
                                                                y_target: y_test }
                                        )
            test_loss_history[epoch] = loss_test
            test_acc_history[epoch] = acc_test

            # record the prediction for confusion matrix  
            test_pred_log[epoch] = y_test_pred

            #print( 'epoch: {}, train_loss: {}%, test_loss: {}%'\
                #.format(epoch, round(train_avg_acc, 2)*100, round(acc_test, 2)*100 ))

            # save trained weight each epoch
            # after looking at the plot, we can choose one as the optimal nn.
            Theta = [ w1, w10, w2, w20, w3, w30 ]
            param_dict[epoch] = Theta

     ### 4. PLOTS THE RESULTS

    # choose optimal weight and save 
    w_vector = train_loss_history + test_loss_history
    op_idx = np.argmin( w_vector )
    Theta = param_dict[ op_idx ]
    fname = "nn_parameters-lr{}.txt".format( args.lr )
    with open(fname,"wb") as handle:
            pickle.dump(Theta, handle)

    # plot the confusion matrix for the optimal model
    y_pred_op = test_pred_log[ op_idx ]
    y_true = np.argmax( y_test, 1 )
    cm = confusion_matrix( y_true, y_pred_op )
    fig1 = plot_confusion_matrix( cm )
    plt.savefig( 'confusion_matrix-lr={}.png'.format( args.lr ) )

    # plot overall prediction error  
    fig2 = plot_acc_result( 1-train_acc_history, 1-test_acc_history )
    plt.savefig( 'compare_acc-lr={}.png'.format( args.lr ) )
    
    # plot overall loss 
    fig3 = plot_loss_result( train_loss_history, test_loss_history )
    plt.savefig( 'compare_loss-lr={}.png'.format( args.lr ) )