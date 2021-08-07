import os 
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
    num_samples = end_idx - start_idx + 1 
    flatten_img = img_size[0] * img_size[1]
    X =np.empty( [num_samples, flatten_img] )
    for i in range( num_samples):
        imgname = dir_path + str(i+1).zfill(pad_size) + '.jpg' 
        img = mpimg.imread( imgname )
        img = img.reshape(-1) / 255
        X[i, :] = img
    return X 

# load the labels
def label_loader( dir_path, start_idx, end_idx, num_classes ):
    num_samples = end_idx - start_idx + 1
    y = np.empty( [num_samples,] )
    rb = open( dir_path )
    lines = rb.readlines()
    for i in range( num_samples ):
        y[i] = lines[i]
    y = onehotencode( y, num_classes ) # one hot encode of  the label
    return y 

# softmax function
def softmax( logits ):
    '''
    Input: the output of regression
    output: softmax probability distribution
    '''
    sigma_m = tf.exp(logits) / tf.reduce_sum( tf.exp(logits), axis = 1, keepdims=True )
    y_pred = tf.clip_by_value( sigma_m, 1e-10, 1.- 1e-10 ) # prevent infinite value 
    return y_pred

# loss function 
def cross_entropy( y_pred, y_target ):
    '''
    loss = - 1/m * sum_k(  y_target[m][k] * log y_pred[m][k] )
    '''
    loss = - tf.reduce_mean( tf.reduce_sum( y_target * tf.log( y_pred ), axis = 1) )
    return loss 

def l2_norm( weight ):
    '''
    loss = .5 * w^2
    '''
    loss = .5 * tf.reduce_sum( weight ** 2 )
    return  loss
    
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
    
# plot tool for result
def plot_result( train_acc_history, test_acc_history  ):
    plt.style.use('ggplot')  
    fig = plt.figure()
    plt.plot( train_acc_history) 
    plt.plot( test_acc_history) 
    plt.xlabel( 'iterations' )
    plt.ylabel( 'accuracy')
    my_x_ticks = np.arange(0, 21, 1)
    plt.xticks(my_x_ticks)
    plt.legend( [' train', 'test'], loc=4 )
    plt.title( 'accuracy train vs. test')
    return fig 

# plot tool for result
def plot_digit_error( digit_table  ):
    plt.style.use('ggplot')
    x = np.arange( 1, 6 )
    fig = plt.figure()
    plt.plot( x, digit_table ) 
    plt.xlabel( 'digits' )
    plt.ylabel( 'accuracy')
    my_x_ticks = np.arange(1, 6, 1)
    plt.xticks(my_x_ticks)
    plt.ylim( [.5, 1.])
    plt.title( 'prediction error for each digits')
    return fig 
 
if __name__ == '__main__':
    
    ## 1. LOAD THE DATA 
    
    # get the current dir path
    dir_path = os.getcwd()

    # load the training data into a np array: X_train 
    train_img_path = dir_path + '/data_prog2/train_data/'    
    X_train = img_loader( train_img_path, 1, 25112, [ 28, 28] )

    # load the training label into a np array: y_train 
    train_label_path = dir_path + '/data_prog2/labels/train_label.txt'
    y_train = label_loader( train_label_path, 1, 25112, 5 )

    # load the test data into a np array: X_test
    test_img_path = dir_path + '/data_prog2/test_data/'    
    X_test = img_loader( test_img_path, 1, 4982, [ 28, 28],4 )

    # load the test label into a np array: y_test 
    test_label_path = dir_path + '/data_prog2/labels/test_label.txt'
    y_test = label_loader( test_label_path, 1, 4982, 5 )

    ## 2. PREPARE THE  COMPUTATIONAL GRAPH

    # hyper paramters
    lr            = 0.01   # learning rate 
    lam           = 0.01     # parameters for regularizer
    batch_size    = 50        # number of samples within a mini-batch

    # other useful values
    max_episodes  = 20        # number of iterations 
    diplay_steps  = 1          # display the loss every n steps
    total_samples = y_train.shape[0]   # total number of training samples
    test_num      = y_test.shape[0]    # total number of testing samples
    input_size    = X_train.shape[1]  # dimension of input stim
    output_size   = y_train.shape[1]  # dimenions of output classes
    batch_num     = total_samples // batch_size + 1 # num of mini batches
    train_indices = np.arange( 0, total_samples ) # train_data indices
    rng = np.random.RandomState( 1234 ) # to help generate random things

    # storages
    train_history     = np.zeros( [ max_episodes, ] ) # record the loss
    test_acc_history  =  np.zeros( [ max_episodes, ] ) # record the loss
    train_acc_history = np.zeros( [ max_episodes, ] ) # record the loss

    # placeholders
    input_x = tf.placeholder( tf.float32, [None, input_size], name = 'x' ) # input 
    y_target = tf.placeholder( tf.float32, [None, output_size], name = 'y' )   # supervied label 
    sample_num = tf.placeholder(dtype=tf.int32, name ='sample_num') # samples per batch

    # set up the weights
    W = tf.Variable( tf.zeros([ input_size, output_size] ), name='W' )
    w0 = tf.Variable( tf.zeros([output_size,] ), name='w0' )

    # regression 
    logits = tf.matmul( input_x, W ) +  w0 

    # predicted probability distribution
    y_pred = softmax( logits  ) 

    # calculate loss
    loss = cross_entropy( y_pred, y_target ) + lam * l2_norm( W ) + lam * l2_norm( w0 )

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_target, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # calculate grad 
    # gradW = d(y_pred)/ d(W) * d( Loss) / d( y_pred) + d(Loss)/ d(W)
    grad_ypred = (y_pred - y_target ) / tf.cast( sample_num , tf.float32 )
    gradW = tf.matmul( tf.transpose(input_x), grad_ypred ) + lam * W 
    gradw0 = tf.reshape(tf.matmul( tf.transpose(tf.ones([sample_num , 1])), grad_ypred), [-1]) \
                     + lam * w0

    # update weights
    newW = W.assign( W - lr * gradW )
    neww0 = w0.assign( w0 - lr * gradw0 )

    # init operation
    init_op = tf.global_variables_initializer()

    ## 3. START TRAINING 

    # construct graph session
    with tf.Session() as sess:

        # init parameters
        sess.run( init_op )

        # training cycle
        for episode in range( max_episodes ):

            # shuffle idx
            rng.shuffle( train_indices )

            # record the average loss 
            train_avg_loss = 0.

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
                _, _, loss_train = sess.run( [newW, neww0 , loss], 
                                                     feed_dict = { input_x: X_batch, 
                                                                          y_target: y_batch, 
                                                                          sample_num: actual_batch }
                                                    )

                # computer average_loss, note: loss_train has been divded by batch_size
                train_avg_loss += loss_train / batch_num

                # record the train loss
                train_history[episode] =  train_avg_loss 

            train_test = sess.run(  accuracy, 
                                           feed_dict = { input_x: X_train, 
                                                                y_target: y_train }
                                          )    
            acc_test = sess.run(  accuracy, 
                                           feed_dict = { input_x: X_test, 
                                                                y_target: y_test }
                                          )

            # record the acc 
            train_acc_history[ episode ]  = train_test
            test_acc_history[ episode ] = acc_test 

        # calculate error for each digits
        digit_table = np.empty( [5,] )
        for class_idx in range( output_size ):
            pick_indices = y_train[:, class_idx ] == 1
            X_digit, y_digit = X_train[ pick_indices, : ], y_train[ pick_indices, : ]
            acc_digit = sess.run( accuracy, 
                                        feed_dict = { input_x: X_digit, 
                                                            y_target: y_digit })

            digit_table[ class_idx ] = acc_digit 

        # save trained weight
        weight = sess.run( W )
        bias = sess.run(w0)  
        # concate 784x5 and 1x5 into a 785 x 5 matrix  on dim=0 
        W = np.concatenate( (weight, bias.reshape(-1,5)), axis=0 )
        with open("multiclass_parameters.txt","wb") as handle:
            pickle.dump(W, handle)


    ### 4. PLOTS THE RESULTS

    # plot weight 
    fig1 = plot_weights( W, 5 )
    #plt.savefig( 'weight1.png' )

    # plot digit error
    fig2 = plot_digit_error( digit_table )
    #plt.savefig( 'digit_error1.png' )

    # plot overal result 
    fig3 = plot_result( train_acc_history, test_acc_history )
    #plt.savefig( 'test_train1.png' )



