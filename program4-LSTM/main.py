import os 
import argparse
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pickle

import tensorflow as tf

## utils: normalize and split the data 
def split_data( data, label ):
    
    #  check the input sample size
    tot_sample = data.shape[0]
    
    # mannual train sample as 6000, valid sample as 1000
    train_num = 6000
    
    # create the total data indeices
    tot_idx = np.arange(0, tot_sample)
    
    # shuffle the indince array
    np.random.RandomState(2020).shuffle(tot_idx)
    
    # train_idx, test_idx
    train_idx, valid_idx = tot_idx[ 0 :train_num], tot_idx[ train_num :tot_sample]
    
    # split the train and test 
    train_data, valid_data = data[ train_idx, :, :, :, : ], data[ valid_idx, :, :, :, : ]
    train_label, valid_label = label[ train_idx, :, :, : ], label[ valid_idx, :, :, : ]
    
    return train_data, train_label, valid_data, valid_label 

def preprocess( data ):
    
    # convert the uni8 datatype to float32
    data = tf.dtypes.cast(data, tf.float32) 
    
    # normalize the data
    data -= tf.reduce_mean(data, axis=(2, 3, 4), keepdims=True)
    data /= tf.math.reduce_std( data, axis=(2, 3, 4), keepdims=True)

    return data 

def conv2d( x, W):
    # stride 1: [1, x_move, y_move, 1] ==> x_move, y_move = 1, 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], 
                              padding='VALID' )

# specific max pooling: 
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], 
                           strides=[1, 2, 2, 1], padding='VALID')

# create a CNN module
def cnn_module( img_input, n_features):
    
    # the method to init the weight
    weight_init = tf.contrib.layers.xavier_initializer()
    bias_init = tf.constant_initializer( 0.)
    
    # img_input: 64 x 64 x 3 
    
    # weight: conv1
    weight_x1 = tf.get_variable( 'weight_x1', 
                                shape = [5, 5, 3, 32 ],
                                initializer = weight_init)
    bias_x1 = tf.get_variable( 'bias_x1', shape = [32,],
                                initializer = bias_init ) 
    # h_conv1 = relu(conv2d( X,W) +b)
    c1 = tf.nn.relu(conv2d( img_input, weight_x1) + bias_x1) #c1: 60 x 60 x 32
    p1 = max_pool( c1 ) #p1: 30 x 30 x 32

    # weight: conv2
    weight_x2 = tf.get_variable( 'weight_x2', 
                                shape = [5, 5, 32, 32 ],
                                initializer = weight_init)
    bias_x2 = tf.get_variable( 'bias_x2', shape = [32,],
                                initializer = bias_init ) 
    # h_conv1 = conv2d( X,W) +b
    c2 = tf.nn.relu(conv2d( p1, weight_x2) + bias_x2) #c2 no padding: 26x 26 x 32
    p2 = max_pool( c2 ) #p1: 13 x 13 x 32

    # weight: conv3
    weight_x3 = tf.get_variable( 'weight_x3', 
                                shape = [3, 3, 32, 64 ],
                                initializer = weight_init)
    bias_x3 = tf.get_variable( 'bias_x3', shape = [64,],
                                initializer = bias_init ) 
    # h_conv1 = conv2d( X,W) +b
    c3 = tf.nn.relu(conv2d( p2, weight_x3) + bias_x3) #c3 no padding: 11 x 11 x 64

    # flatten 
    p_vec = tf.reshape( c3, [-1, 11 * 11 * 64]) # batch_size x (11 * 11 * 64)
    
    #  fully connect 
    weight_phi = tf.get_variable( 'weight_phi', 
                                shape = [11 * 11 * 64, n_features],
                                initializer = weight_init)
    bias_phi = tf.get_variable( 'bias_phi', shape = [n_features,],
                                initializer = bias_init ) 
    features  = tf.matmul(p_vec, weight_phi) + bias_phi
    
    return features  


def rnn_module( rnn_input,  num_units ):
    
    # instantiate a LSTM cell
    lstm_cell = tf.nn.rnn_cell.LSTMCell( num_units )
    #  define your RNN network, the length of teh sequence will be automatically retrived
    output, _ = tf.nn.dynamic_rnn( lstm_cell, rnn_input, dtype = tf.float32 ) 
    # output = h_1,...,h_T, state = (cT, hT)
    
    return output 

def linear_module( hid_states, out_num ):
    
    # the method to init the weight
    weight_init = tf.contrib.layers.xavier_initializer()
    bias_init = tf.constant_initializer( 0.)
    
    # linear module 
    rnn_hidnum = hid_states.shape[1]
    weight_y = tf.get_variable( 'weight_y', 
                            shape = [rnn_hidnum, out_num],
                            initializer = weight_init)
    bias_y = tf.get_variable( 'bias_y', shape = [out_num,],
                                initializer = bias_init ) 
    y_pred = tf.matmul( hid_states, weight_y) + bias_y # [ batch*seq, 14 ]
    return y_pred 
    

def seq_l2_loss( y_pred, y_target ):
    r'''
    we cannot use mse to directly compute the loss. The only l2 distance is the last dim, x and y:
    loss = mean_b [mean_seq [ mean_node [ sqrt[ (x - x')^2 + (y - y')^2] ]]]
    we can treat [batch * seq * node] as the sample,  
    '''
    y_pred = tf.reshape( y_pred, [-1, 2] )
    y_target = tf.reshape( y_target, [-1, 2])
    
    l2_distance = tf.sqrt( tf.reduce_sum((y_pred - y_target)**2, axis=1) )
    loss = tf.reduce_mean( l2_distance )
    
    return loss 

def calc_acc( y_pred, y_target ):
    
    # [bartch, seq, joint, xy] --> [batch*seq, joint, xy]
    y_pred = np.reshape( y_pred, [-1, 7, 2] )
    y_target = np.reshape( y_target, [-1, 7, 2])
    
    # n batch
    tot_batch = y_pred.shape[0]
    
    # [batch*seq, joint, l2_dist ]
    l2_distance = np.sqrt( np.sum( np.square(y_pred - y_target), axis=2) )
    
    # for each batch*seq, check if l2_dist < n pixels
    acc_matrix = np.empty( [20, 7] )
    for n in range( 20):
        acc_matrix[ n, :] = np.sum( l2_distance<n+1, axis=0 )
    
    acc = acc_matrix / tot_batch
    
    return acc 

# plot loss
def plot_loss_results( train_loss_history, test_loss_history  ):
    plt.style.use('ggplot')  
    fig = plt.figure()
    plt.plot( train_loss_history) 
    plt.plot( test_loss_history) 
    plt.xlabel( 'epoch' )
    plt.ylabel( 'loss')
    plt.legend( [' train', 'valid'] )
    plt.title( 'loss train vs. valid')
    return fig 

# plot acc
def plot_acc_vs_pixels( yall_valid_pred, y_valid ):
    acc_valid = calc_acc( yall_valid_pred, y_valid)
    plt.style.use('ggplot')  
    fig = plt.figure()
    for i in range(7):
        plt.plot( acc_valid[:, i]) 
    plt.xlabel( 'Distance(pixels)' )
    plt.ylabel( 'Accuracy(%)')
    plt.legend( ['head', 'r wrist', 'l wrist', 'r elbow', 'l elbow', 'r shoulder', 'l shoulder'], loc =0)
    plt.title( 'Final prediction accuracy')
    return fig 

def plot_examples( data ):
    fig = plt.figure()
    # comput the prediction using the trained model 
    with tf.Session() as sess:
            new_saver = tf.train.import_meta_graph('my_model.meta')
            new_saver.restore(sess, tf.train.latest_checkpoint('./'))

            # Accessing certain nodes of the Tensorflow graph which are needed for testing the model
            graph = tf.get_default_graph()
            input_frames = graph.get_tensor_by_name('input_frames:0') # Input Video Frames
            joint_pos = graph.get_tensor_by_name('joint_pos:0') # Predicted Joint Positions

            predictions = sess.run(joint_pos, feed_dict = {input_frames: data}) # Make predictions using the LSTM+CNN model
    
    # print image and prediction 
    plt.style.use('classic')  
    for i in range(6):
        plt.subplot(2,3, i+1)
        plt.imshow( data[i, -1, :, :, :])
        plt.axis('off')

        for j in range(7):
            x = predictions[i, -1, j, 0]
            y = predictions[i, -1, j, 1]
            plt.scatter( x, y,  color = 'g',s=60)
    
    return fig

def plot_distance( y_pred, y_target ):
    
    num_classes = 7
    
    # reshape to help calculation
    y_pred = np.reshape( y_pred, [-1, num_classes, 2] )
    y_target = np.reshape( y_target, [-1, num_classes, 2])
    
    # construct a long matrix
    cm = np.empty( [num_classes, num_classes])
    
    # for each joint
    for i in range( num_classes):
        for j in range( num_classes):
            
            # select pred and target joints
            joint_target= y_target[:, i, :] 
            joint_pred  = y_pred[ :, j, :]

            # compute the loss and reshape them to a [7,] list
            l2_distance = np.sqrt( np.sum((joint_pred - joint_target)**2, axis=1) ) # [batch* seq,]
            distance = np.mean( l2_distance, axis=0 ) #scalar 

            # assign the distance to cm 
            cm[i, j] = distance 

    fig = plt.figure()
    plt.style.use('classic')  
    plt.imshow( cm, cmap = 'viridis', interpolation='nearest')
    plt.colorbar()
    plt.xlabel( 'Distance(pixels)' )
    plt.ylabel( 'Accuracy(%)')
    plt.title( 'Distance for each joints')
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for i in range( num_classes ):
        for j in range( num_classes):
            plt.text(i-.3,j+.1, np.round(cm[j, i],0))
    return fig

if __name__ == '__main__':
    
     ## 0. HYPERPARAMETER TUNING
        
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--lr', help='learning rate', type=float, default=0.0001)

    # args = parser.parse_args()
    
    # load the data 
    with open('youtube_train_data.pkl', 'rb') as handle:
        train_data, train_labels = pickle.load(handle)

    # normalize and split_data 
    x_train, y_train, x_valid, y_valid  = split_data( train_data, train_labels)


    ## 2. PREPARE THE COMPUTATIONAL GRAPH

    # hyper paramters
    lr          = 0.002  # learning rate
    batch_size  = 200    # number fo the sample with a batch  

    # fix some useful values
    save_steps    = 1                       # save the sess every n epoch 
    display_steps = 5                       # print epoch loss 
    max_epoch     = 50                      # max epoch 
    nfeatures     = 128                     # num of cnn output features 
    rnn_hidnum    = 32                      # num of rnn output nodes
    total_samples = x_train.shape[0]        # total sample number
    seq_len       = x_train.shape[1]        # length of the time sequence 
    img_width     = x_train.shape[2]        # width of the image
    img_height    = x_train.shape[3]        # height of the image
    img_channel   = x_train.shape[4]        # num of channel of the image
    total_valid   = x_valid.shape[0]        # batch num for the validation data.
    batch_num     = total_samples // batch_size # num of mini batches
    valid_batch   = total_valid//batch_size     # num of mini batches in test
    train_indices = np.arange(0, total_samples) # indices to help shuffle
    valid_indices = np.arange(0, total_valid)   # indices of the validation data
    rng = np.random.RandomState( 2020 ) # to help generate random things

    # storages
    train_loss_history  = []  # record the train loss
    valid_loss_history  = []  # record the test loss
    train_acc_history   = []  # record the train acc
    valid_acc_history   = []  # record the test acc 
    valid_pred_log      = dict()  # save the y_pred for plot 

    # initiate the computational graph 
    tf.reset_default_graph()

    # define place holder
    frames_input = tf.placeholder( tf.uint8, [ None,  seq_len, img_width, img_height, img_channel ],
                                                                name = 'input_frames' )
    y_target = tf.placeholder( tf.float32, [ None, seq_len, 7, 2] )

    # preprocess: float32 and normalize
    frames_input_norm = preprocess( frames_input)

    # frame [batch, seq, x, y, chan]  --> img1:T [ batch*seq, x ,y, chan] 
    img_input_vec = tf.reshape(frames_input_norm, [-1, img_width, img_height, img_channel ])

    # forward
    # img1:T  --> convolution --> xt [ batch*seq, nfeatures]
    features_vec = cnn_module( img_input_vec, nfeatures )

    # reshape the features back to [ batch, seq, nfeatures] 
    features = tf.reshape(features_vec, [-1, seq_len, nfeatures])

    #  xt-> LSTM --> ht  [batch, seq, rnn_hidnum]
    hid_states = rnn_module( features, rnn_hidnum)

    # reshape the ht to h1:T [batch*seq, rnn_hidnum]
    hid_states_vec = tf.reshape(hid_states, [-1, rnn_hidnum] )

    # ht --> y_t:  h_t * weight_y + bias_y
    y_pred_vec = linear_module( hid_states_vec, 7*2 )

    # reshape back to [batch, seq, 7, 2]
    y_pred = tf.reshape( y_pred_vec, [-1, seq_len, 7, 2])
    joint_pos = tf.identity( y_pred, name = 'joint_pos' )

    # calculate the loss 
    loss = seq_l2_loss( y_pred, y_target )

    # set optimizer
    optim_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    # save the model 
    tf.get_collection( 'validation_nodes' )

    # add the opt node to the collection
    tf.add_to_collection( 'validation_nodes', frames_input )
    tf.add_to_collection( 'validation_nodes', joint_pos )

    # prepare saver 
    saver = tf.train.Saver() 

    ## 3. START TRAINING

    # init variable value 
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
            valid_avg_loss = 0. 
            valid_avg_acc  = 0.
            yall_valid_pred = np.empty([total_valid, seq_len, 7, 2])

            for idx in range( batch_num ):

                # divide the shuffled dataset into mini-batches. 
                # drop the samples that not in the range of batch_size * batch_num
                idx_start = idx * batch_size 
                idx_end = (idx + 1) * batch_size

                # chosse the suffled indices
                shuffle_indices = train_indices[ idx_start:idx_end ]

                # obtain the training data: X_batch, and training label: y_batch,
                x_batch, y_batch = x_train[shuffle_indices, :, :, :], y_train[shuffle_indices, :, :]

                # train the weight 
                _, loss_train, y_train_pred = sess.run( [optim_step, loss, joint_pos], 
                                                            feed_dict = { frames_input: x_batch, 
                                                                                    y_target: y_batch })

                # calculate the train accuracy each batch
                acc_train = calc_acc(y_train_pred, y_batch)

                # computer average_loss, note: loss_train has been divded by batch_size
                train_avg_loss += loss_train / batch_num
                train_avg_acc  += acc_train / batch_num

            # run the valid data to show the performance of the model 
            for vdx in range( valid_batch ):

                # validation idx
                vdx_start = vdx * batch_size 
                vdx_end = (vdx+ 1) * batch_size

                # chosse the indices without shuffling
                sample_indices = valid_indices[ vdx_start:vdx_end ]

                # obtain the training data: x, and training label: y_batch,
                x_valid_batch, y_valid_batch = x_valid[sample_indices, :, :, :], \
                                            y_valid[sample_indices, :, :]

                loss_valid, y_valid_pred = sess.run( [loss, joint_pos], 
                                        feed_dict = { frames_input: x_valid_batch, 
                                                          y_target: y_valid_batch} )

                # calculate the test accuracy
                acc_valid = calc_acc( y_valid_pred, y_valid_batch)

                # computer average_loss, note: loss_train has been divded by batch_size
                valid_avg_acc  += acc_valid / valid_batch
                valid_avg_loss += loss_valid / valid_batch

                # concatenate batch predictions into the a prediction for the current epoch
                yall_valid_pred[sample_indices, :, :, :] = y_valid_pred

            # record the train loss
            train_loss_history.append( train_avg_loss )
            train_acc_history.append( train_avg_acc )
            valid_loss_history.append( valid_avg_loss )
            valid_acc_history.append( valid_avg_acc )

            # record the prediction for confusion matrix 
            valid_pred_log[epoch] = yall_valid_pred

            if epoch % display_steps == 0:
                print( 'epoch: {}, train_loss: {}, valid_loss: {} '\
                    .format(epoch, np.round(train_avg_loss, 4), np.round(valid_avg_loss, 4) ))

        # save session 
        save_path = saver.save(sess, 'my_model')

    ## 4. Plots
    yall_valid_pred = valid_pred_log[25]

        
    # plot train and test curve 
    fig1 = plot_loss_results(train_loss_history, valid_loss_history )
    plt.savefig( 'figure/overall_loss-lr={}.png'.format( lr ) )

    # plot confusion matrix
    fig2 = plot_distance( yall_valid_pred,y_valid )
    plt.savefig( 'figure/confusion_matrix-lr={}.png'.format( lr ) )

    # plot cummulative distribution function
    fig3 = plot_acc_vs_pixels( yall_valid_pred,y_valid)
    plt.savefig( 'figure/cumulative_joints-lr={}.png'.format( lr ) )

    #  plot random example using the final model 
    samples = np.random.choice( 1000, 6 )
    data = x_valid[samples, :, :, :, :]
    plot_examples(data)
    plt.savefig( 'figure/random_examples-lr={}.png'.format( lr ) )


