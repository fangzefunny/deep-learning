import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt 
import pickle

## utils: normalize and split the data 
def split_data( data, label ):
    
    #  check the input sample size
    tot_sample = data.shape[0]
    
    # mannual train sample as 5500, valid sample as 1500
    train_num = 5500
    valid_num = 1500
    
    # convert the uni8 datatype to float32
    data = data.astype( np.float32)
    
    # normalize the data
    data -= np.mean(data, axis=(2, 3, 4), keepdims=True)
    data /= np.std( data, axis=(2, 3, 4), keepdims=True)
    
    # create the total data indeices
    tot_idx = np.arange(0, tot_sample)
    
    # shuffle the indince array
    np.random.RandomState(2020).shuffle(tot_idx)
    
    # train_idx, test_idx
    train_idx, valid_idx = tot_idx[ 0 :5500], tot_idx[ 5500 :tot_sample]
    
    # split the train and test 
    train_data, valid_data = data[ train_idx, :, :, :, : ], data[ valid_idx, :, :, :, : ]
    train_label, valid_label = label[ train_idx, :, :, : ], label[ valid_idx, :, :, : ]
    
    return train_data, train_label, valid_data, valid_label 

#  convolution function 
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
    # h_conv1 = conv2d( X,W) +b
    c1 = conv2d( x_input, weight_x1) + bias_x1 #c1: 60 x 60 x 32
    p1 = max_pool( c1 ) #p1: 30 x 30 x 32

    # weight: conv2
    weight_x2 = tf.get_variable( 'weight_x2', 
                                shape = [5, 5, 32, 32 ],
                                initializer = weight_init)
    bias_x2 = tf.get_variable( 'bias_x2', shape = [32,],
                                initializer = bias_init ) 
    # h_conv1 = conv2d( X,W) +b
    c2 = conv2d( p1, weight_x2) + bias_x2 #c2 no padding: 26x 26 x 32
    p2 = max_pool( c2 ) #p1: 13 x 13 x 32

    # weight: conv3
    weight_x3 = tf.get_variable( 'weight_x3', 
                                shape = [3, 3, 32, 64 ],
                                initializer = weight_init)
    bias_x3 = tf.get_variable( 'bias_x3', shape = [64,],
                                initializer = bias_init ) 
    # h_conv1 = conv2d( X,W) +b
    c3 = conv2d( p2, weight_x3) + bias_x3 #c3 no padding: 11 x 11 x 64

    # flatten 
    p_vec = tf.reshape( c3, [-1, 11 * 11 * 64]) # batch_size x (11 * 11 * 64)
    
    #  fully connect 
    weight_phi = tf.get_variable( 'weight_phi', 
                                shape = [3 * 3 * 64, n_features],
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


if __name__ == '__main__':
    
     ## 0. HYPERPARAMETER TUNING
        
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--lr', help='learning rate', type=float, default=0.0001)

    # args = parser.parse_args()
    
    ## 1. LOAD THE DATA
    
    # load the data 
    with open('youtube_train_data.pkl', 'rb') as handle:
        train_data, train_labels = pickle.load(handle)
        
    # normalize and split_data 
    x_train, y_train, x_valid, y_valid  = split_data( train_data, train_labels)
    
    
    ## 2. PREPARE THE COMPUTATIONAL GRAPH 
    
    # hyper paramters
    lr          = args.lr  # learning rate
    batch_size  = 500      # number fo the sample with a batch  
    
    #  fix some useful values 
    save_steps    = 1                          # save the sess every n epoch 
    display_steps = 10                      # print epoch loss 
    max_epoch     = 300                     # max epoch 
    total_samples = x_train.shape[0]        # total sample number
    img_width     = x_train.shape[1]        # width of the image
    img_height    = x_train.shape[2]        # height of the image
    img_channel   = x_train.shape[3]        # num of channel of the image
    img_channel   = x_train.shape[4]        # num of channel of the image
    batch_num     = total_samples // batch_size # num of mini batches
    train_indices = np.arange(0, total_samples) # indices to help shuffle
    rng = np.random.RandomState( 2020 ) # to help generate random things
    
    # storages
    train_loss_history = []  # record the train loss
    test_loss_history  = []  # record the test loss
    train_acc_history  = []  # record the train acc
    test_acc_history   = []  # record the test acc 
    test_pred_log = dict()  # save the y_pred for confusion matrix 