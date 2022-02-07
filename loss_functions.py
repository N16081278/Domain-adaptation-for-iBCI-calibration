import tensorflow as tf
import tensorflow.keras.backend as K

# def gaussian_kernel(x1, x2, beta = 1.0):
#     r = tf.transpose(x1)
#     r = tf.expand_dims(r, 2)
#     return tf.reduce_sum(K.exp( -beta * K.square(r - x2)), axis=-1)
  
# def maximumMeanDiscrepancy(x1, x2, beta=1.0):
#     """
#     maximum mean discrepancy (MMD) based on Gaussian kernel
#     function for keras models (theano or tensorflow backend)
    
#     - Gretton, Arthur, et al. "A kernel method for the two-sample-problem."
#     Advances in neural information processing systems. 2007.
#     """
#     x1x1 = gaussian_kernel(x1, x1, beta)
#     x1x2 = gaussian_kernel(x1, x2, beta)
#     x2x2 = gaussian_kernel(x2, x2, beta)
#     diff = tf.reduce_mean(x1x1) - 2 * tf.reduce_mean(x1x2) + tf.reduce_mean(x2x2)
#     return diff


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    params:
        source: sample_size_1 * feature_size 
        target: sample_size_2 * feature_size 
        kernel_mul: 计算每个核的bandwith
        kernel_num: 多核的数量
        fix_sigma: 固定的标准差
    return: (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2)的
                        矩阵，表达形式:
                        [   K_ss K_st
                            K_ts K_tt ]
    """
    
    n_samples = int(tf.shape(source)[0].numpy()) + int(tf.shape(target)[0].numpy())
    total = tf.concat([source, target], axis=0)
    #将total复制（n+m）份
    bs_cnt = int(tf.shape(total)[0].numpy())
    feature_cnt = int(tf.shape(total)[1].numpy())

    total0 = tf.broadcast_to(tf.expand_dims(total, axis=0), shape=[bs_cnt, bs_cnt, feature_cnt]) #将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = tf.broadcast_to(tf.expand_dims(total, axis=1), shape=[bs_cnt, bs_cnt, feature_cnt])
    L2_distance = tf.reduce_sum((total0-total1)**2, axis=2) # 计算高斯核中的|x-y|

    # sigma of guassian_kernel
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = tf.reduce_sum(L2_distance) / (n_samples**2-n_samples)
    # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

    # 高斯核的公式，exp(-|x-y|/bandwith)
    kernel_val = [tf.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    
    return sum(kernel_val) # 将多个核合并在一起

def maximumMeanDiscrepancy(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n = int(tf.shape(source)[0].numpy())
    m = int(tf.shape(target)[0].numpy())

    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
   
    XX = kernels[:n, :n] 
    YY = kernels[n:, n:]
    XY = kernels[:n, n:]
    YX = kernels[n:, :n]

    XX = tf.reshape(tf.reduce_sum(tf.divide(XX, n * n), axis=1), shape=[1, -1])  # K_ss矩阵，Source<->Source
    XY = tf.reshape(tf.reduce_sum(tf.divide(XY, -n * m), axis=1), shape=[1, -1]) # K_st矩阵，Source<->Target
    YX = tf.reshape(tf.reduce_sum(tf.divide(YX, -m * n), axis=1), shape=[1, -1]) # K_ts矩阵,Target<->Source
    YY = tf.reshape(tf.reduce_sum(tf.divide(YY, m * m), axis=1), shape=[1, -1]) # K_tt矩阵,Target<->Target

    	
    loss = tf.reduce_sum(XX + XY) + tf.reduce_sum(YX + YY)
    
    return loss