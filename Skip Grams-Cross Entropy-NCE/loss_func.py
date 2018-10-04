import tensorflow as tf

def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    A =


    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B =

    ==========================================================================
    """
    A = []
    B = []
    A.append(tf.log(tf.exp(tf.diag_part(tf.matmul(inputs,tf.transpose(true_w))))))
    B.append(tf.log(tf.reduce_sum(tf.exp(tf.matmul(inputs,tf.transpose(true_w))),axis=1)))  #Cross Entropy implementation

    return tf.subtract(B, A)

def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weigths: Weights for nce loss. Dimension is [Vocabulary, embedding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimesion is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    """
    A = []
    B = []

    k=sample.shape[0]
    b=inputs.shape[0]
    e=inputs.shape[1]                                                                           #NCE implementation
    unigram_prob_tf=tf.convert_to_tensor(unigram_prob)
    u_o=tf.nn.embedding_lookup(weights, tf.reshape(labels, [b, ]))
    A1 = tf.diag_part(tf.matmul(inputs,tf.transpose(u_o)))+tf.reshape(tf.nn.embedding_lookup(biases,labels),[b,])
    A2 = tf.log(tf.scalar_mul(k,tf.nn.embedding_lookup(unigram_prob_tf,tf.reshape(labels,[b,]))))
    A.append(tf.log_sigmoid(tf.subtract(A1,A2)))
    u_x=tf.nn.embedding_lookup(weights,sample)
    B1 = tf.matmul(inputs,tf.transpose(u_x))+tf.tile(tf.reshape(tf.nn.embedding_lookup(biases,sample),[1,k]),[b,1])
    B2 = tf.tile(tf.log(tf.scalar_mul(k,tf.reshape(tf.nn.embedding_lookup(unigram_prob_tf,sample),[1,k]))),[b,1])
    B.append(tf.reduce_sum(tf.log(1-tf.sigmoid(tf.subtract(B1, B2))+1e-10),axis=1))

    return tf.scalar_mul(-1,tf.add(A,B))