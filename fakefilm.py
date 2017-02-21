#!/usr/bin/env python

import os, sys, argparse
import tensorflow as tf
import numpy as np
import cv2

def resize_image(img, size=(320, 240)):
    """
    Resize image and get the middle part.
    :param img numpy.array:
        The image data
    :param size tuple:
        The resized image dimensions
    :returns numpy.array:
        A numpy.array with the resized image
    """
    sw, sh = size
    h, w = img.shape[:2]
    nsize = (320, round(h*sw/w)) if h / sh > w / sw else (round(w*sh/h), 240)
    res = cv2.resize(img, nsize) # interpolation=cv2.INTER_CUBIC)
    ho, wo = res.shape[:2]
    wo, ho = (wo-nsize[0])//2, (ho-nsize[1])//2
    return res[0+ho:sh+ho, 0+wo:sw+wo]

x = tf.placeholder("float", shape=[None, 76800], name="x_image")
y_ = tf.placeholder("float", shape=[None, 1], name="y_")

def weight_variable(shape, name=None):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)

def bias_variable(shape, name=None):
    return tf.Variable(tf.constant(0.1, shape=shape), name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x, size):
    return tf.nn.max_pool(x, ksize=[1, size, size, 1],
            strides=[1, size, size, 1], padding='SAME')

W_conv1 = weight_variable([12, 16, 1, 4], "W_conv")
b_conv1 = bias_variable([4], "b_conv")

x_image = tf.reshape(x, [-1,240,320,1])

h_conv1 = tf.nn.sigmoid(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool(h_conv1, 20)

W_fc1 = weight_variable([12*16*4, 64])
b_fc1 = bias_variable([64])

h_pool1_flat = tf.reshape(h_pool1, [-1, 12*16*4])
h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([64, 1], "W_fc2")
b_fc2 = bias_variable([1], "b_fc2")

a_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_output = tf.nn.sigmoid(a_conv)
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(a_conv, y_)
train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(cross_entropy)

correct_prediction = tf.equal(tf.round(y_output), y_)
#classification = tf.reduce_mean(tf.cast(correct_prediction, "float"))
classification = tf.reduce_sum(tf.cast(correct_prediction, "float"))

saver = tf.train.Saver()

def main():
    parser = argparse.ArgumentParser()
    # Required arguments: input file.

    parser.add_argument(
        "image_files",
        help="Path to the input image file",
        nargs='+',
    )
    # Optional arguments.
    parser.add_argument(
        "-m", "--model",
        help="Model definition path.",
        default="model/fakefilm",
    )

    args = parser.parse_args()

    print("if", args.image_files)
    print("model", args.model)

    gpaths = []
    images = []
    for fpath in args.image_files:
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        img = resize_image(img)
        img = img.astype(np.float32)/256
        img = img.reshape((76800))
        images.append(img)
        gpaths.append(fpath)
    if not gpaths:
        exit()

    images = np.concatenate(images,axis=0)
    images = images.reshape((len(images)//76800, 76800))
    flags = np.zeros((len(images), 1))

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        ## sess.run(tf.global_variables_initializer())
        saver.restore(sess, args.model)
        scores = y_output.eval(feed_dict={ x:images, keep_prob: 1.0})
        for fn, s in zip(gpaths, scores):
            print(fn, s[0])

    """
        import transdata
        td = transdata.Transdata()
        dotest = lambda images, flags: int(classification.eval(
                feed_dict={ x:images, y_: flags, keep_prob: 1.0}))
        ckppath = 'model/fakefilm.ckpt'
        for i in range(5000):
            images, flags = td.next_train_batch()
            if (i+1)%100 == 0:
                train_classification = classification.eval(feed_dict={
                    x:images, y_: flags, keep_prob: 1.0})
                print("step %d, training classification %g/%d"%(i+1, train_classification, len(flags)))
            if (i+1)%1000 == 0:
                print("save model step >>>>>> ", i+1)
                saver.save(sess, ckppath, global_step=i+1)
                td.testall(dotest)
            train_step.run(feed_dict={x: images, y_: flags, keep_prob: 0.5})
    """

if __name__ == '__main__':
    main()

