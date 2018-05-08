###############################################################################
# Author: Wang Xin                                                            #
# Function: Input embeddings to get the search accuracy                       #
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import lfw
import os
import sys
import math
import pickle
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
from search_acc import SearchFabric 

#arg_lfw_pairs = '/media/wangx/HDD1/yarn-dyed-fabric/pairs_fabrics.txt' 

def main(args):
    """
    Output embedding of image as numpy array, npy file will be dumped in
    current directroy.
    
    Usage:
        python get_emb_array.py --lfw_dir /media/wangx/HDD1/yarn-dyed-fabric/crop1600resize224/val \
                                --model_dir /home/wangx/models/yarn-dyed-fabric/20171202-234221


    --lfw_dir: Directroy store test images
    --model_dir: Directroy checkpoint and meta files
    """

    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            # Get the paths for the corresponding images
            paths = []          
            for root, dirs, files in os.walk(args.lfw_dir):
                for f in files:
                    paths.append(os.path.join(root, f))
            assert len(paths) > 0, "Number of paths is 0"

            # Load the model
            facenet.load_model(args.model_dir)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            
            image_size = images_placeholder.get_shape()[1]
            embedding_size = embeddings.get_shape()[1]
            
            print('Running forward pass now.')
            batch_size = args.lfw_batch_size
            nrof_images = len(paths)
            nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches):
                start_index = i*batch_size
                end_index = min((i+1)*batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
            
            # Dump np array and image paths
            emb_array_name = os.path.basename(args.model_dir.rstrip('/')) + '_emb.pkl'
            emb_paths_name = os.path.basename(args.model_dir.rstrip('/')) + '_emb_paths.pkl'
            pickle.dump(emb_array, open(emb_array_name,'wb'))        
            print('Saved emb_array: %s' % emb_array_name)
            pickle.dump(paths, open(emb_paths_name,'wb'))
            print('Saved emb_paths: %s' % emb_paths_name)
        
    print("Start get top_n search accuracy...")
    SearchModule = SearchFabric(paths=emb_paths_name,
                                emb_array=emb_array_name)
    accuracy_in_top_n, _ = SearchModule.quick_accuracy_along_top_n(20)
    for i in range(10):
        print("The top %s accuracy is %s" % (i+1, accuracy_in_top_n[i]))

def parse_arguments(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--lfw_dir', type=str,
                        help="Directory stores test images")
    parser.add_argument('--model_dir', type=str,
                        help="Directory stores checkpoint and mete files")
    parser.add_argument('--lfw_file_ext', type=str,
                        help="Image format", default='jgp')
    parser.add_argument('--lfw_batch_size', type=int,
                        help="Size of batch to feed", default=160)
    return parser.parse_args(args)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
