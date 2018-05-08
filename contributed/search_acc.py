###############################################################################
# Author: Wang Xin                                                            #
# Function: Input embeddings to get the search accuracy                       #
###############################################################################

from __future__ import print_function
from __future__ import division
import pickle
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import os 
import sys
import random
import tensorflow as tf
import facenet

emb_array = '/home/wangx/logs/yarn-dyed-fabric/20171202-234221/20171202-234221_emb.npy'
paths = '/home/wangx/logs/yarn-dyed-fabric/20171202-234221/20171202-234221_emb_paths.txt'
model_dir = '/home/wangx/models/yarn-dyed-fabric/20171202-234221'
#model = SearchFabric(paths=paths, emb_array=emb_array)

class SearchFabric(object):
    """
    If given image paths and their embedding, this class have functions to 
    return the result of search similar images.

    Methods:
        get_similar:
        show_search:
        get_search_acc_top_n:
    """

    def __init__(self, paths=paths, emb_array=emb_array):
        """
        Args:
            emb_array: name of a file stores embedding of images 
            paths: name of a file stores paths of images, each image should
                   has embedding in emb_array with same index
        """
        emb_array = pickle.load(open(os.path.expanduser(emb_array), 'rb'))
        paths = pickle.load(open(os.path.expanduser(paths), 'rb'))
        assert emb_array.shape[0] > 0, "Number of emb_array is 0"
        assert len(paths) > 0, "Number of paths is 0"
        self.emb_array = emb_array
        self.paths = paths


    def l2_dist(self, v1, v2):
        """
        Args:
            v1, v2: numpy array
        """
        return ((v1 - v2)**2).sum()
        

    def is_same(self, i1, i2):
        """
        Args:
            i1, i2: index of two images
        Return:
            bool: True if they are same fabric
        """
        p1 = self.paths[i1]
        p2 = self.paths[i2]
        return os.path.dirname(p1) == os.path.dirname(p2)
       

    def get_similar(self, embed):
        """
        Return a index list, the most similar embedding's index will be sorted at begining
        Args:
            embed: numpy array, embedding of images
        """
        # Compute l2 with all other embedding
        all_dist = [self.l2_dist(embed, other) for other in self.emb_array]
        
        # Get most similar image's index 
        sorted_dist = [i[0] for i in sorted(enumerate(all_dist), key=lambda x:x[1])]
        return sorted_dist


    def show_search(self, index, top_n=6):
        """
        Input an index of self.emb_array, show the index of the most similar image
        Args:
            index: index in self.emb_array
            top_n: show the most n result of similar image
        Return:
        Print:
            The top n index
            N similar image, the first one must be itself, so similar image start from
            second one
        """
        similar = self.get_similar(self.emb_array[index])[:top_n]
        fig, axs = plt.subplots(1,top_n, figsize=(10,2))
        print("Show top n index: ")
        print(similar)  
        # Show top_n images
        for j in range(top_n):
            axs[j].axis('off')
            img = PIL.Image.open(self.paths[similar[j]],'r')
            axs[j].imshow(img)
            dist = self.l2_dist(self.emb_array[index],
                                self.emb_array[similar[j]])
            print("%s: %s" % (self.paths[similar[j]].split('/')[-1],  dist))
        plt.show()


    def show_search_from_path(self, filename, model_dir, top_n=6):
        """
        Args:
            filename: path of image file in disk
            model_dir: path of trained model, for example, '/home/wangx/models/yarn-dyed-fabric/20171202-234221'
            top_n: show top n search results
        """
        # get one embedding based on filename
        with tf.Graph().as_default():
            with tf.Session() as sess:
                # Load model
                facenet.load_model(model_dir)

                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                
                image_size = images_placeholder.get_shape()[1]
                embedding_size = embeddings.get_shape()[1]

                print("Running forward pass now.")
                images=facenet.load_data([filename], False, False, image_size)
                pickle.dump(images[0], open('images0.pkl','wb'))
                feed_dict = {images_placeholder:images,
                             phase_train_placeholder:False}
                emb_array = sess.run(embeddings, feed_dict=feed_dict)

        # Search similar images
        similar = self.get_similar(emb_array[0])[:top_n]
        fig, axs = plt.subplots(1,top_n, figsize=(10,2))
        print("Show top n index: ")
        print(similar)  
        # Show top_n images
        for j in range(top_n):
            axs[j].axis('off')
            img = PIL.Image.open(self.paths[similar[j]],'r')
            axs[j].imshow(img)
            dist = self.l2_dist(emb_array,
                                self.emb_array[similar[j]])
            print("%s: %s" % (self.paths[similar[j]].split('/')[-1],  dist))
        plt.show()


    def get_search_acc_top_n(self, top_n = 3):
        """
        Calculate search accuracy while using current emb_array.
        Args:
            top_n: define accuracy of correctly find same image in top_n
        Return:
            accuracy: top_n accuracy
            false_search: index of image which can not be correctly found
        """
        count = 0
        false_search = []
        for i, embed in enumerate(self.emb_array):
            similar = self.get_similar(self.emb_array[i])
            
            # Check if correct image in top_n
            for n in range(top_n):
                # similar[0] is always itself
                if self.is_same(similar[0], similar[n+1]):
                    count += 1
                    break
                else:
                    false_search.append(i)
            
            # Show test progress
            sys.stdout.write('Test completed %.3f %%\r' % (float(i+1)/len(self.emb_array)*100))
            sys.stdout.flush()

        accuracy = float(count)/len(self.emb_array)      
        print("Count of correct search: ", count)
        print("Search accuracy: ", accuracy)
        
        return accuracy, false_search


    def draw_accuracy_along_top_n(self, before_top_n = 10):
        """
        Get each top_n accuracy from 1 to before_top_n, it will take a long time,
        then plot accuracy accuracy.

        Args:
            before_top_n: int
        Return:
            acc_history: list of accuracy history
        """
        acc_history = []
        for i in range(before_top_n):
            print('Start test top %s' % (i+1))
            current_acc, _ = self.get_search_acc_top_n(i+1) 
            acc_history.append(current_acc)
        plt.plot(acc_history)
        plt.title('Accuracy history')
        ax = plt.gca()
        ax.set_xticks(list(range(before_top_n)))
        ax.set_xticklabels(list(range(1,1+before_top_n)))
        plt.show()
        with open('acc_history.txt') as f:
            for x in acc_history:
                f.write(str(x) + '\n')
        return acc_history


    def quick_accuracy_along_top_n(self, before_top_n = 10):
        """get accuracy in top n by iterate emb_array once, which is quickly

        Args:
            before_top_n: int, if correct image found in 0,1,2,...,n
        Return:
            accuracy_in_top_n: list of floats, top1,top2,...,top_n accuracy
            false_search_top_nl: list of lists, false research while
                                 top1,top2,...,top_n
        """
        found_in_top_n = [0]*before_top_n
        false_search_top_n = []
        for i in range(before_top_n):
            false_search_top_n.append([])

        for i, embed in enumerate(self.emb_array):
            similar = self.get_similar(self.emb_array[i])
            # Check if correct image in top_n
            for n in range(before_top_n):
                # similar[0] is always itself
                if self.is_same(similar[0], similar[n+1]):
                    for idx in range(n,before_top_n):
                        found_in_top_n[idx] += 1
                    break
                else:
                    false_search_top_n[n].append(i)
            # Show test progress
            sys.stdout.write('Test completed %.3f %%\r' % (float(i+1)/len(self.emb_array)*100))
            sys.stdout.flush()
        accuracy_in_top_n = [i/len(self.emb_array) for i in found_in_top_n]

        ask = raw_input("Do you want to save acc_history now? (y/n): ")
        if ask == 'y' or ask =='Y':
            with open('acc_history.txt', 'w') as f:
                for x in accuracy_in_top_n:
                    f.write(str(x) + '\n')
            print("Already saved acc_history.txt.")
            with open('false_search_top_n.pkl', 'wb') as f:
                pickle.dump(false_search_top_n, f)
            print("Already dumped false_search_top_n.pkl")

        return accuracy_in_top_n, false_search_top_n
