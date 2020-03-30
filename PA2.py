# Submitted by

# S K Aravind, SXA190006

# Paresh Dashore, PXD190004

# PA2.py (Ensemble Methods)



import numpy as np
np.seterr(divide='ignore', invalid='ignore')
np.random.seed(69)

import os

import math

from time import time

import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import confusion_matrix

# seaborn for plotting heatmap of confusion matrix
import seaborn as sn

from sklearn.tree import DecisionTreeClassifier

from sklearn import tree




def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)
    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """
    
    k=dict()
    for i in range(len(Xtrn[0])):
        temp=str(i+1)
        unique = np.unique(Xtrn[:,i])
        k[temp] = unique
                
    return k


def entropy(y):
    """ 
    Computes entropy of label distribution. 
    """

    n_labels = len(y)
    if n_labels <= 1:
        return 0
    value,counts = np.unique(y, return_counts=True)
    probs = counts / n_labels
    ent = 0.
    for i in probs:
        ent += i * math.log(i,2)

    return ent




def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.
    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """

    Hy = x
    Hyx = y

    return Hy-Hyx




def add_to_tree(u,k): 
    node=[]
    Hy=-entropy(u[:,-1])
    
    max_gain = -float('inf')
    N = len(u)
    for i in k:
        mask = u[:,int(i)-1]
        for j in k[i]:
            p = u[mask==j]
            q = u[mask!=j]
            sum1=-1*(entropy(p[:,-1]))*(len((p))/N)
            sum2=-1*(entropy(q[:,-1]))*(len((q))/N)
            information_gain=mutual_information(Hy,sum1+sum2)
            if max_gain <= information_gain:
                feature = i
                feature_node = j
                max_gain = information_gain
    
    return feature_node,feature


def helper(u,k,max_depth=0):
    ## The variable k gives us the attribute value pair
    if max_depth==0:
        return majority(u)
    
    if purity(u):
        return majority(u)
        
    if len(k)==0:
        return majority(np.column_stack((Xtrn, ytrn)))
    
    
    feature_node,feature=add_to_tree(u,k)
    u_check = u[:,int(feature)-1]
    u_false=u[u_check!=feature_node]
    u_true=u[u_check==feature_node]
    k_false=partition(u_false)
    k_true=partition(u_true)
    
    return {
        (feature,feature_node,False):helper(u_false,k_false,max_depth-1),
        (feature,feature_node,True):helper(u_true,k_true,max_depth-1)
    }
    

def majority(u):
    ytrn=u[:,-1]
    counts = np.bincount(ytrn)
    return np.argmax(counts)

def purity(u):
    ytrn=u[:,-1]
    if len(np.unique(ytrn)) == 1:
        return True
    else:
        return False

    
def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider.
    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """
    Xtrn = x
    ytrn = y
    u=np.column_stack((Xtrn, ytrn))

    ## The variable k gives us the attribute value pair
    k=partition(Xtrn)
    return helper(u,k,max_depth)




def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.
    Returns the predicted label of x according to tree
    """

    if type(tree) != dict:
        return tree

    feature = int(list(tree.keys())[0][0])-1
    feature_val = list(tree.keys())[0][1]
    if x[feature] == feature_val:
        return predict_example(x, tree[list(tree.keys())[1]])
    else:
        return predict_example(x, tree[list(tree.keys())[0]])



def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)
    Returns the error = (1/n) * sum(y_true != y_pred)
    """

    # mse
    return (1/len(y_pred)) * sum(y_true != y_pred)


    

def bagging(x,y,max_depth,num_trees):
    """
    Bagging algorithm implemented from scratch. Given the training data and labels, max_depth and number stumps, return an array of h_i, i.e. all decision tree hypothesis.
    """

    k=list()
    for i in range(num_trees):
        
        index = np.random.choice(x.shape[0], x.shape[0] ,replace=True)
        decision_tree = id3(x[index], y[index],  max_depth=max_depth)
        k.append(decision_tree)
        
    return k


def prediction_example(x, h_ens):
    """
    Function to predict the label of a datapoint given the h_ens array of alpha_i and h_i
    """
    
    # Bagging case
    if h_ens[0][0]=='bag':
        d=dict()
        max_count = 0
        max_attr = None
        for i in range(len(h_ens)):
            a=predict_example(x, h_ens[i][1])
            if a not in d:
                d[a]=1
            else:
                d[a]+=1
            if d[a] >= max_count:
                max_count = d[a]
                max_attr = a
        return max_attr
    
    # Boosting case
    else:
        d=dict()
        max_count = 0
        max_attr = None
        for i in h_ens:
            prediction=predict_example(x, i[1])
            if prediction not in d:
                d[prediction]=i[0]
            else:
                d[prediction]+=i[0]
            if d[prediction] >= max_count:
                max_count = d[prediction]
                max_attr = prediction
                
        return max_attr

    

def boosting(x, y, max_depth, num_stumps):
    """
    Boosting algorithm implemented from scratch. Given the training data and labels, max_depth and number stumps, return an array of tuples (alpha_i, h_ens), the alpha value along with all decision tree hypothesis.
    """
    
    h_ens=list()
    df=np.column_stack((x, y))
    sign = np.array([1]*len(x))
    
    for i in range(num_stumps):
        w=np.array([1/len(x)]*len(x))
        error=0
        tree=id3(x, y, max_depth = max_depth)
        
        for j in range(len(y)):
            prediction = predict_example(x[j], tree)
            if prediction != y[j]:
                error += w[j]
                sign[j] = 1
            else:
                sign[j] = -1
        
        alpha_i=(1/2)*math.log((1-error)/(error+ 1e-5))
        
        h_ens.append((alpha_i, tree))
        
        w = w * np.exp(sign * alpha_i)
            
        normalization_sum = np.sum(w)
        w=w/normalization_sum
            
        indices = [i for i in np.random.choice(x.shape[0],x.shape[0] , p=w, replace=True)]
        x=x[indices]
        y=y[indices]   
            
    return h_ens


       


if __name__ == '__main__':

    # Load the training data
    M = np.genfromtxt('mushroom.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)

    ytrn = M[:, 0]

    Xtrn = M[:, 1:]

    # Load the test data

    M = np.genfromtxt('mushroom.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)

    ytst = M[:, 0]

    Xtst = M[:, 1:]
    
    start = time()
    print('Bagging, Our Algo...')

    ######## part a ###########
    #Bagging
    sn.set(font_scale=1.4)
    index=1
    for max_depth in [3,5]:

        for num_trees in [10,20]:

            k=bagging(Xtrn,ytrn,max_depth,num_trees)
            h_ens=list()
            for i in k:
                h_ens.append(('bag',i))
            
            y_pred = [prediction_example(i,h_ens) for i in Xtst]
                

            cf_mat = confusion_matrix(ytst,y_pred)
            
            print('Test Error Our Algo: ',compute_error(ytst, y_pred))
            plt.subplot(2,2,index)
            index+=1
            plt.title('Depth : '+str(max_depth)+' stump_count: '+str(num_trees))
            sn.heatmap(cf_mat, annot=True, annot_kws={"size": 16}, fmt='g')

        plt.suptitle('Confusion Matrix Our Algo', fontsize=12)
        plt.tight_layout()
    plt.show()

        
    print('Boosting, Our Algo...')

    ######## part b ###########
    #Boosting
    sn.set(font_scale=1.4)
    index=1
    for height in [1,2]:
        
        for stump_count in [20,40]:
            
            h_ens=boosting(Xtrn,ytrn,height,stump_count)
            
            y_pred = [prediction_example(i,h_ens) for i in Xtst]
            
            cf_mat = confusion_matrix(ytst,y_pred)

            print('Test Error Our Algo: ',compute_error(ytst, y_pred))

            plt.subplot(2,2,index)
            index+=1
            plt.title('Depth : '+str(height)+' stump_count: '+str(stump_count))
            sn.heatmap(cf_mat, annot=True, annot_kws={"size": 16}, fmt='g')

        plt.suptitle('Confusion Matrix Our Algo', fontsize=12)
        plt.tight_layout()
    plt.show()


    ######## part c ###########

    #bagging using scikit-learn
    
    end = time()
    
    # print('Time taken: ',end-start)
    # print('Bagging, Sklearn...')

    from sklearn.ensemble import BaggingClassifier
    from sklearn.utils import resample
    sn.set(font_scale=1.4)
    index=1
    for max_depth in [3,5]:

        for num_trees in [10,20]:
            tree = BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=0, max_depth=max_depth), n_estimators=num_trees, random_state=69)
            tree.fit(Xtrn, ytrn)

            y_pred = tree.predict(Xtst)

            print('Test Error Sklearn: ',compute_error(ytst, y_pred))

            cf_mat = confusion_matrix(ytst,y_pred)
            plt.subplot(2,2,index)
            index+=1
            plt.title('Depth : '+str(max_depth)+' \nnum_of_trees: '+str(num_trees))
            sn.heatmap(cf_mat, annot=True, annot_kws={"size": 16}, fmt='g')

        plt.suptitle('Confusion Matrix (Sklearn)', fontsize=12)
        plt.tight_layout()
    plt.show()


    #boosting using scikit-learn
    
    print('Boosting, Sklearn...')
    sn.set(font_scale=1.4)
    index=1
    for max_depth in [1,2]:

        for n_estimators in [20,40]:

            tree = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth),algorithm="SAMME",n_estimators=n_estimators)
            tree.fit(Xtrn, ytrn)

            y_pred = tree.predict(Xtst)

            print('Test Error Sklearn: ',compute_error(ytst, y_pred))
            
            cf_mat = confusion_matrix(ytst,y_pred)
            plt.subplot(2,2,index)
            index+=1
            plt.title('Depth : '+str(max_depth)+' \nnum_of_trees: '+str(n_estimators))
            sn.heatmap(cf_mat, annot=True, annot_kws={"size": 16}, fmt='g')

        plt.suptitle('Confusion Matrix (Sklearn)', fontsize=12)
        plt.tight_layout()
    plt.show()
            


        
        