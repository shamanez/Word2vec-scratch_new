import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad


def normalizeRows(x):
    """ Row normalization function """
    # Implement a function that normalizes each row of a matrix to have unit length
    
    ### YOUR CODE HERE
    y = np.linalg.norm(x,axis=1,keepdims=True)
    x /= y
    ### END YOUR CODE
    return x

def l1_normalize_rows(x):
    """ l1 row normalization function """
    y = None

    y = np.sum(x,axis=1,keepdims=True)
    x /= y
    return x

def l2_normalize_rows(x):
    """ l1 row normalization function """
    y = None

    y = np.linalg.norm(x,axis=1,keepdims=True)
    x /= y
    return x

def test_normalize_rows():
    print("Testing normalizeRows...")
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]])) 
    # the result should be [[0.6, 0.8], [0.4472, 0.8944]]
    print(x)
    assert (np.amax(np.fabs(x - np.array([[0.6,0.8],[0.4472136,0.89442719]]))) <= 1e-6)
    print("")

def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models """
    #print ("\n ##############Printing inside the softmazgardandgradient###############")

    #print("printing the predicted vec",predicted)
    #print("Prinitng the token  target which is the U vector",target)
#here the target means the token of the score function position that we need to maximize the log probability. again target is the position of the nearby word vector which is U we need to take. this will do for all vectors in each windowow. and we update parameters in each time.

    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, assuming the softmax prediction function and cross      
    # entropy loss.                                                   
    
    # Inputs:
    # - predicted: numpy ndarray, predicted word vector (\hat{r} in
    #   the written component)
    # - target: integer, the index of the target word
    # - outputVectors: "output" vectors (as rows) for all tokens
    # - dataset: needed for negative sampling, unused here.
    
    # Outputs:                                                        
    # - cost: cross entropy cost for the softmax word prediction
    # - gradPred: the gradient with respect to the predicted word
    #        vector
    # - grad: the gradient with respect to all the other word
    #        vectors
    
    # We will not provide starter code for this function, but feel
    # free to reference the code you previously wrote for this
    # assignment!
    
    ### YOUR CODE HERE
    N, D     = outputVectors.shape   #here this is 5*3 matrix
    r    = predicted #This is the predicted or normally the centerword or the input. 1*3 matric
    prob = softmax(r.dot(outputVectors.T))   #score function by multiplying the outputvectors(U) with the predicted vector(vc)(Input). 1*3 
    #here the output vectors are like the all posible U vectors(corpus) we multiply these vectors with input Vc vec. 
    #Here the prob will out put a matrix of 1*5 
    #print "printing the shape of prob vector"
    #print prob.shape  
    cost = -np.log(prob[target])   #taking the cross entrophy loss.   #with the target number this help to maximize the score of correct class
#use the operation like back prop to distributre gradietns 
    dx   = prob   #taking the gradient 
    dx[target] -= 1.  #Derivation of the cross entropy loss in the softmaz function 
   #here we have got 5 classes that means the input vec with 5 all the vec in the corpus(output U) 
   #now we have to transform gradients 
#here the gradietns is the global gradient with respect to the 
    grad     = dx.reshape((N,1)) * r.reshape((1,D))     # gradient with respect to the other vectors . that means the all 5 output vec. in each diemtion   #gradient of the 5 vectors 
#r.reshape((1,D))  this is actually 1*3 dimentional vactor 
#dx.reshape((N,1) this is 5*1 vector 
    #print grad.shape 
    gradPred = (dx.reshape((1,N)).dot(outputVectors)).flatten()  #gradient with respect to the predicted vector(vc) 1*3
    ### END YOUR CODE
#so here in get the gradients for all vectrs in the corpus then the gradient for the vc vector. also this goes iterativly in a windows
    return cost, gradPred, grad  #send the gradients and loss for the skipgram model. 5*3

def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, 
    K=10):
    """ Negative sampling cost function for word2vec models """
    #print "#############printing inside the negsampling loss function#########################"
#this also has implemented for skipgram model.
    #skip gram model
    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, using the negative sampling technique. K is the sample  
    # size. You might want to use dataset.sampleTokenIdx() to sample  
    # a random word index. 
    # 
    # Note: See test_word2vec below for dataset's initialization.
    #                                       
    # Input/Output Specifications: same as softmaxCostAndGradient     
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    ### YOUR CODE HERE
#K is number of negative sample vectors.
    N, D     = outputVectors.shape  #out put vectors are the total number of vectors ..

    cost     = 0
    gradPred = np.zeros_like(predicted)        #this is the gradients of the predicted vector. 
    grad     = np.zeros_like(outputVectors)    # this is for the gradients of the all the vector.

    #negative_samples = np.array([dataset.sampleTokenIdx() for i in range(K)], dtype='int64')
    negative_samples = []    
 #when using negative sample words we need to take the words that use more frequnetly.  also less frequenclty used words should play a role


    for k in range(K):   #this goes for 10 times. this is the negative sample . we see the product of input vc with other 10 u vectors
        new_idx = dataset.sampleTokenIdx()  #this is to genarate indexes for negative sampling .And there shouldn't be target variable.
        #print ("Printing the new index",new_idx)
        while new_idx == target:   #when the new_index is eual to the target U . we need to remove it from negsampling set
            new_idx = dataset.sampleTokenIdx() #so we need to remove that endex and add new index 
         
        negative_samples += [new_idx]    #adding the indexes to negative sample arrar.
  
    indices = [target]
    indices += negative_samples #here the indices matrix is a combination of target indices and other negative sampling indexes.
 

    labels = np.array([1] + [-1 for k in range(K)])   #lables vector . 1 for the first element then -1 for all other elements . 1*11 metric
    vecs = outputVectors[indices]  #this is 11*3 metrices 

#Here the predicted means the vc vector. 3*1 vector 
    
#elementvise multiplication. 1 elemet for the positve log prob and other one for the negative log prob
    z        = np.dot(vecs, predicted) * labels #element wise multiplication.   #this is 11*1 matrix
    probs    = sigmoid(z)  #this is 1*11 matrix 
    cost     = - np.sum(np.log(probs))  #this is the total cost function.  summing up the log loss , negative and postive both. This give the cost for one outside word in one window.

#Here the cost function is much more efficient since it don't have to calculate the expensive softmaz score function. 

    dx = labels * (probs - 1)   #taking the probabilities  Elementr wise operation. plus one in lable metrix for miximize u vec and other for outside U vects which we need to minimize  1*11 matrix. vec is 11*3 matrix 
    gradPred = dx.reshape((1,K+1)).dot(vecs).flatten()   #this gives the gradient of the vc.
    #print dx.reshape((1,K+1)).dot(vecs).shape 
    #print  gradPred.shape    #gradient for the Vc vector or the inside vec in the skip gram model.
    gradtemp = dx.reshape((K+1,1)).dot(predicted.reshape(1,predicted.shape[0])) #gradient of postive sample U vectors and negative sample 
    #print gradtemp.shape  #this is the gradient for  postive sample vec Uo and other U vectors in the negative sample vec 
#11*3 matrix
    #print gradPred.shape 
    #print gradtemp.shape 
    for k in range(K+1):
      #here the grad matrix is zeros of 5*3 it's like we have 5 corpus elements. so we need to add the gradients for each word place.
        grad[indices[k]] += gradtemp[k,:]
        
#     t = sigmoid(predicted.dot(outputVectors[target,:]))
#     cost = -np.log(t)
#     delta = t - 1
#     gradPred += delta * outputVectors[target, :]
#     grad[target, :] += delta * predicted
#     for k in xrange(K):
#         idx = dataset.sampleTokenIdx()
#         t = sigmoid(-predicted.dot(outputVectors[idx,:]))
#         cost += -np.log(t)
#         delta = 1 - t
#         gradPred += delta * outputVectors[idx, :]
#         grad[idx, :] += delta * predicted

    ### END YOUR CODE
  
    return cost, gradPred, grad   #cost as the loss for each u inside vec in each window. then the gradpred for 

#this is the skip gram model for the word to vec . where predict the context given the center word Vc
def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ Skip-gram model in word2vec """
    #print("\n ######### printing inside the skp gram model for each iteration  ###########")
    
    #print "Printing the current/Center word"


 

    # Implement the skip-gram model in this function.

    # Inputs:
    # - currentWord: a string of the current center word
    # - C: integer, context size
    # - contextWords: list of no more than 2*C strings, the context words
    # - tokens: a dictionary that maps words to their indices in
    #      the word vector list
    # - inputVectors: "input" word vectors (as rows) for all tokens
    # - outputVectors: "output" word vectors (as rows) for all tokens
    # - word2vecCostAndGradient: the cost and gradient function for
    #      a prediction vector given the target word vectors,
    #      could be one of the two cost functions you
    #      implemented above

    # Outputs:                                                        
    # - cost: the cost function value for the skip-gram model
    # - grad: the gradient with respect to the word vectors
    # We will not provide starter code for this function, but feel
    # free to reference the code you previously wrote for this
    # assignment!

    ### YOUR CODE HERE
    cost    = 0
    gradIn  = np.zeros_like(inputVectors)   #this is the input vectors(vc)    5*3 matrix
    gradOut = np.zeros_like(outputVectors)   #This is for out[ut vectprs. probably U vecotrs   5*3 matrix
#simple hashing method for find the Vc by it's token
    c_idx     = tokens[currentWord]  #token of the center words. When whe current word 
    #here the predicted word vec is rely on the center word(Vc)
    predicted = inputVectors[c_idx, :]   #this is 1*3 matrix which is the input(vc) vector to the algorithem. This change with the c_idx
    #__TODO__: can be switched to vectorized;
    # target (need to know shape; think its just a number)
    # hence target = np.zeros(len(contextWords))?
    # can add a newaxis(?) to allow for broadcasting
#here when sgd wrapper calls for this skipgram model
    
    for j in contextWords:   # we itarate the context word 
        target = tokens[j]  #this is the token of the surounding word that we work we need to give this as an input. vector U
        #print ("target this is the toekn of the first context word",target)
        c_cost, c_gradPred, c_grad = word2vecCostAndGradient(predicted, target, outputVectors, dataset) 
        cost += c_cost #add the cost
#this is the gradient decent . we update this for each context word in the windows. 
        #print cost 
        gradIn[c_idx,:] += c_gradPred  #here we fill the gradin vector with grad of the Vc with respect to each context word. we add the grad.
        gradOut         += c_grad   #addd the grad in all iterations 
        

    ### END YOUR CODE
    #print "Printing the gradin with respect to each context word in the windows"
    #print gradIn
    #print gradOut    
#still no parameter update has happened only gradients 
  
    return cost, gradIn, gradOut #this returns the total cost , total grads in inputvec and the total grads in all outvec(U). for a single window

def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ CBOW model in word2vec """
   
    # Implement the continuous bag-of-words model in this function.            
    # Input/Output specifications: same as the skip-gram model        
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    #################################################################
    # IMPLEMENTING CBOW IS EXTRA CREDIT, DERIVATIONS IN THE WRIITEN #
    # ASSIGNMENT ARE NOT!                                           #  
    #################################################################

    # Inputs:
    # - currrentWord: a string of the current center word
    # - C: integer, context size
    # - contextWords: list of no more than 2*C strings, the context words
    # - tokens: a dictionary that maps words to their indices in
    #      the word vector list
    # - inputVectors: "input" word vectors (as rows) for all tokens
    # - outputVectors: "output" word vectors (as rows) for all tokens
    # - word2vecCostAndGradient: the cost and gradient function for
    #      a prediction vector given the target word vectors,
    #      could be one of the two cost functions you
    #      implemented above

    # Outputs:
    # - cost: the cost function value for the skip-gram model
    # - grad: the gradient with respect to the word vectors
    # We will not provide starter code for this function, but feel
    # free to reference the code you previously wrote for this
    # assignment!
    
    cost    = 0
    gradIn  = np.zeros_like(inputVectors)   #5*3 matrix 
    gradOut = np.zeros_like(outputVectors) #5*3 matrix all the words.
    ### YOUR CODE HERE
    c_idx     = tokens[currentWord]  #index of the center word. 
    onehot    = np.zeros((len(contextWords), len(tokens)))  # this is 6*5 vector  
    #print tokens 
    
    
    #print contextWords 
  
   
    for i, word in enumerate(contextWords):  #This is a representation for each context word as a one hot vector  as a little corpus /
        #print i
        #print onehot[i, tokens[word]]
        onehot[i, tokens[word]] += 1.      #actually this representation helps to map the context word to relevent word vecotrs 
    
   #there are 5 input vectors. each 3 dimetional. a,b,c,d,e. 

    d = np.dot(onehot, inputVectors)  #this is a 4*3 vector. 
    predicted = (1.0 / len(contextWords)) * np.sum(d, axis=0)  #add up all the output word vectors and normalize them.
    cost, gradPred, gradOut = word2vecCostAndGradient(predicted, c_idx, outputVectors, dataset)  #then the normal procedure. send the predicted and try to mximize the probabilty of score of center word but here with respect to output wordvec.
#c_idx send the position of the  centor word in the score matrix in softmax. so it's so same with the other model. here what  
    
    gradIn = np.zeros(inputVectors.shape)   #5*3 gradient of the input vectors. 
    for word in contextWords:
        gradIn[tokens[word]] += (1.0 / len(contextWords) )* gradPred  #each context word is get updated with the  normalized gradient. This run for for loop in a number of context word and update the context word vec with grad .
    ### END YOUR CODE
   
    return cost, gradIn, gradOut #this gives the grad out and gradin alos the cost.

#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################
#this is a stocastic gradient dcent procedure. insaide of this it decides which model it should use Cbow or skipgram model
def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
    #print("\n ######### printing inside the SGD model  ###########")
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
   
#here the C is used in genarating random centerword and contexts
    
    #print "Printig tokens"
    #print tokens 
    
  
    #print  getRandomContext()
   
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N//2,:]  #take the first 5 as input and    5*3 matrix
    outputVectors = wordVectors[N//2:,:] #next 5 as the output vectors     5*3 matrix
    for i in range(batchsize):     #stocastic gradient dcent procedure
        C1 = random.randint(1,C)   #obtaining a rando number . between 1 and 5
        centerword, context = dataset.getRandomContext(C1)    #defining the center word and other sorrounding words . contexts words rely on the C1.  #this function is in the data_utiliti.
        #print context,centerword
        #print "Printing the center word in the window at each iteration"
        #print centerword 
        #print "surrounding words at each iteration"
        #print context
        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1
        #here this is the model we ae using to get the gradients . This can be input as a parameter. skipgram or CBOW. Take one itaration as a one window. if this is 50 it means there are 50 dynemic windwos. 

#out pu the total cost and grad update of the each window 
        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient)
#c is  the total cost in the window like wise gin is the total grads in the each window same with the gout. just a total
# 
       
        cost += c / batchsize / denom   #add cost as a fraction of the iterations 
        grad[:N//2, :] += gin / batchsize / denom   #updating the outside vectors.  
        grad[N//2:, :] += gout / batchsize / denom
        #print("\n==== End of this iteration    ====")
        #still no parameter update this actually normalize the grad and cost and return it.
    
    return cost, grad


