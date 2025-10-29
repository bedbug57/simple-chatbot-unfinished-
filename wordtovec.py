import random
import math

def softmax(x): #turn a list into probality(kinda weird but works similiar to ReLU)
    e_x=[math.exp(i) for i in x]
    sum_e=sum(e_x)
    return [j/sum_e for j in e_x]

def dot(a, b): #𝚺a[i]*b[i]
    return sum([a_i*b_i for a_i, b_i in zip(a, b)])

def vector_add(a, b):
    return [ai+bi for ai, bi in zip(a, b)]

def vector_sub(a, b):
    return [ai-bi for ai, bi in zip(a, b)]

def scalar_mul(vec, scalar):
    return [v*scalar for v in vec]

def build_vocab(sentence):
    #bulid vocab
    word_set=set()
    for word in sentence:
        word_set.add(word)
    word_list=list(word_set)
    #words go into a map and a matching number comes out(map={word[0]: number[0], word[1]:number[1]...}), map is based on word_list, it return an INDEX that will be used with w1(w1[word_to_index])
    word_to_index={word: i for i, word in enumerate(word_list) } #words->neural net
    index_to_word={i: word for i, word in enumerate(word_list) } #neural net->words
    vocab_size=len(word_list)
    return word_list, word_to_index, index_to_word, vocab_size

def generate_training_pairs(sentences, window_size):
    training_pairs=[]
    for sentence in sentences:
        for i, word in enumerate(sentence):
            for w in range(-window_size, window_size+1):
                context_index=i+w#word from a list of words+window(neightbour words for meaning)
                if w!=0 and 0<=context_index<len(sentence):#border so index doesnt go out of range
                    training_pairs.append((word, sentence[context_index]))#word=the; training_pairs=[["the", "quick"], ["the", "brown"]]
    return training_pairs

def initialize_weights(vocab_size, embedding_size):
    w_input=[[random.uniform(-1, 1) for _ in range(embedding_size)] for _ in range(vocab_size)]#learnable list of numbers that symbolise a word based on index given to it and then learning this word for each item in list, represents word itself
    w_output=[[random.uniform(-1, 1) for _ in range(vocab_size)] for _ in range(embedding_size)]#learnable list thats used in nn that calculates meaning of given word based on guessing what neightbour words it has and then comparing it to the actual neightbour words
    return w_input, w_output

def train_skipgram(training_pairs, word_to_index, vocab_size, W_input, W_output, embedding_size, learning_rate, epochs):
    for epoch in range(epochs):
        for center_word, context_word in training_pairs:
            #convert number that represents a word back into it's index
            x_idx=word_to_index[center_word]
            y_idx=word_to_index[context_word]

            h=W_input[x_idx] #embedding vector for center word

            u=[dot(h, [W_output[d][j] for d in range(embedding_size)]) for j in range(vocab_size)] #output scores for each word
            y_pred=softmax(u) #make probalities from scores

            y_true=[0]*vocab_size
            y_true[y_idx]=1 #what we want to come out(used for training), for example: y_true=[0, 0, 1, 0], output=[0.2, 0.01, 0.78, 0.01](after softmax so it adds up to 1), each output index goes closer to the y_true index
            e=[y_predj-y_truej for y_predj, y_truej in zip(y_pred, y_true)] #calculate error

            # Update w_output
            for j in range(vocab_size):
                for d in range(embedding_size):
                    W_output[d][j]-=learning_rate*e[j]*h[d]

            # Update w_input
            grad=[0.0 for _ in range(embedding_size)]
            for d in range(embedding_size):
                grad[d]=0.0
                for j in range(vocab_size):
                    grad[d]+=e[j]*W_output[d][j]
            for d in range(embedding_size):
                W_input[x_idx][d]-=learning_rate*grad[d]
    return W_input, W_output

def get_embedding(word, word_to_index, W_input): #get embedings for 1word from corpus
    idx=word_to_index[word]
    return W_input[idx]

