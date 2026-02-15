from persnn import forward, backward
from wordtovec import build_vocab, generate_training_pairs, initialize_weights, train_skipgram, get_embedding, softmax
from termcolor import colored
from random import uniform
import pickle
import os
from math import sqrt
logo=colored('''
        ###################################################################################
            ,@#@#+,     ##                       ..+#+..       .,####,.     ..,+##@+. 
    ####   .@#"   "'"   ,#@           .##+,.  .   @"    '"#,   ##'     ##   ###"         ####
   +       @#           ,#@'       ,#@"   '"#@+   @.      ##.  ##      ##   '"@@#,           +
    ####   @#   ,,..    +##'       #@       #@+   @+      "##  @+      ##       "@++.    ####
   +       '##  "#@##"  +#@"       ##       @#+   @+      +#"  @+      ##  ..      @@#+      +
    ####    "+##..+'"   .###@@##+'  "###@@##""#+  @"+,...##    "#####@#"    #+@...+##''  ####
                """"      ""''"''       "''"   '    '"''"'        "'"'""       """''"''
        ###################################################################################
    ''', 'yellow')
print(logo)

is_prunwb_loaded=False
is_uinput_loaded=False
inset=[32, 256, 256, 256, 256, 64]
outset=[256, 256, 256, 256, 64, 32]
errors=[]
try:
    with open('data/prunwb.pkl', 'rb') as file:
        data=pickle.load(file)
        pw=data['w']
        pb=data['b']
    is_prunwb_loaded=True
except:
    errors.append("prunwb")
if len(errors)>0:
    print(f"you can't even read {errors} correctly, bitch")

def get_zero_embedding(dim=32):
    return [0.0]*dim
def cosine_similarity(a, b):
    dot=sum(x*y for x, y in zip(a, b))
    na=sqrt(sum(x*x for x in a))
    nb=sqrt(sum(x*x for x in b))
    if na==0 or nb==0:
        return -1.0
    return dot/(na*nb)
def find_closest_word(vec, W_input, index_to_word):
    best_sim=-2.0
    best_word=None
    for idx, emb in enumerate(W_input):
        sim = cosine_similarity(vec, emb)
        if sim>best_sim:
            best_sim=sim
            best_word=index_to_word[idx]
    return best_word, best_sim
def mean_pool(embeddings):
    if not embeddings: return [0.0]*32
    return [sum(col)/len(embeddings) for col in zip(*embeddings)]

run=True
run_train=False
run_norm=False
while run:
    options=str(input("options> "))
    if options=="1":
        exit()
    elif options=="help":
        print(colored("1.exit\n2.load input\n3.create wb for prun\n4.show data\n5.run norm\n6.run train\n7.delete prun wb\n8.delete input\n9.clear", "grey"))
    elif options=="2":
        uinput=input("uinput> ")
        tkinput=uinput.split()
        is_uinput_loaded=True
    elif options=="3":
        pw=[]
        pb=[]
        for i in range(6):
            e=[]
            for j in range(int(outset[i])):
                c=[]
                for k in range(int(inset[i])):
                    a=uniform(-0.1, 0.1)
                    d=round(a, 5)
                    c.append(d)
                e.append(c)
            pw.append(e)
        for i in range(6):
            d=[]
            for j in range(int(outset[i])):
                a=uniform(-0.1, 0.1)
                c=round(a, 5)
                d.append(c)
            pb.append(d)
        with open('data/prunwb.pkl', 'wb') as file:
            pickle.dump({'w': pw, 'b': pb}, file)
        is_prunwb_loaded=True
    elif options=="4":
        try:
            print(f"prun:\n{pw[0][0][:8]}\n{pb[0][:8]}")
        except:
            print(colored("you lack prun data", "red"))
        try:
            tkinput[0]
            print(f"tkinput:\n{tkinput}")
        except:
            print(colored("you lack input", "red"))
    elif options=="5":
        if is_prunwb_loaded and is_uinput_loaded:
            run_norm=True
            run=False
        else:
            print(colored("go load prunwb/uinput dumbass", "red"))
    elif options=="6":
        if is_prunwb_loaded:
            run_train=True
            run=False
        else:
            print(colored("go load prunwb dumbass", "red"))
    elif options=="7":
        if os.path.exists('data/prunwb.pkl'):
            os.remove('data/prunwb.pkl')
            pw=[]
            pb=[]
            is_prunwb_loaded=False
    elif options=="8":
        uinput=""
        tkinput=[]
        is_uinput_loaded=False
    elif options=="9":
        os.system('cls' if os.name == 'nt' else 'clear')
        print(logo)

if run_train:
    epoch=int(input("epoch> "))
    for i in range(epoch):
        _, _, files=next(os.walk("data/inps"))
        file_count=(len(files))
        for j in range(file_count):
            with open(f'data/inps/inp{j}.txt', 'r') as file:
                inp=file.read().splitlines()
                inp1_str, inp2_str=inp
                inp1=inp1_str.split()
                inp2=inp2_str.split()
                max_len=max(len(inp1), len(inp2))
                while len(inp1)<max_len:
                    inp1.append('PAD')
                while len(inp2)<max_len:
                    inp2.append('PAD')
            word_list1, word_to_index1, index_to_word1, vocab_size1=build_vocab(inp1)
            training_pairs1=generate_training_pairs([inp1], 4)
            wtvw11, wtvw12=initialize_weights(vocab_size1, 32)
            word_list2, word_to_index2, index_to_word2, vocab_size2=build_vocab(inp2)
            training_pairs2=generate_training_pairs([inp2], 4)
            wtvw21, wtvw22=initialize_weights(vocab_size2, 32)
            wtvw11, wtvw12=train_skipgram(training_pairs1, word_to_index1, vocab_size1, wtvw11, wtvw12, 32, 0.001, 1000)
            wtvw21, wtvw22=train_skipgram(training_pairs2, word_to_index2, vocab_size2, wtvw21, wtvw22, 32, 0.001, 1000)

            wtv_inp_np=[]
            for k in range(len(inp1)):
                if inp1[k]=='PAD':
                    wtv_inp_np.append(get_zero_embedding())
                else:
                    wtv_inp_np.append(get_embedding(inp1[k], word_to_index1, wtvw11))
            wtv_out_np=[]
            for k in range(len(inp2)):
                if inp2[k]=='PAD':
                    wtv_out_np.append(get_zero_embedding())
                else:
                    wtv_out_np.append(get_embedding(inp2[k], word_to_index2, wtvw21))
            for k in range(len(inp1)):
                x_k=wtv_inp_np[k]
                y_k=wtv_out_np[k]
                out, norml, real=forward(x_k, pw, pb, inset, outset)
                pw, pb = backward(x_k, out, y_k, pw, pb, 0.0001, norml, real)

            wtv_inp_np=[get_embedding(inp1[k], word_to_index1, wtvw11) for k in range(len(inp1))]
            wtv_out_np=[get_embedding(inp2[k], word_to_index2, wtvw21) for k in range(len(inp2))]
            wtv_inp_p=mean_pool(wtv_inp_np)
            wtv_out_p=mean_pool(wtv_out_np)
            out, norml, real=forward(wtv_inp_p, pw, pb, inset, outset)
            pw, pb=backward(wtv_inp_p, out, wtv_out_p, pw, pb, 0.0001, norml, real)

            with open('data/prunwb.pkl', 'wb') as file:
                pickle.dump({'w': pw, 'b': pb}, file)
        if i%10==0:
            print(i)

if run_norm:
    word_list1, word_to_index1, index_to_word1, vocab_size1=build_vocab(tkinput)
    training_pairs1=generate_training_pairs([tkinput], 4)
    wtvw11, wtvw12=initialize_weights(vocab_size1, 32)
    wtvw11, wtvw12=train_skipgram(training_pairs1, word_to_index1, vocab_size1, wtvw11, wtvw12, 32, 0.001, 1000)

    outs=[]
    for k in range(len(tkinput)):
        word_emb=get_embedding(tkinput[k], word_to_index1, wtvw11)
        out, norml, real=forward(word_emb, pw, pb, inset, outset)
        outs.append(out)

    word_embs=[get_embedding(tkinput[k], word_to_index1, wtvw11) for k in range(len(tkinput))]
    wtv_inp=mean_pool(word_embs) 
    out, norml, real=forward(wtv_inp, pw, pb, inset, outset)
    print(out)

    decoded=[]
    decoded_sim=[]
    for out_vec in outs:
        word, sim=find_closest_word(out_vec, wtvw11, index_to_word1)
        decoded.append(word)
        decoded_sim.append(f"sim={sim:.3f}")

    print(decoded, decoded_sim)
