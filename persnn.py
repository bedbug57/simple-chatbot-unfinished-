#activations
def relu(x, alpha=0.01):
    return [xi if xi > 0 else alpha * xi for xi in x]
def relu_der(x, alpha=0.01):
    return [1 if xi > 0 else alpha for xi in x]
#list operation support
def vector_sub(a, b):
    return [x-y for x, y in zip(a, b)]
#matrix operation support
def transpose(matrix):
    return list(map(list, zip(*matrix)))
def dot(matrix, vector):
    result=[]
    for row in matrix:
        s=0
        for x, y in zip(row, vector):
            s+=x*y
        result.append(s)
    return result

def layer(input_, w, b, in_size, out_size):
    out2=[]
    for i in range(out_size):
        out=0
        for j in range(in_size):
            out+=input_[j]*w[i][j]
        out+=b[i]
        out2.append(out)
    return out2
def forward(input_, w, b, in_size, out_size):
    l1_=layer(input_, w[0], b[0], in_size[0], out_size[0])
    l1=relu(l1_)
    l2_=layer(l1, w[1], b[1], in_size[1], out_size[1])
    l2=relu(l2_)
    l3_=layer(l2, w[2], b[2], in_size[2], out_size[2])
    l3=relu(l3_)
    l4_=layer(l3, w[3], b[3], in_size[3], out_size[3])
    l4=relu(l4_)
    l5_=layer(l4, w[4], b[4], in_size[4], out_size[4])
    l5=relu(l5_)
    l6=layer(l5, w[5], b[5], in_size[5], out_size[5])
    return l6, [l5, l4, l3, l2, l1], [l5_, l4_, l3_, l2_, l1_]
def backward(input_, out, y, w, b, rate, norml, real):
    error=vector_sub(out, y)
    delta6=error
    
    w6_T=transpose(w[5])
    delta5_part=dot(w6_T, delta6)
    delta5=[delta5_part[i]*relu_der(real[0][i]) for i in range(len(delta5_part))]

    w5_T=transpose(w[4])
    delta4_part=dot(w5_T, delta5)
    delta4=[delta4_part[i]*relu_der(real[1][i]) for i in range(len(delta4_part))]

    w4_T=transpose(w[3])
    delta3_part=dot(w4_T, delta4)
    delta3=[delta3_part[i]*relu_der(real[2][i]) for i in range(len(delta3_part))]


    w3_T=transpose(w[2])
    delta2_part=dot(w3_T, delta3)
    delta2=[delta2_part[i]*relu_der(real[3][i]) for i in range(len(delta2_part))]


    w2_T=transpose(w[1])
    delta1_part=dot(w2_T, delta2)
    delta1=[delta1_part[i]*relu_der(real[4][i]) for i in range(len(delta1_part))]

    for i in range(len(w[5])):
        for j in range(len(w[5][0])):
            w[5][i][j]-=rate*delta6[i]*norml[0][j]
    for i in range(len(b[5])):
        b[5][i]-=rate*delta6[i]

    for i in range(len(w[4])):
        for j in range(len(w[4][0])):
            w[4][i][j]-=rate*delta5[i]*norml[1][j]
    for i in range(len(b[4])):
        b[4][i]-=rate*delta5[i]
    
    for i in range(len(w[3])):
        for j in range(len(w[3][0])):
            w[3][i][j]-=rate*delta4[i]*norml[2][j]
    for i in range(len(b[3])):
        b[3][i]-=rate*delta4[i]

    for i in range(len(w[2])):
        for j in range(len(w[2][0])):
            w[2][i][j]-=rate*delta3[i]*norml[3][j]
    for i in range(len(b[2])):
        b[2][i]-=rate*delta3[i]

    for i in range(len(w[1])):
        for j in range(len(w[1][0])):
            w[1][i][j]-=rate*delta2[i]*norml[4][j]
    for i in range(len(b[1])):
        b[1][i]-=rate*delta2[i]

    for i in range(len(w[0])):
        for j in range(len(w[0][0])):
            w[0][i][j]-=rate*delta1[i]*input_[j]
    for i in range(len(b[0])):
        b[0][i]-=rate*delta1[i]

    return w, b

