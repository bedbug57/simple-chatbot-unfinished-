def tokenize(text, sout):
    text=text.lower()
    for ch in ",.!?;:\"()":
        text=text.replace(ch, " ")
    return text.split() if sout else text
def build_bigram_model(words):
    model={}
    for i in range(len(words)-1):
        w1=words[i]
        w2=words[i + 1]
        if w1 not in model:
            model[w1]={}
        if w2 not in model[w1]:
            model[w1][w2]=0
        model[w1][w2]+=1
    return model
def predict_next_word(current, model):
    current=current.lower()
    if current not in model:
        return None
    candidates=model[current]
    next_word=max(candidates, key=candidates.get)
    return next_word