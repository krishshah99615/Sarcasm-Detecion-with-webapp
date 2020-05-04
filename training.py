import tensorflow as tf
from tensorflow.keras.layers import Dense,Embedding,GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt





def data():
    d = list(pd.read_json('data/Sarcasm_Headlines_Dataset.json',lines=True)['headline'])
    d2 = list(pd.read_json('data/Sarcasm_Headlines_Dataset_v2.json',lines=True)['headline'])
    l=list(pd.read_json('data/Sarcasm_Headlines_Dataset.json',lines=True)['is_sarcastic'])
    l2=list(pd.read_json('data/Sarcasm_Headlines_Dataset_v2.json',lines=True)['is_sarcastic'])
    s = np.array(d)
    l=np.array(l)

    return s,l
def train(train_size,vocab_size,embedding_dim,max_len,oov_tok,epoch,sentences,labels):
    '''train_size=20000
    vocab_size=10000
    embedding_dim=16
    max_len=32
    oov_tok="<OOV>"
    '''
    
    train_sent=sentences[0:train_size]
    test_sent=sentences[train_size:]
    train_labels=labels[0:train_size]
    test_labels=labels[train_size:]
    tokenizer =Tokenizer(num_words=vocab_size,oov_token=oov_tok)
    tokenizer.fit_on_texts(train_sent)
    word_index = tokenizer.word_index
    r_word_index=dict(((k,v) for (v,k) in word_index.items()))
    
    
    f = open('tokenizer.pkl', 'wb') 
    pickle.dump(tokenizer, f)
    f.close()
    
    
    train_seq = tokenizer.texts_to_sequences(train_sent)
    test_seq = tokenizer.texts_to_sequences(test_sent)
    
    train_padded=pad_sequences(train_seq,maxlen=max_len,truncating='post',padding='post')
    test_padded=pad_sequences(test_seq,maxlen=max_len,truncating='post',padding='post')
    
    model = Sequential([
            Embedding(vocab_size,embedding_dim,input_length=max_len),
            GlobalAveragePooling1D(),
            Dense(24,activation='relu'),
            Dense(1,activation='sigmoid')
            ])
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    h= model.fit(train_padded,train_labels,epochs=epoch,validation_data=(test_padded,test_labels))
    model.save('model.h5')
    t={
            'history':h,
            'model':model,
            'tokenizer':tokenizer,
            'r_wi':r_word_index}
    return t
def plotting(h,s):
    plt.plot(h.history[s])
    plt.plot(h.history['val_'+s])
    plt.xlabel('Epochs')
    plt.ylabel('s')
    plt.legend([s,'val_'+s])
    plt.show()

def predict(s,t):
    s=[s]
    s_seq=t['tokenizer'].texts_to_sequences(s)
    s_seq_pad=pad_sequences(s_seq,truncating='post',padding='post',maxlen=32)
    ans=t['model'].predict(s_seq_pad)
    return ans
s,l = data()   
a = train(20000,1000,16,32,"<OOV>",30,s,l)
'''
plotting(a['history'],"accuracy")
plotting(a['history'],"loss")
predict("I really think this is amazing. honest.",a)


e = a['model'].layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)



import io
out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, 1000):
  word = a['r_wi'][word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()
'''