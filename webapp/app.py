import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask,request,render_template

app=Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
  if request.method == 'POST':
    s = str(request.form['s'])
    a = predict(s)
    return render_template('index.html',a=a)
  return render_template('index.html',a=None)

def predict(s):
    
    model = tf.keras.models.load_model('model.h5')
    f = open("tokenizer.pkl","rb")
    tokenizer = pickle.load(f)
    s=[s]
    s_seq=tokenizer.texts_to_sequences(s)
    s_seq_pad=pad_sequences(s_seq,truncating='post',padding='post',maxlen=32)
    ans=model.predict(s_seq_pad)[0][0]
    if ans>=0.5:
      return "Sarcastic"
    else:
      return "Not Sarcastic"
    
if __name__ == '__main__':
  app.run(debug=True)