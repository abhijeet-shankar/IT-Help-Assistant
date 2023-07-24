from tkinter import *
from tkinter import ttk
import joblib
from tkinter import filedialog
import csv
import pandas as pd
import numpy as np
import nltk
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import string
import re
from nltk.stem import WordNetLemmatizer
from string import punctuation
from nltk.corpus import stopwords
from tkinter import ttk
import numpy as np
from numpy.linalg import norm

def sent_tokens_func(text):
  return nltk.sent_tokenize(text)

def word_tokens_func(text):
  return nltk.word_tokenize(text)  

def to_lower(text):
  if not isinstance(text,str):
    text = str(text)
  return text.lower()

def number_omit_func(text):
  output = ''.join(c for c in text if not c.isdigit())
  return output

def remove_punctuation(text):
  return ''.join(c for c in text if c not in punctuation) 

def stopword_remove_func(sentence):
  stop_words = stopwords.words('english')
  return ' '.join([w for w in nltk.word_tokenize(sentence) if not w in stop_words])

def lemmatize(text):
          wordnet_lemmatizer = WordNetLemmatizer()
          lemmatized_word = [wordnet_lemmatizer.lemmatize(word)for word in nltk.word_tokenize(text)]
          return " ".join(lemmatized_word)

def preprocess(text):
        lower_text = to_lower(text)
        sentence_tokens = sent_tokens_func(lower_text)
        word_list = []
        for each_sent in sentence_tokens:
            lemmatizzed_sent = lemmatize(each_sent)
            clean_text = number_omit_func(lemmatizzed_sent)
            clean_text = remove_punctuation(clean_text)
            clean_text = stopword_remove_func(clean_text)
            word_tokens = word_tokens_func(clean_text)
            for i in word_tokens:
                word_list.append(i)
        return " ".join(word_list)

def core():
    global pb
    global main,mainl
    global sentence_embeddings,model,preprocess,df
    pb['value']+=40
    mainl.config(text="Loading Dataset")
    main.update()
    df = pd.read_excel(r'df_withoutdup_final_origi1.xlsx')
    #df=pd.read_csv('Preprocessed_data.csv')
    pb['value']+=55
    mainl.config(text="Loading Transformer")
    main.update()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # sentences=df['preprocessed combined data'].values.tolist()
    # sentence_embeddings = np.array(model.encode(sentences))
    #joblib.dump(sentence_embeddings, 'embeddings.joblib')
    mainl.config(text="Opening App")
    pb['value']+=10
    main.update()

def task():
    core()
    main.destroy()

main = Tk()
main.config()
main.attributes('-topmost',True)
main.title("Loading ...Please Wait !")
mainl = Label(main, text="Loading Dataset",fg='White',bg='Black')
main.config(bg='Black')
mainl.grid(row=2,column=1,sticky="news",padx=10,pady=10)
pb = ttk.Progressbar(
    main,
    orient='horizontal',
    mode='determinate',
    length=280
)
pb.grid(row=1,column=1,sticky="news",padx=10,pady=10)
main.after(200, task)
main.mainloop()

def sol(*args):
    global model,df
    sentence_embeddings =joblib.load('embeddings.joblib')
    intext=str(e1.get())
    if intext=="":
        e2.config(text="No Solution -_-",bg='White',fg='black', wraplength=350,anchor="nw")
    else:
        intext=preprocess(intext)
        intext_embedding=np.transpose(np.array(model.encode([intext])))
        cosine=np.dot(sentence_embeddings,intext_embedding)/(norm(sentence_embeddings,axis=1)*norm(intext_embedding))
        index = np.where(cosine == np.amax(cosine))
        index=list(index[0])
        e2.config(text=df['Solution'][index[0]],bg='White',fg='black', wraplength=350,anchor="nw")

app=Tk()
app.title('IT Help Assistant')
app.configure(bg='Black')
app.resizable(0,0)
app.attributes('-topmost',True)
l1=Label(app,text='Enter Query -',fg='PowderBlue',bg='Black',font=('MS Serif',12,'bold')).grid(row=0,column=0)

e1=Entry(app)
e1.bind('<Return>',sol)
e1.grid(row=0,column=1,sticky='news',padx=10,pady=10)

l2=Label(app,text='Recommended Solution(s) -',fg='PowderBlue',bg='Black',font=('MS Serif',14,'bold'),wraplength=200).grid(row=1,column=0)
e2=Label(app,text='Solution will appear here (Press Enter after entering query in input box and wait for few seconds)',wraplength=200 ,height=20, width=50,bg='Black',fg='blue',font=('Arial',16,'bold'))

e2.grid(row=1,column=1,sticky='nw',padx=10,pady=10)

style=ttk.Style(app)
style.theme_use("clam")
style.configure("B.TButton", font=("MS Serif", 18,'bold'), foreground="black", background="Red")
style.map("B.TButton", foreground=[('active', 'black')], background=[('active', 'dark red')])

b1=ttk.Button(app,text='Exit',command=app.destroy,style='B.TButton')
b1.grid(row=2,column=0,pady=10,columnspan=2)

app.mainloop()

