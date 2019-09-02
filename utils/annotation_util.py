import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
#Topic = D3001
#Summaries - A & D

docs = ["APW19981016.0240", "APW19981022.0269", "APW19981026.0220",
        "APW19981027.0491", "APW19981031.0167", "APW19981113.0251",
        "APW19981116.0205", "APW19981118.0276", "APW19981120.0274",
        "APW19981124.0267"]


def split_sentences(doc):
    sentences = sent_tokenize(doc)
    for j in reversed(range(len(sentences))):
        sent = sentences[j]
        sentences[j] = sent.strip()
        if sent == '':
            sentences.pop(j)
    return sentences

def annot(doc, save_summ):
    for summ in os.listdir("doc/"):
        if doc in summ:
            with open("doc" +"/" + summ,'r') as f:
                summ = f.read()
            sents = split_sentences(summ)



    aligns = {}
    for id, i in enumerate(sents):
        print(i)
        print()
        align = input()
        aligns[id] = align


    df = pd.DataFrame(aligns.items())
    df.columns = ["sents-index", "alignment"]
    df["doc-sents"] = sents
    df.to_csv("doc"+save_summ+"-summD-alignment.csv", index=False)



