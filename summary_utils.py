import os
from spacy.lang.en import English

def split_sentences(raw_text):
    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    doc = nlp(raw_text)
    sentences = [sent.string.strip() for sent in doc.sents if len(sent.string.strip("\n")) > 0]
    return sentences

#creating a file for each sentence in the input document
def doc_sentences_extract(input_doc, doc_sent_dir, topic):
    with open(input_doc,'r') as f:
        doc = f.read()
    sentences = split_sentences(doc)#sent_tokenize(doc)

    if not os.path.exists(doc_sent_dir):
        os.makedirs(doc_sent_dir)
    for sent_idx, sentence in enumerate(sentences):
        html_path = os.path.join(doc_sent_dir, topic + str(sent_idx)+'.html')
        with open(html_path, 'w') as f:
            f.write(sentence)

    return {k:v for k,v in enumerate(sentences)}

#creating a directory for each sentence in the summary, and saving a sentence in a file in it
def summ_sentences_extract(input_summ, summ_sent_dir, ref_summary):
    with open(input_summ,'r') as f:
        summ = f.read()
    sentences = split_sentences(summ)#sent_tokenize(summ)
    for sent_idx, sentence in enumerate(sentences):
        sent_dir = os.path.join(summ_sent_dir, str(sent_idx))
        if not os.path.exists(sent_dir):
            os.makedirs(sent_dir)
        html_path = os.path.join(sent_dir, ref_summary + '.html')
        with open(html_path, 'w') as f:
            f.write(sentence)
    return {k:v for k,v in enumerate(sentences)}