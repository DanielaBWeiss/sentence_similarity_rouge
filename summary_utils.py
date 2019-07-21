import os
from spacy.lang.en import English

def split_sentences(raw_text):
    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    doc = nlp(raw_text)
    sentences = [sent.string.strip() for sent in doc.sents if len(sent.string.strip("\n")) > 0]
    return sentences

#creating a file for each sentence in the input document
'''
Creating a filef or each sentence in an input document.
If multi_doc option is passed, then the input_doc represents a 'topic' directory where documents reside.
Else, the input_doc represents the document file.
'''
def doc_sentences_extract(input_doc, doc_sent_dir, topic, multi_doc=False):

    if multi_doc:
        sentences = []
        for filename in os.listdir(input_doc):
            if os.path.isdir(input_doc +"/" + filename):
            with open(input_doc +"/" + filename,'r') as f:
                doc = f.read()
            sentences.extend(split_sentences(doc))
    else:
        with open(input_doc, 'r') as f:
            doc = f.read()
        sentences = split_sentences(doc)

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


def delete_mac_files(directories):
    for dir in directories:
        # Deleting Mac files
        if os.path.exists(os.path.abspath(dir + "/.DS_STORE")): os.remove(
            os.path.abspath(dir + "/.DS_STORE"))

