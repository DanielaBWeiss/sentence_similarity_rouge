import os
import pandas as pd
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS


TEMP_SUMM_SENTS = "temp_summ_sents"

def split_sentences(raw_text):
    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    doc = nlp(raw_text)
    sentences = [sent.string.strip() for sent in doc.sents if len(sent.string.strip("\n")) > 0]
    return sentences

#creating a file for each sentence in the input document
'''
Creating a file for each sentence in an input document.
If multi_doc option is passed, then the input_doc represents a 'topic' directory where documents reside.
Else, the input_doc represents the document file.
'''
def doc_sentences_extract(input_doc, doc_sent_dir, topic, multi_doc=False):

    sentences = []
    if os.path.exists(input_doc + "/" + "document_sentences.txt"):
        print("Loading document sentences from 'document_sentences.txt'")
        with open(input_doc + "/" + "document_sentences.txt", 'r') as f:
            sentences = f.readlines()
        df = pd.read_csv(input_doc +"/" +"doc_num_sents.csv")
        doc_sents = df.set_index('0')['1'].to_dict()
    else:
        doc_sents = {}
        print("Writing document sentences to files")
        if multi_doc:
            i = 0
            for filename in os.listdir(input_doc):
                if os.path.isdir(input_doc +"/" + filename): continue
                with open(input_doc +"/" + filename,'r') as f:
                    doc = f.read()
                sents = split_sentences(doc)
                doc_sents[i] = len(sents) + len(sentences)
                i += 1
                sentences.extend(sents)
            df = pd.DataFrame.from_dict(doc_sents.items())
            df.to_csv(input_doc +"/" +"doc_num_sents.csv")
        else:
            with open(input_doc, 'r') as f:
                doc = f.read()
            sentences = split_sentences(doc)
            doc_sents[0] = len(sentences)
            df = pd.DataFrame.from_dict(doc_sents.items())
            df.to_csv(input_doc + "/" + "doc_num_sents.csv")
        with open(input_doc + "/" + "document_sentences.txt", 'w') as f:
            for sent in sentences:
                if not sent:
                    f.write("\n")
                f.write(sent + "\n")


        if not os.path.exists(doc_sent_dir):
            os.makedirs(doc_sent_dir)
        for sent_idx, sentence in enumerate(sentences):
            html_path = os.path.join(doc_sent_dir, topic + str(sent_idx)+'.html')
            if os.path.exists(html_path): continue
            with open(html_path, 'w') as f:
                f.write(sentence)

    return {k:v for k,v in enumerate(sentences)}, doc_sents

#creating a directory for each sentence in the summary, and saving a sentence in a file in it
def summ_sentences_extract(input_summ, summ_sent_dir, ref_summary):
    """

    Args:
        input_summ: Directory for all summaries of the same topic
        summ_sent_dir: Directory for where to save summary sentences
        ref_summary: Name for summary sentence files

    Returns:
        Extracted sentences
    """
    sentences = []
    if os.path.exists(input_summ + "/text_sentences/" + "summary_sentences.txt"):
        print("Loading summary sentences from 'summary_sentences.txt'")
        with open(input_summ+ "/text_sentences/"+ "summary_sentences.txt", 'r') as f:
            sentences = f.readlines()
        df = pd.read_csv(input_summ + "/text_sentences/" + "summ_num_sents.csv")
        summ_sents = df.set_index('0')['1'].to_dict()

    else:
        summ_sents = {}
        print("Writing summary sentences to files")
        i = 0
        for summ in os.listdir(input_summ):
            if ".html" in summ:
                with open(input_summ +"/" + summ,'r') as f:
                    summ = f.read()
                sents = split_sentences(summ)
                summ_sents[i] = len(sents) + len(sentences)
                i += 1
                sentences.extend(sents)
        df = pd.DataFrame.from_dict(summ_sents.items())

        if not os.path.exists(input_summ + "/text_sentences"):
            os.makedirs(input_summ + "/text_sentences")
        df.to_csv(input_summ + "/text_sentences/" + "summ_num_sents.csv")
        with open(input_summ + "/text_sentences/"+ "summary_sentences.txt", 'w') as f:
            for sent in sentences:
                if not sent:
                    f.write("\n")
                f.write(sent + "\n")

        for sent_idx, sentence in enumerate(sentences):
            sent_dir = os.path.join(summ_sent_dir, str(sent_idx))
            if not os.path.exists(sent_dir):
                os.makedirs(sent_dir)
            html_path = os.path.join(sent_dir, ref_summary + '.html')
            with open(html_path, 'w') as f:
                f.write(sentence)
    return {k:v for k,v in enumerate(sentences)}, summ_sents


def delete_mac_files(directories):
    for dir in directories:
        # Deleting Mac files
        if os.path.exists(os.path.abspath(dir + "/.DS_STORE")): os.remove(
            os.path.abspath(dir + "/.DS_STORE"))


def find_sent_index(file_dict, ind):
    '''

    Args:
        file_dict: dictionary containing integer keys representing document numbers, and values for the range of sentence indicies
        ind: the current sentence (can be document or summary) being looked at

    Returns: a string key: (document number):(sentence number)

    '''
    for k,v in file_dict.items():
        if int(ind) < v:
            if k == 0:
                return str(k) + "/" + str(ind)
            else:
                sent_ind = int(ind) - file_dict[k-1]
                return str(k) + "/" + str(sent_ind)


def write_temp_summ_files(summ_dir, summ_sent_dir, sentence, ref_summary, overlap_index):
    main_tmp_sents = os.path.join(os.path.dirname(str(summ_dir)),TEMP_SUMM_SENTS)
    curr_summ_sent = os.path.join(os.path.dirname(str(summ_dir)), TEMP_SUMM_SENTS , str(summ_sent_dir))
    curr_overlap =  os.path.join(os.path.dirname(str(summ_dir)), TEMP_SUMM_SENTS ,str(summ_sent_dir) ,str(overlap_index))
    if not os.path.exists(main_tmp_sents):
        os.makedirs(main_tmp_sents)
    if not os.path.exists(curr_summ_sent):
        os.makedirs(curr_summ_sent)
    if not os.path.exists(curr_overlap):
        os.makedirs(curr_overlap)

    html_path = os.path.join(curr_overlap, ref_summary + '.html')
    with open(html_path, 'w') as f:
        f.write(sentence)
    return curr_overlap

def remove_overlap_words(summ_sent, best_sentence):
    nlp = English()
    sum = nlp(summ_sent.strip("\n"))
    doc = nlp(best_sentence.strip("\n"))
    overlap = get_overlap(sum, doc)
    new_summary_sent = ""
    for i, token in enumerate(sum):
        if token.text.lower() in overlap: continue
        new_summary_sent += token.text
        if i < len(sum) - 1:
            new_summary_sent += " "
    return new_summary_sent


def overlap_threshold(ref_sentence, sys_sentence):
    nlp = English()
    sum = nlp(ref_sentence.strip("\n"))
    doc = nlp(sys_sentence.strip("\n"))
    overlap = get_overlap(sum, doc)
    if len(overlap) >=2: return True
    else: return False

'''
TODO: option to stem all words and take overlap
'''
def get_overlap(ref, sys):
    sum_tokens_set = {word.text.lower() for word in ref if word.text not in STOP_WORDS and not word.is_punct}
    doc_tokens_set = {word.text.lower() for word in sys if word.text not in STOP_WORDS and not word.is_punct}
    overlap = sum_tokens_set.intersection(doc_tokens_set)
    return overlap