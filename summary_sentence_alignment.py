import os
import argparse
from nltk.tokenize import sent_tokenize
import calculateRouge
import numpy as np
import time
import pandas as pd

from utils import *
from rouge_sentence_similarity import RougeSentenceSimilarity


def remove_and_calculate_rouge(summ_sent, summ_dir, best_sentence,  stop_words, overlap_index, RS):
    new_sentence = remove_overlap_words(summ_sent, best_sentence)
    new_summ_dir_path = write_temp_summ_files(RS.summ_sent_dir, summ_dir, new_sentence, RS.summ_topic, overlap_index)

    all_data = RS.get_rouge_scores(new_summ_dir_path, calculateRouge.COMPARE_SAME_LEN, stop_words, override_dir=True)

    _, best_sentence = RS.get_top_sentences(all_data, new_sentence, summ_dir, save=False, previous_sentence=best_sentence)

    return new_sentence, best_sentence



if __name__ == "__main__":
    '''
    At the moment this dataset alignment between documents and summaries only works for the same topic, where all documents
    reside in the same folder, and all summaries in the same folder.
    All files must be in text DUC format, but have to be added an ending of ".html" (That is the used Rouge library requires it).
    '''
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_doc", default=None, type=str, required=True) #directory for all documents of the same topic
    parser.add_argument("--input_summ", default=None, type=str, required=True) #directory for all summaries for the same toipc
    parser.add_argument("--doc_sent_dir",  default=None, type=str, required=True) #desired directory to save document sentences
    parser.add_argument("--summ_sent_dir", default=None, type=str, required=True) #desired directory to save summary sentences
    parser.add_argument("--multi-doc", default=False, action='store_true')
    parser.add_argument("--topic", default='D30001.M.100.T.', type=str)
    parser.add_argument("--ref_summ", default='D30001.M.100.T.D', type=str)
    parser.add_argument("--remove_stopwords", action='store_true')
    parser.add_argument("--metric", default="f1", type=str)


    args = parser.parse_args()

    stop_words = calculateRouge.REMOVE_STOP_WORDS if args.remove_stopwords else calculateRouge.LEAVE_STOP_WORDS

    #split doucments into sentences and write rouge format files
    RS = RougeSentenceSimilarity(args.ref_summ, args.topic, args.input_doc, args.input_summ, args.doc_sent_dir, args.summ_sent_dir, args.metric)
    RS.write_and_transform_documents(args.multi_doc)


    print("For every summary sentence out of {}, calculating Rouge similarity to a total of {} sentences".format(len(RS.summ_sents), len(RS.doc_sents)))

    '''
    For each summary sentence, we calculate ROUGE scores of all document sentences and save to csv, as well as
    the top three doc sentences for each summary sentences.
    '''
    summary_rouge_sim_indicies = []
    alignment_dataset = []
    for summ_sent_idx, summ_dir in enumerate(os.listdir(RS.summ_sent_dir)):
        summ_doc_ind = find_sent_index(RS.summ_sent_indicies, summ_dir)
        startTime = time.time()
        print("summary sent # ", summ_sent_idx)

        all_data = RS.get_rouge_scores(summ_dir, calculateRouge.COMPARE_SAME_LEN, stop_words)

        best_triple, best_sentence = RS.get_top_sentences(all_data, RS.summ_sents[int(summ_dir)], summ_dir)
        summary_rouge_sim_indicies.append(best_triple)

        if overlap_threshold(RS.summ_sents[int(summ_dir)], best_sentence[1]):
            aligned_tuple = [summ_doc_ind+":"+summ_dir, RS.summ_sents[int(summ_dir)], str(best_sentence[0])+":"+best_sentence[1], best_sentence[2]]
            current_summary_sent = RS.summ_sents[int(summ_dir)]
        else:
            alignment_dataset.append((summ_dir,RS.summ_sents[int(summ_dir)], None,None, None, None))
            continue

        for i in range(2):
            current_summary_sent, best_sentence = remove_and_calculate_rouge(current_summary_sent, summ_dir,best_sentence[1],  stop_words, i, RS)
            if overlap_threshold(current_summary_sent, best_sentence[1]):
                aligned_tuple.extend([str(best_sentence[0])+":"+best_sentence[1], best_sentence[2]])
            else:
                aligned_tuple.extend([None,None])
        aligned_tuple = tuple(aligned_tuple)
        alignment_dataset.append(aligned_tuple)


        #rouge_vec = extract_rouge(allData, systemNames, summaryLengths)
        #rouge_mat[:, summ_sent_idx] = rouge_vec
        curTime = time.time()
        print('Current input done! Elapsed time: {} seconds!'.format(curTime - startTime))
        print('')

    #np.savetxt("rouge_matrix.csv", rouge_mat, delimiter=",")
    sim_df = pd.DataFrame(summary_rouge_sim_indicies, columns=["summary_index","summary_sent", "best-avg-rouge-01","best-avg-rouge-02", "best-avg-rouge-03"])
    sim_df= sim_df.sort_values(by=["summary_index"])
    sim_df.to_csv("rouge_similarities/summaries_and_best_doc_sents.csv", index=False)
    align_df = pd.DataFrame(alignment_dataset,
                          columns=["summary_index", "summary_sent", "01-sentence", "01-avgF1_rouge",
                                   "02-overlap-sentence", "02-overlap-avgF1_rouge", "03-overlap-sentence", "03-overlap-avgF1_rouge"])
    align_df = align_df.sort_values(by=["summary_index"])
    align_df.to_csv("rouge_similarities/alignment_dataset_"+RS.doc_topic+".csv", index=False)

    '''
    alignment dictionary between summary sentences and doc sentences based on 
    overlapped words threshold of 2 and average rouge F1 scores.
    '''



    print('---- DONE WITH ALL INPUTS')


