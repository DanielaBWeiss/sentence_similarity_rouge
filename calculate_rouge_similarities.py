import os
import argparse
from nltk.tokenize import sent_tokenize
import calculateRouge
import numpy as np
import time
import pandas as pd

from summary_utils import *


def extract_rouge(analyzedData, systemNames, summaryLengths):
    '''
    Outputs the analyzedData to a CSV file with the format:
    ROUGE_type,<summLen1>_r,<summLen2>_r,<summLenK>_r,<summLen1>_p,<summLen2>_p,<summLenK>_p,<summLen1>_f,<summLen2>_f,<summLenK>_f
    where each line is a rougeType, and systems are divided into sections.
    '''
    # with open(outputFilepath, 'w') as outF:
    #     # header line
    #     # example: ROUGE_type,050_r,100_r,200_r,400_r,050_p,100_p,200_p,400_p,050_f,100_f,200_f,400_f
    #     firstLineParts = ['ROUGE_type']
    #     firstLineParts.extend(
    #         ['{}_{}'.format(summLen, measure_type) for measure_type in ['r', 'p', 'f'] for summLen in summaryLengths])
    #     firstLine = ','.join(firstLineParts)
    #     outF.write(firstLine + '\n\n')
    #
    #     # the csv is divided into section for each system:
    #     for sysName in systemNames:
    #         outF.write(sysName + '\n')

    rouge_vec = np.zeros((len(systemNames)))
    for sysName in systemNames:
        if sysName in analyzedData:
            summLen = summaryLengths[0]  # we should get here only 1 summary length

            # the rest of the line is for the columns <sys_len>.<r/p/f>, if there's no value, then 0:
            mean_rouge = np.mean([analyzedData[sysName][summLen][rougeType]['recall'] \
                                      if analyzedData[sysName][summLen][rougeType]['recall'] != -1 \
                                      else 0 \
                                  for rougeType in ['R1', 'R2', 'RL']])

            rouge_vec[int(sysName)] = mean_rouge

    return rouge_vec


if __name__ == "__main__":
    '''
    At the moment this only works for a single document and a single reference summary
    
    '''
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_doc", default=None, type=str, required=True)
    parser.add_argument("--input_summ", default=None, type=str, required=True)
    parser.add_argument("--doc_sent_dir", default=r'C:\Users\user\Documents\Phd\rouge\SummEval_referenceSubsets\code_score_extraction\doc_sentences', type=str)
    parser.add_argument("--summ_sent_dir", default=r'C:\Users\user\Documents\Phd\rouge\SummEval_referenceSubsets\code_score_extraction\summ_sentences', type=str)
    parser.add_argument("--topic", default='D30001.M.100.T.', type=str)
    parser.add_argument("--ref_summ", default='D30001.M.100.T.A', type=str)
    # parser.add_argument("--bert_model", default=None, type=str, required=True,
    #                         help="Bert pre-trained model selected in the list: bert-base-uncased, "
    #                              "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    #
    #     ## Other parameters
    # parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    # parser.add_argument("--layers", default="-1,-2,-3,-4", type=str)
    # parser.add_argument("--max_seq_length", default=128, type=int,
    #                         help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
    #                             "than this will be truncated, and sequences shorter than this will be padded.")
    # parser.add_argument("--batch_size", default=32, type=int, help="Batch size for predictions.")
    # parser.add_argument("--local_rank",
    #                         type=int,
    #                         default=-1,
    #                         help = "local_rank for distributed training on gpus")
    # parser.add_argument("--no_cuda",
    #                         action='store_true',
    #                         help="Whether not to use CUDA when available")

    args = parser.parse_args()

    doc_topic = args.topic
    summ_topic = args.ref_summ

    #Sentence Splitting the document and reference summary
    doc_sent = doc_sentences_extract(args.input_doc, args.doc_sent_dir, doc_topic)
    summ_sent = summ_sentences_extract(args.input_summ, args.summ_sent_dir, summ_topic)

    #creating a rouge matrix where each sentence is compared to every other sentence
    rouge_mat = np.zeros((len(doc_sent),len(summ_sent)))

    #Deleting Mac files
    if os.path.exists(os.path.abspath(args.doc_sent_dir + "/.DS_STORE")): os.remove(os.path.abspath(args.doc_sent_dir+ "/.DS_STORE"))
    if os.path.exists(os.path.abspath(args.summ_sent_dir + "/.DS_STORE")): os.remove(os.path.abspath(args.summ_sent_dir + "/.DS_STORE"))


    '''
    For each summary sentence, we calculate ROUGE scores of all document sentences and save single output to csv, as well as
    a Rouge matrix which contains all doc sentences, compared to all ref summary sentence.
    '''
    summary_rouge_sim_indicies = []

    for summ_sent_idx, summ_dir in enumerate(os.listdir(args.summ_sent_dir)):
        print("summary sent , ", summ_dir)

        INPUTS = [(calculateRouge.COMPARE_SAME_LEN, os.path.join(args.summ_sent_dir,summ_dir),args.doc_sent_dir, None, None, calculateRouge.LEAVE_STOP_WORDS)]
        print("Inputs ", INPUTS)
        startTime = time.time()
        # Go over each input:
        compareType, refFolder, sysFolder, outputPath, ducVersion, stopWordsRemoval = INPUTS[0]

        print('---- NEXT INPUT')
        # get the different options for comparison, each system sum is put into comparison with all ref summaries, and by summary length
        #Example output for a single summary document split sentences, to a ref summary:
        # (Topic ID)-['D30001'] (Split sents) - ['0', '1', '10', '11', '12', '13', '14', '15', '16', '2', '3', '4', '5', '6', '7', '8', '9']
        # (Summ length) - ['100']
        taskNames, systemNames, summaryLengths = calculateRouge.getComparisonOptions(sysFolder, refFolder)

        # get ROUGE scores:
        allData = calculateRouge.runRougeCombinations(compareType, sysFolder, refFolder, systemNames, summaryLengths,
                                                      ducVersion, stopWordsRemoval)
        sents_list = []
        for k,v in allData.items():
            for i in summaryLengths:
                if i in v:
                    sents_list.append((doc_sent[int(k)],k, i, v[i]["R1"]["f1"], v[i]["R2"]["f1"], v[i]["R3"]["f1"], v[i]["RL"]["f1"], v[i]["RS"]["f1"]))


        # Output scores to CSV - taking only F1 scores of - R1,R2,R3,RL,RS

        #calculateRouge.outputToCsv(allData, os.path.join(args.summ_sent_dir,summ_dir,"rouge_sim.csv"), systemNames, summaryLengths)
        df = pd.DataFrame(sents_list, columns = [summ_sent[int(summ_dir)],"sent_index", "summary_length", "R1-F1", "R2-F1", "R3-F1", "RL-F1", "RS-F1"])

        #Calculating rouge avergae of R1,R2, and RL as was done in Fei Liu's abstract summarization paper
        df["rouge_avg"] = (df["R1-F1"] + df["R2-F1"] + df["RL-F1"]) / 3

        df.to_csv("rouge_similarities/"+taskNames[0]+"-summary_sent_"+str(summ_dir)+".csv", index=False)
        df = df.sort_values(by=["rouge_avg"], ascending=False).reset_index(drop=True)

        summary_rouge_sim_indicies.append((summ_dir, summ_sent[int(summ_dir)], df.loc[0,"sent_index"]+":" + df.loc[0,df.columns[0]], df.loc[1,"sent_index"]+":" +df.loc[1,df.columns[0]], df.loc[2,"sent_index"]+":" + df.loc[2,df.columns[0]]))
        #rouge_vec = extract_rouge(allData, systemNames, summaryLengths)
        #rouge_mat[:, summ_sent_idx] = rouge_vec
        curTime = time.time()
        print('Current input done! Elapsed time: {} seconds!'.format(curTime - startTime))


    #np.savetxt("rouge_matrix.csv", rouge_mat, delimiter=",")
    sim_df = pd.DataFrame(summary_rouge_sim_indicies, columns=["summary_index","summary_sent", "best-avg-rouge-01","best-avg-rouge-02", "best-avg-rouge-03"])
    sim_df= sim_df.sort_values(by=["summary_index"])
    sim_df.to_csv("rouge_similarities/summaries_and_best_doc_sents.csv", index=False)
    print('---- DONE WITH ALL INPUTS')


