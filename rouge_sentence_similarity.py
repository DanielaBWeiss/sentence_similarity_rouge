import calculateRouge
import pandas as pd
import sys

from utils import *


class RougeSentenceSimilarity():
    """
    Rouge Sentence Similarity class saves all document and summary configurations that are required to pass to the Rouge python
    module.
    This class contains functions that write and transform document files to sentences, and retrieves and saves rouge scores into
    panda dataframes that contain similarity scores between sentences based on avergae rouge scores of a chosen metric.
    """
    def __init__(self, summ_topic, doc_topic, input_doc, input_summ, doc_dir, summ_dir, metric):
        self.doc_sent_dir = doc_dir
        self.summ_sent_dir = summ_dir
        self.input_doc = input_doc
        self.input_summ = input_summ
        self.summ_topic = summ_topic
        self.doc_topic = doc_topic
        self.metric = metric
        self.duc_version = 2004

        self.task_names = None
        self.summary_lengths = None
        self.system_names = None
        self.remove_stop_words = None
        self.output_path = None
        self.doc_sentences_indicies = None
        self.doc_sents = None
        self.summ_sents = None

        # Deleting Mac files
        delete_mac_files([self.doc_sent_dir, self.summ_sent_dir, self.input_doc])

    def write_and_transform_documents(self, multi_doc):
        """Splitting all document sentences to sentence files, and splitting each summary sentence to its own file in its own
        directory.
        """
        try:
            self.doc_sents, self.doc_sent_indicies = doc_sentences_extract(self.input_doc, self.doc_sent_dir, self.doc_topic, multi_doc)
            self.summ_sents, self.summ_sent_indicies = summ_sentences_extract(self.input_summ, self.summ_sent_dir,self.summ_topic)
        except:
            print("Error: {} occurred when extracting and writing documents and summaries into sentence files.".format(sys.exc_info()[0]))
            exit(1)


    def get_rouge_scores(self, summ_dir, compare_type, remove_stop_words, override_dir=False):
        """ Retrieving rouge scores for all document sentences and current summary sentence.

        Depending on the doc topic and summary length, calculateRouge creates scores for all combinations
        that depend on the choices passed to the function.
        Returns:
            all_data: dictionary containing all rouge scores for current summary sentence against all document sentences.
        """
        if override_dir:
            ref_folder = summ_dir
        else:
            ref_folder = os.path.join(self.summ_sent_dir,summ_dir)

        csv_dir = os.path.join(self.summ_sent_dir,summ_dir)
        # get the different options for comparison, each system sum is put into comparison with all ref summaries, and by summary length
        self.task_names, self.system_names, self.summary_lengths = calculateRouge.getComparisonOptions( self.doc_sent_dir, ref_folder)

        # get ROUGE scores:
        all_data = calculateRouge.runRougeCombinations(compare_type, self.doc_sent_dir, ref_folder, self.system_names, self.summary_lengths, self.duc_version, remove_stop_words)

        return all_data


    def get_top_sentences(self, all_data, summ_sent, summ_dir, save=True, previous_sentence=None):
        """ Function to save all rouge scores for all document sentences per summary reference.
        Args:
            all_data: contains all rouge scores
            summ_sent: the current summary reference sentence
            summ_dir: the summary reference sentence's file's directory location

        Returns:
            triple: The top three document sentences w.r.t avg rouge scores.
            best_sentence:

        """
        sents_list = []
        # k - sentence index, i - summary length, and the rest are the scores
        for k, v in all_data.items():
            for i in self.summary_lengths:
                if i in v:
                    sents_list.append((self.doc_sents[int(k)], k, i, v[i]["R1"]["f1"], v[i]["R2"]["f1"], v[i]["R3"]["f1"],
                                       v[i]["RL"]["f1"], v[i]["RS"]["f1"]))

        # Output scores to CSV - taking only F1 scores of - R1,R2,R3,RL,RS
        summ_ind = find_sent_index(self.summ_sent_indicies,summ_dir)
        df = pd.DataFrame(sents_list,
                          columns=[summ_ind+":"+summ_sent, "sent_index", "summary_length", "R1-F1", "R2-F1", "R3-F1",
                                   "RL-F1", "RS-F1"])

        # Calculating rouge avergae of R1,R2, and RL as was done in Fei Liu's abstract summarization paper
        df["rouge_avg"] = (df["R1-F1"] + df["R2-F1"] + df["RL-F1"]) / 3

        df = df.sort_values(by=["rouge_avg"], ascending=False).reset_index(drop=True)
        if save:
            df.to_csv("rouge_similarities/" + self.task_names[0] + "-summary_sent_" + str(summ_dir) + ".csv", index=False)

        best_ind = find_sent_index(self.doc_sent_indicies, df.loc[0, "sent_index"])
        best_2ind = find_sent_index(self.doc_sent_indicies, df.loc[1, "sent_index"])
        best_3ind = find_sent_index(self.doc_sent_indicies, df.loc[2, "sent_index"])

        best_indicies = [best_ind, best_2ind, best_3ind]
        triple = (summ_dir, summ_sent, best_ind + ":" + df.loc[0, df.columns[0]],
                best_2ind + ":" + df.loc[1, df.columns[0]], best_3ind + ":" + df.loc[2, df.columns[0]])
        best_sentence = (best_ind, df.loc[0, df.columns[0]], df.loc[0, "rouge_avg"])
        if previous_sentence:
            found=False
            for i in range(3):
                if df.loc[i, df.columns[0]] == previous_sentence: continue
                else:
                    best_sentence = (best_indicies[i],df.loc[i, df.columns[0]], df.loc[i, "rouge_avg"] )

        return triple, best_sentence

