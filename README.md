# SC

###Experimenting with Sentence Consolidation


**Current work**

Currently this repository is intended to experiment with Rouge sentence similarity on document sentences and reference summaries.

The data I use is multi-document dataset - DUC-04
(Can be found here - https://duc.nist.gov/duc2004/tasks.html)

Right now this work aims to replicate this paper's ground-truth:
https://arxiv.org/pdf/1906.00077.pdf

Code parts for calculating sentence similarity using Rougeis taken from: https://github.com/oriern/SummEval_referenceSubsets/tree/master/code_score_extraction

#####To Run the Code

Follow instruction on Rouge installation here:
https://github.com/OriShapira/SummEval_referenceSubsets

Get the DUC-04 Data

You will need a document directory with a single DUC document
and a summary directory with a single reference summary.

You will also need a directory inside both of those that will hole the rouge sentences
(Rouge needs to convert each document and summary sentence into a SEE format file)

Example command for running the code:
<code> python calculate_rouge_similarities.py --input_doc [doc/APW19981016.0240] --input_summ [summ/D30001.M.100.T.A] --doc_sent_dir [doc/rouge_sents] --summ_sent_dir [summ/rouge_sents] </code>


Where --input_doc is the input document, --input_summ is the reference summary for that document.
--doc_sent_dir is the directory for which the document sentences will be saved, and --summ_sent_dir is the directory where the summary sentences will be saved.
 ** you will also need a "rouge_similarities" folder for the output rouge_similarity scores.
 
 
 --Currently I am calculating the similarities based on the average R1,R2,RL as was done in the above mentioned paper.
 the file: "summaries_and_best_doc_sents.csv" represents the top three average F1 scores for each summary sentence.
 