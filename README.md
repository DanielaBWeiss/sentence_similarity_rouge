# Aligning Sentences

### Experimenting with Sentence Similarity using Rouge Averages


**Current work**

Currently this repository is intended to experiment with the Rouge similarity scores between document and summary sentences.
The goal is to understand if using rouge as a semantic similarity measure between sentences is a good enough indicator as sentences that should be consolidated together.

The data I use is multi-document dataset - DUC-04
(Can be found here - https://duc.nist.gov/duc2004/tasks.html)

Right now this work aims to replicate this abstract summarization paper's ground-truth:
https://arxiv.org/pdf/1906.00077.pdf

Basically the goal is to create an alignment dataset between summary sentences, and document sentences.
The intuition is that summary sentences are created using one or a merge of more sentences from the input documents.

Some code parts for using Rouge library is taken from: https://github.com/oriern/SummEval_referenceSubsets/tree/master/code_score_extraction

##### To Run the Code

Follow instruction on Rouge installation here:
https://github.com/OriShapira/SummEval_referenceSubsets

Get the DUC-04 Data

You will need a document directory with all DUC documents under this one directory (at the moment I only process for a single topic),
and a summary directory with all the summaries for that single topic).
**All duc files (both summaries and document, although the user doesn't have to handle the document, the program transforms them), 
must have a ".html" ending, using the txt format of the DUC-04 documents and summaries. (just rename and add .html)

You will also need a directory inside both of those that will hole the rouge sentences
(Rouge needs to convert each document and summary sentence into a SEE format file)

Example command for running the code:
<code> python summary_sentence_alignment.py --input_doc doc --multi-doc --input_summ summ --doc_sent_dir doc/sentences --summ_sent_dir summ/sentences

Where --input_doc is the input document directory, --input_summ is summary directory containing the reference summaries for that document.
**<italic>Note: No other files (directories are okay) should be under the document and summary directory.</italic>
--doc_sent_dir is the directory for which the document sentences will be saved, and --summ_sent_dir is the directory where the summary sentences will be saved.
 ** you will also need a "rouge_similarities" folder for the output rouge_similarity scores.
 
 
 --Currently I am calculating the similarities based on the average R1,R2,RL F1 scores, as was done in the above mentioned paper.
 the file: "summaries_and_best_doc_sents.csv" represents the top three average F1 scores for each summary sentence,
 where as the "alignment_dataset" csv file contains the actual alignment dataset described in the paper.
 - This means that after calculating the rouge scores for a summary sentence, the best document sentence is chosen and all
 overlapping words with the summary sentence are taken out.
 - After this the rouge scores are calculated again using all document sentences, and if the next best sentence
 has more than 2 overlapping words with the summary sentence, it is also chosen as an aligned sentence to the summary sentence.
 (In the paper this is only done for two sentences, while I do it for one more).
 
