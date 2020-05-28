import pandas as pd
from Bio import SeqIO
import numpy as np
import pickle
import tensorflow
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras import models
from deeplift.layers import NonlinearMxtsMode
from deeplift.conversion import kerasapi_conversion as kc
from deeplift.dinuc_shuffle import dinuc_shuffle
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(42)
tensorflow.set_random_seed(42)

data = pd.read_csv('Data.csv')
data.set_index('Gene_id', inplace=True)


geneIDs = []
categories = []
seqs = []
dinuc_shuf_seqs = []

for prec, trec in zip(SeqIO.parse('promoter.fa', 'fasta'), SeqIO.parse('terminators.fa', 'fasta')):
    ID = prec.id

    pseq = str(prec.seq)
    pshuff = dinuc_shuffle(pseq)

    tseq = str(trec.seq)
    tshuff = dinuc_shuffle(tseq)

    seq = pseq + tseq
    shuff_seq = pshuff + tshuff

    if ID in data.index:
        seqs.append(seq)
        dinuc_shuf_seqs.append(shuff_seq)
        geneIDs.append(ID)

        category = data['label'][ID]
        if category == 'expressed':
            categories.append(1)
        else:
            categories.append(0)


categories = np_utils.to_categorical(np.array(categories), 2)

codes = {'A': [1, 0, 0, 0],
         'C': [0, 1, 0, 0],
         'G': [0, 0, 1, 0],
         'T': [0, 0, 0, 1]}


def one_hotencoder(sequence):
    one_hotencoding = np.zeros(shape=(4, len(sequence)))
    for i, nt in enumerate(sequence):
        one_hotencoding[:, i] = codes[nt]
    return one_hotencoding


one_hot_seq = np.expand_dims(np.array([one_hotencoder(seq) for seq in seqs], dtype=np.float32), 3)
one_hot_dinuc_shuff_seq = np.expand_dims(np.array([one_hotencoder(seq) for seq in dinuc_shuf_seqs],
                                                  dtype=np.float32), 3)

one_hot_seq[:, :, 1000:1003, :] = 0
one_hot_seq[:, :, 1997:2000, :] = 0
one_hot_dinuc_shuff_seq[:, :, 1000:1003, :] = 0
one_hot_dinuc_shuff_seq[:, :, 1997:2000, :] = 0

print('One Hot Encoding finished')

# Get testing genes from pickle file
file = open('Results/PICKLE2020-05-26215517', 'rb')
tested_IDs = pickle.load(file)[0]
testing_indices = [geneIDs.index(gene) for gene in tested_IDs]

test_data = one_hot_seq[testing_indices]
test_data_shuffled = one_hot_dinuc_shuff_seq[testing_indices]
test_categories = np.argmax(categories[testing_indices], axis=1)

# Load model and make predictions
model = models.load_model('Results/model2020-05-26215517.h5')
predictions = np.argmax(model.predict(test_data), axis=1)

print('Predictions done')
print(predictions)
deeplift_model =\
    kc.convert_model_from_saved_files('Results/model2020-05-26215517.h5',
                                      nonlinear_mxts_mode=NonlinearMxtsMode.DeepLIFT_GenomicsDefault)

deeplift_contribs_func = deeplift_model.get_target_contribs_func(find_scores_layer_idx=0,
                                                                 target_layer_idx=-2)

print('deeplift sound')
# calculate deeplift for each gene (only tp and tn)

tp_data = []
tn_data = []
tp_shuf_data = []
tn_shuf_data = []

for j, indx in enumerate(testing_indices):
    test_gene = geneIDs[indx]
    if test_categories[j] == 1 and predictions[j] == 1:
        tp_data.append(test_data[j])
        tp_shuf_data.append(test_data_shuffled[j])
    elif test_categories[j] == 0 and predictions[j] == 0:
        tn_data.append(test_data[j])
        tn_shuf_data.append(test_data_shuffled[j])

tp_data = np.array(tp_data)
tn_data = np.array(tn_data)
tp_shuf_data = np.array(tp_shuf_data)
tn_shuf_data = np.array(tn_shuf_data)

scores_tp = np.array(deeplift_contribs_func(task_idx=1, input_data_list=[tp_data],
                                            input_references_list=[tp_shuf_data], batch_size=10,
                                            progress_update=1000))
scores_tn = np.array(deeplift_contribs_func(task_idx=1, input_data_list=[tn_data],
                                            input_references_list=[tn_shuf_data], batch_size=10,
                                            progress_update=1000))

av_tp_scores = np.squeeze(np.average(scores_tp, axis=0))
av_tn_scores = np.squeeze(np.average(scores_tn, axis=0))

final_tp_scores = np.average(av_tp_scores, axis=0)
final_tn_scores = np.average(av_tn_scores, axis=0)


def visualise(data):
    sns.set()
    plt.plot(data)
    plt.axhspan(-0.00025, 0, alpha=0.3, color='r')
    plt.text(500, -0.00020, 'Promoter', fontweight='bold')
    plt.text(2000, -0.00020, 'Terminator', fontweight='bold')
    plt.show()

visualise(final_tp_scores)