import pandas as pd
import numpy as np
from Bio import SeqIO
import os
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras import models
from tensorflow import set_random_seed
from deeplift.layers import NonlinearMxtsMode
from deeplift.dinuc_shuffle import dinuc_shuffle
from deeplift.conversion import kerasapi_conversion as kc
from sklearn.metrics import accuracy_score

np.random.seed(42)
set_random_seed(42)
pd.options.display.width = 0

# Load data for gene_families from PLAZA and data for norm_counts
norm_counts = pd.read_csv('/nam-99/ablage/nam/peleke/GSE80744_ath1001_tx_norm_2016-04-21-UQ_gNorm_normCounts_k4.tsv',
                          sep='\t')
norm_counts.set_index('gene_id', drop=True, inplace=True)
print(norm_counts.head())
'''norm_counts.replace(0, 0.001, inplace=True)
norm_counts = norm_counts.apply(np.log2)
norm_counts = norm_counts.sub(norm_counts.mean(axis=1), axis=0).div(norm_counts.std(axis=1), axis=0)
print(norm_counts.head())'''

fam_df = pd.read_csv('/nam-99/ablage/nam/peleke/data.txt', sep='\t')
protein_binding_genes = fam_df['# gene_id'].values.tolist()
df = norm_counts
df = df[df.index.isin(protein_binding_genes)]
print(df.head(30))
gene_ids = list(df.index)
genomekeys_to_normcounts = list(df.columns)

# All promoter per genome and all ids per genome
proms_all_genomes = []
keys_all_genomes = []
exp_count = []
path_norm_promoters = '/nam-99/ablage/nam/peleke/norm_promoters.fa'
pad = 'N'*50

print('Processing Promoters')
for genome in os.listdir('/nam-99/ablage/nam/peleke/snp_promoters'):
    genome_path = '/nam-99/ablage/nam/peleke/snp_promoters/' + genome
    ecotype_id = genome.split('.')[0]
    genome_key = 'X' + ecotype_id

    if genome_key in genomekeys_to_normcounts:
        proms_per_genome = []
        id_per_genome = []
        keys_all_genomes.append(genome_key)
        for rec_alt in SeqIO.parse(genome_path, 'fasta'):
            id_alt = rec_alt.id.split(':')[1]
            seq_alt = str(rec_alt.seq)
            sequence = seq_alt
            if id_alt == 'AT4G15360':
                proms_all_genomes.append(sequence)
                exp_count.append(df[genome_key][id_alt])

print(len(set(proms_all_genomes)))
print(len(proms_all_genomes))

# Encode sequences
codes = {'A': [1, 0, 0, 0],
         'C': [0, 1, 0, 0],
         'G': [0, 0, 1, 0],
         'T': [0, 0, 0, 1],
         'N': [0, 0, 0, 0]}


def one_hot_encoder(seq):
    one_hot_encoding = np.zeros(shape=(4, len(seq)))
    for i, nt in enumerate(seq):
        one_hot_encoding[:, i] = codes[nt]
    return one_hot_encoding


encoded_proms = []
encoded_shuff_proms = []
label = [0 if x < 1 else 1 for x in exp_count]
classes = np_utils.to_categorical(label, num_classes=2)

for promoter in proms_all_genomes:
    shuf_prom = dinuc_shuffle(promoter)
    encoded_proms.append(one_hot_encoder(promoter))
    encoded_shuff_proms.append(one_hot_encoder(shuf_prom))

prepared_proms = np.expand_dims(np.array(encoded_proms, dtype=np.float32), axis=3)
prepared_shuff_proms = np.expand_dims(np.array(encoded_shuff_proms, dtype=np.float32), axis=3)
print(prepared_proms.shape)
print(prepared_shuff_proms.shape)
print(classes.shape)


model = models.load_model('/nam-99/ablage/nam/peleke/Models/model2020-07-30150217.h5')

predictions = np.argmax(model.predict(prepared_proms), axis=1)
print(predictions)
actual = np.argmax(classes, axis=1)
print('Predictions done')

deeplift_model =\
    kc.convert_model_from_saved_files('/nam-99/ablage/nam/peleke/Models/model2020-07-30150217.h5',
                                      nonlinear_mxts_mode=NonlinearMxtsMode.DeepLIFT_GenomicsDefault)

deeplift_contrib_func = deeplift_model.get_target_contribs_func(find_scores_layer_idx=0,
                                                                target_layer_idx=-2)
# Calculate contributions scores for tps and tns
tp = []
tp_shuff = []
tn = []
tn_shuff = []

for pred, true, enc_seq, enc_shuf_seq in zip(predictions, actual, prepared_proms, prepared_shuff_proms):
    if pred == 1 and true == 1:
        tp.append(enc_seq)
        tp_shuff.append(enc_shuf_seq)
    elif pred == 0 and true == 0:
        tn.append(enc_seq)
        tn_shuff.append(enc_shuf_seq)

tp_data = np.array(tp)
tn_data = np.array(tn)
tp_shuff_data = np.array(tp_shuff)
tn_shuff_data = np.array(tn_shuff)
print(tp_data.shape)
print(tn_data.shape)
print(accuracy_score(actual, predictions))

scores_tp = np.array(deeplift_contrib_func(task_idx=1, input_data_list=[tp_data],
                                           input_references_list=[tp_shuff_data],
                                           batch_size=5, progress_update=10))
scores_tn = np.array(deeplift_contrib_func(task_idx=1, input_data_list=[tn_data],
                                           input_references_list=[tn_shuff_data],
                                           batch_size=5, progress_update=10))

av_tp_scores = np.squeeze(np.average(scores_tp, axis=0))
av_tn_scores = np.squeeze(np.average(scores_tn, axis=0))

pd.DataFrame(av_tp_scores).to_csv('/nam-99/ablage/nam/peleke/Scores/snp_AT4G15360_2_tp_scores.csv')
pd.DataFrame(av_tn_scores).to_csv('/nam-99/ablage/nam/peleke/Scores/snp_AT4G15360_2_tn_scores.csv')

