import pandas as pd
import numpy as np
import random
from Bio import SeqIO
from tensorflow.python.keras.utils import np_utils

np.random.seed(42)

Data = pd.read_csv('labels.csv', usecols=['Geneid', 'label'])
Data.set_index('Geneid', inplace=True)

geneID = []
labels = []

# normal sequences
seqs = []
seq_prom = []
seq_ter = []

# di-nucleotide shuffled seq
seq_ds = []
seq_dsprom = []
seq_dster = []

# single-nucleotide shuffle sequences
seq_ss = []
seq_ssprom = []
seq_sster = []


for Prec, Trec in zip(SeqIO.parse('promoter.fa', 'fasta'), SeqIO.parse('terminators.fa', 'fasta')):
    ID = Prec.id

    seqProm = str(Prec.seq)
    seq_ssP = list(seqProm)
    random.shuffle(seq_ssP)
    sing_shuffle_Prom = ''.join(seq_ssP)
    
    seqTer = str(Trec.seq)
    seq_ssT = list(seqTer)
    random.shuffle(seq_ssT)
    sing_shuffle_Term = ''.join(seq_ssT)
    
    sequence = seqProm + seqTer
    shuffled_sequence = sing_shuffle_Prom + sing_shuffle_Term
    
    if ID in Data.index:
        seqs.append(sequence)
        seq_prom.append(seqProm)
        seq_ter.append(seqTer)

        seq_ss.append(shuffled_sequence)
        seq_ssprom.append(seq_ssP)
        seq_sster.append(seq_ssT)

        geneID.append(ID)
        category = Data.loc[str(ID), 'label']
        if category == 'expressed':
            labels.append(1)
        else:
            labels.append(0)


# Parsing di-nucleotide shuffled sequences
for Prec, Trec in zip(SeqIO.parse('dinucl_shuf_promoters.fa', 'fasta'), SeqIO.parse('dinucl_shuf_terminators.fa', 'fasta')):
    ID = Prec.id
    seq_prom = str(Prec.seq)
    seq_term = str(Trec.seq)
    seq_merged = seq_prom + seq_term

    if ID in Data.index:
        seq_ds.append(seq_merged)
        seq_dsprom.append(seq_prom)
        seq_dster.append(seq_term)


#  Custom one-hot encoder
codes = {'A': [1, 0, 0, 0],
         'C': [0, 1, 0, 0],
         'G': [0, 0, 1, 0],
         'T': [0, 0, 0, 1]}


def onehot_encoder(seq):
    one_hot_encoded = np.zeros(shape=(4, len(seq)))
    for i, nt in enumerate(seq):
        one_hot_encoded[:, i] = codes[nt]
    return one_hot_encoded

# Encoding normal sequenecs

one_hot_seq = np.expand_dims(np.array([onehot_encoder(seq) for seq in seqs], dtype=np.float32), 3)
one_hot_pro = np.expand_dims(np.array([onehot_encoder(seq) for seq in seq_prom], dtype=np.float32), 3)
one_hot_ter = np.expand_dims(np.array([onehot_encoder(seq) for seq in seq_ter], dtype=np.float32), 3)

# masking
# NB:we have 4 slicing indices because the last determines the channels
one_hot_seq[:, :, 1000:1003, :] = 0
one_hot_seq[:, :, 1997:2000, :] = 0
one_hot_pro[:, :, 1000:1003, :] = 0
one_hot_ter[:, :, 497:500, :] = 0

# Encoding shuffled sequences
ss_encoded = np.expand_dims(np.array([onehot_encoder(seq) for seq in seq_ss], dtype=np.float32), 3)
ss_prom_encoded = np.expand_dims(np.array([onehot_encoder(seq) for seq in seq_ssprom], dtype=np.float32), 3)
ss__ter_encoded = np.expand_dims(np.array([onehot_encoder(seq) for seq in seq_sster], dtype=np.float32), 3)

ds_encoded = np.expand_dims(np.array([onehot_encoder(seq) for seq in seq_ds], dtype=np.float32), 3)
ds_prom_encoded = np.expand_dims(np.array([onehot_encoder(seq) for seq in seq_dsprom], dtype=np.float32), 3)
ds_term_encoded = np.expand_dims(np.array([onehot_encoder(seq) for seq in seq_dster], dtype=np.float32), 3)

geneID = np.array(geneID)
labels = np_utils.to_categorical(np.array(labels), num_classes=2)