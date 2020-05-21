import numpy as np
from tensorflow.compat.v1 import set_random_seed
import pandas as pd
from Bio import SeqIO
from tensorflow.python.keras.utils import np_utils
import itertools
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPool2D, Flatten
from tensorflow.keras import Sequential
from tensorflow.keras import backend
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

np.random.seed(42)
set_random_seed(42)


Data = pd.read_csv('Data.csv', usecols=['Gene_id', 'label', 'max_TPM', 'sample'])
Data.set_index('Gene_id', inplace=True)


geneID = []
geneDes = []
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
    description = Prec.description

    seqProm = str(Prec.seq)
    seq_ssP = list(seqProm)
    np.random.shuffle(seq_ssP)
    sing_shuffle_Prom = ''.join(seq_ssP)
    
    seqTer = str(Trec.seq)
    seq_ssT = list(seqTer)
    np.random.shuffle(seq_ssT)
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
        geneDes.append(description)
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

df = pd.DataFrame({'Gene_id': geneID, 'Description': geneDes})
data = pd.read_csv('Data.csv')
data = data.merge(df, how='inner', on='Gene_id')
data.set_index('Gene_id', inplace=True)

# Custom one-hot encoder
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


labels = np_utils.to_categorical(np.array(labels), num_classes=2)
# geneID = np.array(geneID)
# Creating Train and Test sets
samples = [1, 2, 3, 4, 5]


def train_test_select():
    test_sample = 3
    train_samples = samples
    train_samples.remove(test_sample)

    testing_genes = data[data['sample'] == test_sample].index
    testing_genes = np.array(list(set(testing_genes).intersection(set(geneID))))
    # In testing set downsample expressed to balance expressed genes
    unexpressed_test_genes = [gene for gene in testing_genes if data.loc[gene, 'label'] == 'unexpressed']
    expressed_test_genes = [gene for gene in testing_genes if data.loc[gene, 'label'] == 'expressed']
    expressed_test_genes = np.random.choice(expressed_test_genes, len(unexpressed_test_genes))
    testing_genes = np.concatenate((expressed_test_genes, unexpressed_test_genes), axis=0)

    training_genes = []
    for sample in train_samples:
        genes = list(data[data['sample'] == sample].index)
        training_genes.append(genes)
    training_genes = list(itertools.chain(*training_genes))
    un_train_genes = [gene for gene in training_genes if data.loc[gene, 'label'] == 'unexpressed']
    ex_train_genes = [gene for gene in training_genes if data.loc[gene, 'label'] == 'expressed']
    ex_train_genes = np.random.choice(ex_train_genes, len(un_train_genes))
    training_genes = np.concatenate((ex_train_genes, un_train_genes), axis=0)

    train_indices = np.array([geneID.index(gene) for gene in training_genes])
    np.random.shuffle(train_indices)
    test_indices = np.array([geneID.index(gene) for gene in testing_genes])

    return train_indices, test_indices


def create_sets(sequences, label):
    train_indx, test_indx = train_test_select()
    x_train = sequences[train_indx]
    y_train = label[train_indx]
    x_test = sequences[test_indx]
    y_test = label[test_indx]

    return x_train, y_train, x_test, y_test


model = Sequential()

model.add(Conv2D(64, kernel_size=(4, 8), padding='valid', input_shape=[4, 3000, 1],
                 activation='relu'))
model.add(Conv2D(64, kernel_size=(1, 8), padding='same', activation='relu'))
model.add(Conv2D(64, kernel_size=(1, 8), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(1, 8), strides=(1, 8), padding='same'))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(1, 8), padding='same', activation='relu'))
model.add(Conv2D(128, kernel_size=(1, 8), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(1, 8), strides=(1, 8), padding='same'))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(1, 8), padding='same', activation='relu'))
model.add(Conv2D(64, kernel_size=(1, 8), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(1, 8), strides=(1, 8), padding='same'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, 'relu'))
model.add(Dropout(0.25))
model.add(Dense(64, 'relu'))
model.add(Dense(2, 'softmax'))

print(model.summary())

Er_Stop = EarlyStopping(monitor='val_loss', patience=3, verbose=0)
x_train, y_train, x_test, y_test = create_sets(one_hot_seq, labels)

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=256, epochs=40,
          validation_data=(x_test, y_test), callbacks=[Er_Stop])
backend.clear_session()
