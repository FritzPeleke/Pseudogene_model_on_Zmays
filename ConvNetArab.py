import pandas as pd
import numpy as np
from Bio import SeqIO
import os
from itertools import chain
from tensorflow.python.keras.utils import np_utils
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend
from tensorflow import set_random_seed

np.random.seed(42)
set_random_seed(42)
pd.options.display.width = 0

# Load data for gene_families from PLAZA and data for norm_counts
norm_counts = pd.read_csv('/nam-99/ablage/nam/peleke/GSE80744_ath1001_tx_norm_2016-04-21-UQ_gNorm_normCounts_k4.tsv',
                          sep='\t')
fam_df = pd.read_csv('/nam-99/ablage/nam/peleke/data.txt', sep='\t')

# Edit normalised counts to keep only those for genes in familiy
protein_binding_genes = fam_df['# gene_id'].values.tolist()
df = norm_counts.set_index('gene_id')
df = df[df.index.isin(protein_binding_genes)]
gene_ids = list(df.index)
genomekeys_to_normcounts = list(df.columns)


# All promoter per genome and all ids per genome
proms_all_genomes = []
IDS_all_genomes = []
key_all_genomes = []

print('Processing Promoters')
for genome in os.listdir('/nam-99/ablage/nam/peleke/promoters'):
    genome_path = '/nam-99/ablage/nam/peleke/promoters/' + genome
    ecotype_id = genome.split('.')[0]
    genome_key = 'X' + ecotype_id

    if genome_key in genomekeys_to_normcounts:
        proms_per_genome = []
        ID_per_genome = []
        key_all_genomes.append(genome_key)
        for rec in SeqIO.parse(genome_path, 'fasta'):
            ID = rec.id.split(':')[1]
            seq = str(rec.seq)
            if ID in gene_ids:
                proms_per_genome.append(seq)
                ID_per_genome.append(ID)
        proms_all_genomes.append(proms_per_genome)
        IDS_all_genomes.append(ID_per_genome)

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


prepared_prom = []
categories = []

for promoters, ids, genome in zip(proms_all_genomes, IDS_all_genomes, key_all_genomes):
    encoded_prom = []
    label = [0 if df[genome][gene] < 1 else 1 for gene in ids]
    for seq in promoters:
        encoded_prom.append(one_hot_encoder(seq))

    prepared_prom.append(np.array(encoded_prom, dtype=np.float32))
    categories.append(label)

true_labels = np.array(list(chain(*categories)))
full_data = np.expand_dims(np.concatenate(prepared_prom, axis=0), 3)
classes = np_utils.to_categorical(true_labels, num_classes=2)

print('One Hot encoding done, Data Prepared')
# Downsample expreesed genes to meet unexpressed genes
indices_unexp = np.where(true_labels == 0)[0]
indices_ex = np.where(true_labels == 1)[0]
print(len(indices_ex), len(indices_unexp))
selected_idx_exp = np.random.choice(indices_ex, len(indices_unexp))

sel_genes_ex = np.take(full_data, selected_idx_exp, axis=0)
sel_cls_ex = np.take(classes, selected_idx_exp, axis=0)
sel_genes_unex = np.take(full_data, indices_unexp, axis=0)
sel_cls_unex = np.take(classes, indices_unexp, axis=0)
new_data = np.concatenate((sel_genes_ex, sel_genes_unex), axis=0)
new_classes = np.concatenate((sel_cls_ex, sel_cls_unex), axis=0)

# split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(new_data, new_classes, test_size=0.2,
                                                    shuffle=True, random_state=42)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Build model
backend.clear_session()

model = Sequential()
model.add(Conv2D(64, kernel_size=(4, 8), padding='valid', input_shape=(4, 1000, 1)))
model.add(Activation('relu'))
model.add(Conv2D(64, kernel_size=(1, 8), padding='same'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(1, 8), padding='same'))
model.add(Dropout(rate=0.25))

model.add(Conv2D(128, kernel_size=(1, 8), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, kernel_size=(1, 8), padding='same'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(1, 8), padding='same'))
model.add(Dropout(rate=0.25))

model.add(Conv2D(64, kernel_size=(1, 8), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, kernel_size=(1, 8), padding='same'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(1, 8), padding='same'))
model.add(Dropout(rate=0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('softmax'))
print(model.summary())

# Compile and Train
Er_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=256, epochs=40,
          validation_data=(X_test, y_test), callbacks=[Er_stop])











