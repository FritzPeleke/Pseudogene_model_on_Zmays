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
from datetime import datetime

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
df['unexp'] = (df == 0).sum(axis=1)
df = df[df['unexp'] > 0]
gene_ids = list(df.index)
genomekeys_to_normcounts = list(df.columns)
pad = 'N'*50

alt_pad_ref_allgenomes = []
IDS_all_genomes = []
keys_all_genomes = []
path_norm_promoters = '/nam-99/ablage/nam/peleke/norm_promoters.fa'

print('Processing promoters')
for genome in os.listdir('/nam-99/ablage/nam/peleke/promoters'):
    genome_path = '/nam-99/ablage/nam/peleke/promoters/' + genome
    ecotype_id = genome.split('.')[0]
    genome_key = 'X' + ecotype_id

    if genome_key in genomekeys_to_normcounts:
        proms_per_genome = []
        id_per_genome = []
        keys_all_genomes.append(genome_key)
        for rec_alt, rec_ref in zip(SeqIO.parse(genome_path, 'fasta'), SeqIO.parse(path_norm_promoters, 'fasta')):
            id_alt = rec_alt.id.split(':')[1]
            id_ref = rec_ref.id.split(':')[1]
            seq_alt = str(rec_alt.seq)
            seq_ref = str(rec_ref.seq)
            sequence = seq_alt + pad + seq_alt
            if id_alt == id_ref and id_alt in gene_ids:
                proms_per_genome.append(sequence)
                id_per_genome.append(id_alt)
            alt_pad_ref_allgenomes.append(proms_per_genome)
            IDS_all_genomes.append(id_per_genome)

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

for promoters, ids, genome in zip(alt_pad_ref_allgenomes, IDS_all_genomes, keys_all_genomes):
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
                                                    shuffle=True, random_state=42, stratify=new_classes)

# Build model

backend.clear_session()

model = Sequential()
model.add(Conv2D(64, kernel_size=(4, 3), padding='valid', input_shape=(4, 2050, 1)))
model.add(Activation('relu'))
model.add(Conv2D(64, kernel_size=(1, 3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(rate=0.25))

model.add(Conv2D(128, kernel_size=(1, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, kernel_size=(1, 3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(rate=0.25))

model.add(Conv2D(64, kernel_size=(1, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, kernel_size=(1, 3), padding='same'))
model.add(Activation('relu'))
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
LR = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1)

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=256, epochs=40,
          validation_data=(X_test, y_test), callbacks=[Er_stop, LR])

now = datetime.now().strftime('%Y-%m-%d%H%M%S')
model.save('/nam-99/ablage/nam/peleke/Models/' + 'model' + now + '.h5')
backend.clear_session()