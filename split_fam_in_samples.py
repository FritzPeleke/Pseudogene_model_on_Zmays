import pandas as pd
import numpy as np

np.random.seed(42)

dat = pd.read_csv('gene_families.csv', usecols=['family_id', 'Gene_id'])
labels = pd.read_csv('labels.csv', usecols=['Geneid', 'max_TPM', 'label'])

labels.rename(columns={'Geneid': 'Gene_id'}, inplace=True)
dat = dat.merge(labels, how='inner')

list_of_families = list(set([x for x in dat.family_id]))
np.random.shuffle(list_of_families)
splits = np.array_split(np.array(list_of_families), 5)

sample_id = []
indx = []
for i, sample in enumerate(splits):
    for fam_id, gene_id in zip(dat.family_id, dat.Gene_id):
        if fam_id in sample:
            Gene_index = list(dat.Gene_id).index(gene_id)
            sample_id.append(i+1)
            indx.append(Gene_index)

dat.loc[indx, 'sample'] = sample_id

dat.to_csv('Data.csv', index=False)
