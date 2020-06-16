import pyranges as pr
import itertools
import pandas as pd
import os
pd.options.display.width = 0


def adjust_vcf(file, chrom_size):
    records = []
    dfs = []
    with open(file, 'r') as read_obj:
        for line in itertools.islice(read_obj, 7, None):
            records.append(list(line.split('\t')))
    print('Processing vcf')
    df = pd.DataFrame(records)
    df.columns = df.iloc[0]
    df.drop(0, inplace=True)
    df = df[['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', ]]
    df['DEL'] = [len(x) for x in df['REF']]
    df['IN'] = [len(x) for x in df['ALT']]
    df['SHIFT'] = df['IN'] - df['DEL']

    for chr in range(1, chrom_size + 1):
        dfs.append(df[df['#CHROM'] == str(chr)])

    vcf = pd.concat(dfs)
    return vcf


def adjust_gtf(file, vcf_file, new_file):
    vcf = vcf_file
    df = pr.read_gtf(file).df
    df = df[df['Feature'] == 'gene']
    df = df[df['gene_biotype'] == 'protein_coding']
    df = df[['Chromosome', 'Source', 'Feature', 'Start', 'End', 'Score', 'Strand', 'gene_id']]
    # To ensure we use just chromosome 1-5, excluding Mt and Pt
    df = df[df['Chromosome'].isin(['1', '2', '3', '4', '5'])]
    df.reset_index(inplace=True, drop=True)
    print('processing gtf')
    for idx, shift in enumerate(vcf['SHIFT']):
        if shift != 0:
            position = int(vcf['POS'][idx])
            chrom = vcf['#CHROM'][idx]
            for chr, start, end in zip(enumerate(df['Chromosome']), df['Start'], df['End']):
                start = int(start)
                end = int(end)
                if chr[1] == chrom:
                    if position < start and position < end:
                        df.loc[chr[0], 'Start'] = df.loc[chr[0], 'Start'] + shift
                        df.loc[chr[0], 'End'] = df.loc[chr[0], 'End'] + shift
                    elif start < position < end:
                        df.loc[chr[0], 'End'] = df.loc[chr[0], 'End'] + shift
    df.to_csv(new_file, header=False, index=False, sep='\t')


for file in os.listdir('/nam-99/ablage/nam/peleke/vcf_files'):
    vcf_path = '/nam-99/ablage/nam/peleke/vcf_files/' + file
    vcf = adjust_vcf(vcf_path, 5)
    ecotype_id = file.split('_')[1].split('.')[0]
    file_name_new_gtf = '/nam-99/ablage/nam/peleke/variant_gtfs/' + ecotype_id + '.gtf'
    ref_gtf = '/nam-99/ablage/nam/peleke/Arabidopsis_thaliana.TAIR10.46.gtf'
    if not os.path.isfile(file_name_new_gtf):
        adjust_gtf(ref_gtf, vcf, file_name_new_gtf)
