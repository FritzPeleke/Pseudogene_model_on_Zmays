import pyranges as pr
import itertools
import pandas as pd
pd.options.display.width = 0


def adjust_vcf(file, chrom_size):
    records = []
    dfs = []
    with open(file, 'r') as read_obj:
        for line in itertools.islice(read_obj, 7, None):
            records.append(list(line.split('\t')))
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


vcf = adjust_vcf('intersection_10001.vcf', 5)


def adjust_gtf(file, vcf_file):
    vcf = vcf_file
    df = pr.read_gtf(file).df
    df = df[df['Feature'] == 'gene']
    df = df[df['gene_biotype'] == 'protein_coding']
    df = df[['Chromosome', 'Source', 'Feature', 'Start', 'End', 'Score', 'Strand', 'gene_id']]
    df = df[df['Chromosome'].isin(['1', '2', '3', '4', '5'])]# To ensure we use just chromosome 1-5, excluding Mt and Pt
    df.reset_index(inplace=True, drop=True)
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
    df.to_csv('variant.gtf', header=False, index=False, sep='\t')


adjust_gtf('Arabidopsis_thaliana.TAIR10.46.gtf', vcf)
