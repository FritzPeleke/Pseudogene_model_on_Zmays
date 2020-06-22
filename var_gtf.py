import pyranges as pr
import pandas as pd
import os
pd.options.display.width = 0


def edit_vcf(input_vcf, chr_no):
    df = pd.read_csv(input_vcf, skiprows=7, sep='\t')
    df['DEL'] = [len(x) for x in df['REF']]
    df['IN'] = [len(x) for x in df['ALT']]
    df['SHIFT'] = df['IN'] - df['DEL']
    df = df[df['SHIFT'] != 0]
    df.reset_index(drop=True, inplace=True)
    dfs = [df[df['#CHROM'] == x] for x in range(1, chr_no + 1)]
    vcf = []
    for df in dfs:
        df['cum_sum'] = df['SHIFT'].cumsum()
        df.reset_index(drop=True, inplace=True)
        vcf.append(df)

    return vcf


def edit_gtf(input_gtf, chr_no):
    df = pr.read_gtf(input_gtf).df
    df = df[['Chromosome', 'Source', 'Feature', 'Start', 'End', 'Score', 'Strand', 'gene_id', 'gene_biotype']]
    df = df[df['Feature'] == 'gene']
    df = df[df['gene_biotype'] == 'protein_coding']
    df.drop(df[df['Chromosome'] == 'Mt'].index, inplace=True)
    df.drop(df[df['Chromosome'] == 'Pt'].index, inplace=True)
    df = df.astype({'Chromosome': 'int32'})
    dfs = [df[df['Chromosome'] == x] for x in range(1, chr_no + 1)]
    gtf = []
    for df in dfs:
        df.reset_index(drop=True, inplace=True)
        gtf.append(df)

    return gtf


def max_less(iterable, val):
    a = [num for num in iterable if num < val]
    return max(a, default='None')


def get_gtfvariant(in_vcf, in_gtf, out_gtf):
    print('Processing')
    edited_gtfs = []

    for vcf, gtf in zip(in_vcf, in_gtf):
        closest_to_start = []
        closest_to_end = []
        pos = vcf['POS'].tolist()
        Starts = gtf['Start'].tolist()
        Ends = gtf['End'].tolist()
        print(vcf.tail(2))
        for start in Starts:
            closest_to_start.append(max_less(pos, start))
        for end in Ends:
            closest_to_end.append(max_less(pos, end))
        idx_close_to_start = [pos.index(x) if x != 'None' else 'None' for x in closest_to_start]
        idx_close_to_end = [pos.index(y) if y != 'None' else 'None' for y in closest_to_end]
        cumsums_start = [vcf.loc[x, 'cum_sum'] if x != 'None' else 0 for x in idx_close_to_start]
        cumsums_end = [vcf.loc[y, 'cum_sum'] if y != 'None' else 0 for y in idx_close_to_end]

        gtf['start_shift'] = cumsums_start
        gtf['end_shift'] = cumsums_end
        gtf['pos_nearest_tostrt'] = closest_to_start
        gtf['pos_nearest_toend'] = closest_to_end
        gtf.loc[gtf['pos_nearest_tostrt'] == 'None', 'pos_nearest_tostrt'] = 1000000000
        gtf.loc[gtf['pos_nearest_toend'] == 'None', 'pos_nearest_toend'] = 1000000000

        gtf['Start'] = [start + cum if start > pos else start
                        for start, cum, pos in zip(gtf['Start'], gtf['start_shift'], gtf['pos_nearest_tostrt'])]
        gtf['End'] = [end + cum if end > pos else end
                      for end, cum, pos in zip(gtf['End'], gtf['end_shift'], gtf['pos_nearest_toend'])]
        gtf['Feature'] = gtf['Feature'].astype(str) + ':' + gtf['gene_id'].astype(str)
        edited_gtfs.append(gtf)
        print(gtf.tail(2))
    final_gtf = pd.concat(edited_gtfs)
    final_gtf.reset_index(drop=True, inplace=True)
    final_gtf.drop(labels=['start_shift', 'end_shift', 'pos_nearest_tostrt', 'pos_nearest_toend'], inplace=True,
                   axis=1)

    print(final_gtf.tail())
    print(final_gtf.shape)

    return final_gtf.to_csv(out_gtf, index=False, header=False, sep='\t')


for file in os.listdir('/nam-99/ablage/nam/peleke/vcf_files'):
    gtf = edit_gtf('Arabidopsis_thaliana.TAIR10.46.gtf', 5)
    vcf_path = '/nam-99/ablage/nam/peleke/vcf_files/' + file
    vcf = edit_vcf(vcf_path, 5)
    ecotype_id = file.split('_')[1].split('.')[0]
    output_gtf = '/nam-99/ablage/nam/peleke/variant_gtfs/' + ecotype_id + '.gtf'
    get_gtfvariant(vcf, gtf, output_gtf)
