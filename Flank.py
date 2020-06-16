import itertools
import pandas as pd

def flank(gtf_file, pro_size):
    new_start = []
    new_end = []
    records = []
    with open(gtf_file, 'r') as read_obj:
        for line in itertools.islice(read_obj, 0, None):
            records.append(list(line.split('\t')))
            if line.split('\t')[6] == '+':
                new_start.append(int(line.split('\t')[3]) - pro_size)
                new_end.append(int(line.split('\t')[3]) - 1)
            else:
                new_start.append(int(line.split('\t')[4]) + 1)
                new_end.append(int(line.split('\t')[4]) + pro_size)

    df = pd.DataFrame(records)
    df[3] = new_start
    df[4] = new_end
    df[7] = [x.split("\n")[0] for x in df[7]]

    return df.to_csv("/home/peleke/PycharmProjects/Arabidopsis/Athalianagenome/var.1000.gtf",
                     header=False, index=False, sep='\t')


flank("/home/peleke/PycharmProjects/Arabidopsis/Athalianagenome/Arabidopsis_thaliana.TAIR10.46.var.gtf", 1000)
#flank("/home/peleke/PycharmProjects/Arabidopsis/Athalianagenome/genes_only_normal.gtf", 1000)
