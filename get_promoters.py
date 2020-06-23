import os

os.system('module load bedtools')

for gtf in os.listdir('/nam-99/ablage/nam/peleke/1000upstream_gtfs/'):
    ecotype_id = gtf.split('.')[0]
    variant_gtf = '/nam-99/ablage/nam/peleke/1000upstream_gtfs/' + gtf
    variant_genome = '/nam-99/ablage/nam/peleke/variant_genomes/' + ecotype_id + '.fa'
    save_path = '/nam-99/ablage/nam/peleke/promoters/' + ecotype_id + '.fa'
    os.system('bedtools getfasta -fo {} -fi {} -bed {} -s -name'.format(save_path, variant_genome, variant_gtf))
    print('Done with ecotype_id ', ecotype_id)



