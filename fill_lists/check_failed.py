from glob import glob

# config
#fill_lists = ['pp22_part1.txt', 'pp22_part2.txt']
#fill_lists = ['pp23.txt']
fill_lists = ['pp24_part1.txt', 'pp24_part2.txt']

check_dir  = '/cephfs/brilshare/leeja/hf_reprocessed/hfoc/24'

fills = []
for file in fill_lists:
    with open(file, 'r') as f:
        fills.extend([int(i.split(' ')[-1]) for i in f.readlines()])

reproc = [int(i.split('/')[-1]) for i in glob(f'{check_dir}/*')]
#for fill in fills:
#    if fill not in reproc:
#        print(' -', fill)

# compare to hfet
hfet = [int(i.split('/')[-1]) for i in glob(f'/cephfs/brilshare/alshevel/hf_reprocessed/hfet/24_final/*')]
for fill in sorted(hfet):
    if fill not in reproc:
        print(' -', fill)
