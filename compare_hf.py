from glob import glob
import tables
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# config
fill_lists = ['fill_lists/pp22_part1.txt', 'fill_lists/pp22_part2.txt']
#fill_lists = ['fill_lists/pp23.txt']
#fill_lists = ['fill_lists/pp24_part1.txt', 'fill_lists/pp24_part2.txt']

hfoc_dir  = '/cephfs/brilshare/leeja/hf_reprocessed/hfoc/22'
hfet_dir  = '/cephfs/brilshare/alshevel/hf_reprocessed/hfet/22'
dt_dir    = '/eos/cms/store/group/dpg_bril/comm_bril/2022/online/orbit-integrated'

suffix = 22

fills = []
for file in fill_lists:
    with open(file, 'r') as f:
        fills.extend([int(i.split(' ')[-1]) for i in f.readlines()])

# TODO
fills = []
for file in glob(hfoc_dir + '/*'):
    fills.append(int(file.split('/')[-1]))

# manually remove special exception
try:
    fills.remove(8178)
except ValueError:
    pass
#fills = fills[:10]

# method for integrating files
def integrate(f, table, chunk_size=50000):
    tot = 0
    with tables.open_file(f) as file:
        table = file.root[table]
        for start in range(0, table.nrows, chunk_size):
            stop = min(start + chunk_size, table.nrows)
            chunk = table.read(start, stop, field='avgraw')
            tot += chunk[np.isfinite(chunk)].sum()
    return tot

# load lumi
hfoc_lumi, hfet_lumi, dt_lumi = [], [], []
online_hfoc, online_hfet = [], []
failed = []
for fill in fills:
    tot_et, tot_oc = 0, 0
    tot_hfet, tot_hfoc, tot_dt = 0, 0, 0

    try:
        # integrate hfoc
        tot_oc = integrate(hfoc_dir + f'/{fill}/{fill}.hd5', 'hfoclumi')

        # integrate hfet
        files = glob(hfet_dir + f'/{fill}/{fill}*.hd5')
        for f in files:
            tot_et += integrate(f, 'hfetlumi')

        # integrate online
        files = glob(dt_dir + f'/{fill}*.hd5')
        for f in files:
            tot_hfet += integrate(f, 'hfetlumi')
            tot_hfoc += integrate(f, 'hfoclumi')
            tot_dt   += integrate(f, 'dtlumi')
    except:
        failed.append(fill)
        continue

    hfet_lumi.append(tot_et)
    hfoc_lumi.append(tot_oc)
    online_hfet.append(tot_hfet)
    online_hfoc.append(tot_hfoc)
    dt_lumi.append(tot_dt)

print("Failed", failed)
for f in failed:
    fills.remove(f)

# plot
plt.title('Lumi Ratio vs Fill', fontsize='xx-large')
plt.xlabel('Fill', fontsize='x-large')
plt.ylabel('HFOC / HFET', fontsize='x-large')
plt.scatter(fills, np.array(hfoc_lumi) / np.array(hfet_lumi)) #TODO manually adjusted ratio
plt.axhline(1)
plt.savefig(f'/eos/user/l/leeja/HF/plots_hfpipe/compare_{suffix}.png')

# save to csv
df = pd.DataFrame.from_dict({
    "fill" : fills,
    "hfet" : hfet_lumi,
    "hfoc" : hfoc_lumi,
    "dt"   : dt_lumi,
    "hfet_online" : online_hfet,
    "hfoc_online" : online_hfoc,
})
df.to_csv(f'/eos/user/l/leeja/HF/plots_hfpipe/compare_{suffix}.csv')
