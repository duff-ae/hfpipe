import os
import numpy as np
import tables as pt

base = "/cephfs/brilshare/alshevel/hf_origin/hfet/23_fixed"

fill_list = sorted([int(fill) for fill in os.listdir(base)])

print(fill_list)

for fill in fill_list:
    
    f = f"{base}/{fill}/{fill}.hd5"
    if not os.path.exists(f):
        continue

    
    if int(fill) <= 8741:
        continue

    print("fix", f)

    with pt.open_file(f, "r+") as h5:
        tab = h5.root.hfet
        arr = tab.read()

        names = list(arr.dtype.names)
        #names[names.index("data")] = "bxraw"

        newdtype = [(n, arr.dtype[o]) for n, o in zip(names, arr.dtype.names)]
        new = np.empty(arr.shape, dtype=newdtype)

        for o, n in zip(arr.dtype.names, names):
            new[n] = arr[o]

        h5.remove_node("/", "hfet")
        h5.create_table("/", "hfetlumi", new)