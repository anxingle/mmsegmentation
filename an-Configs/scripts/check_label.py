import os, numpy as np, cv2, glob
root='/data/an/mmsegmentation/datasets/tongue_seg_v0/ann_dir/train'
bad=[]
for p in glob.glob(os.path.join(root,'**','*.png'), recursive=True):
    m=cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if m is None:
        bad.append((p,'read_fail')); continue
    if m.ndim==3:
        bad.append((p,'rgb_mask')); continue
    u=np.unique(m)
    if not set(u.tolist()).issubset({0,1}):
        bad.append((p, f'unique={u[:20]}'))
print('BAD COUNT:',len(bad))
for i,(p,msg) in enumerate(bad[:50]):
    print(i, msg, '->', p)
