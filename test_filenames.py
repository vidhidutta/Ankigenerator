# test_filenames.py
ppt_basename = "anatomy_review"
slide_idx    = 5
regions      = [(10,10,50,50), (100,100,40,40)]

filenames = []
for i, _ in enumerate(regions):
    q = f"{ppt_basename}_slide{slide_idx}_reg{i}_q.png"
    o = f"{ppt_basename}_slide{slide_idx}_reg{i}_o.png"
    filenames.extend([q, o])

print("\n".join(filenames))
