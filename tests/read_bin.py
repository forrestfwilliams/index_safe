import data

burst0 = data.Offset(start=109035, stop=153814954)
with open('swath_IW2_VV.tiff', 'rb') as f:
    f.seek(burst0.start)
    golden = f.read(burst0.stop - burst0.start)


with open('test0_IW2_VV.tiff', 'rb') as f:
    test = f.read()

breakpoint()

print('done')

