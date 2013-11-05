import aipy
import numpy as np
import pylab

uv = aipy.miriad.UV('test.uv')
# Module aipy.loc now is aipy.cal
aa = aipy.cal.get_aa('pwa303',uv['sdf'],uv['sfreq'],uv['nchan'])
# src = aipy.src.get_src('vir')
# See modifications in tutorial.rst
srcs = aipy._src.misc.get_srcs(srcs=['vir'])
# print len(srcs)  # result = 1
src = srcs[0]
data, uvw, wgts = [], [], []
uv.select('auto',0,0,include=False)
# print len(list(uv.all()))  # result =3018
for(crd,t,(i,j)),d in uv.all():
    aa.set_jultime(t)
    src.compute(aa)
    try:
        d = aa.phs2src(d,src,i,j)
        crd = aa.gen_uvw(i,j,src=src)
    except aipy.phs.PointingError:
        continue
    uvw.append(np.squeeze(crd.compress(np.logical_not(d.mask), axis=2)))
    # MaskedArray.compressed(): Return all the non-masked data as a 1-D array.
    # Notes: The result is not a MaskedArray!
    data.append(d.compressed())
    # len(data[-1]) is usually not the same length, 61/62/0/64 etc
    wgts.append(np.array([1.]*len(data[-1])))
data,uvw,wgts = np.concatenate(data),np.concatenate(uvw,axis=1),np.concatenate(wgts)

# import matplotlib.pyplot as plt

# u = [u1 for [u1,v1,w1] in uvw.T]
# v = [v2 for [ud21,v2,w2] in uvw.T]
# # print len(u),len(v)
# plt.figure()
# plt.scatter(u,v)
# plt.show()
# err


im = aipy.img.ImgW(size=200,res=0.5)
# The length is 2 times than before
uvw,data,wgts = im.append_hermitian(uvw,data,wgts=wgts)
im.put(uvw,data,wgts=wgts)
pylab.subplot(221)
pylab.imshow(np.abs(im.uv))
pylab.subplot(222)
#print 'uv = ',im.uv.shape,'\nbm = ',im.bm[0].shape,'\nlen(bm) = ',len(im.bm)
# NOTE: Now im.bm is a list
pylab.imshow(np.abs(im.bm[0]))
pylab.subplot(223)
pylab.imshow(np.log10(im.image(center=(200,200))))
pylab.subplot(224)
# NOTE: Now im.bm_image returns a list
pylab.imshow(np.log10(im.bm_image(center=(200,200))[0]))
pylab.show()



