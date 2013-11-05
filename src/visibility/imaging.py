import numpy as np
import matplotlib.pyplot as plt
import aipy as ap


uv = ap.miriad.UV('sim3.uv')
# freqs = np.array([0.8]) # GHz
# nchan = (freqs.shape)[0]
# sdf = 0.0
# sfreq = freqs[0]
# aa = ap.cal.get_aa('ant_array',freqs)
sdf = 0.01
sfreq = 0.8
nchan = 2
aa = ap.cal.get_aa('ant_array',sdf,sfreq,nchan)
ref_ra = '250:00:00'
ref_dec = '40:00:00'
ref_src = ap.fit.RadioFixedBody(ref_ra,ref_dec,name='center') # the center reference source
data, uvw, wgts = [], [], []
uv.select('auto',0,0,include=False)
for (crd,t,(i,j)),d in uv.all():
    aa.set_jultime(t)
    ref_src.compute(aa)
    try:
        d = aa.phs2src(d,ref_src,i,j)
        crd = aa.gen_uvw(i,j,src=ref_src)
    except(ap.phs.PointingError):
        continue
    uvw.append(np.squeeze(crd.compress(np.logical_not(d.mask),axis=2)))
    data.append(d.compressed())
    wgts.append(np.array([1.] * len(data[-1])))
data,uvw,wgts = np.concatenate(data),np.concatenate(uvw,axis=1),np.concatenate(wgts)
# data,uvw,wgts = np.concatenate(data),np.array(uvw).T,np.concatenate(wgts)

size = 20
res = 0.5
im = ap.img.ImgW(size=size,res=res)
uvw,data,wgts = im.append_hermitian(uvw,data,wgts=wgts)
im.put(uvw,data,wgts=wgts)
plt.figure()
plt.subplot(221)
ext1 = (0,2*size,0,2*size)
plt.imshow(np.abs(im.uv),extent=ext1,origin='lower')
plt.xlabel('N')
plt.ylabel('N')
plt.subplot(222)
plt.imshow(np.abs(im.bm[0]),extent=ext1,origin='lower')
plt.xlabel('N')
plt.ylabel('N')
plt.subplot(223)
ext2 = (-0.5/res,0.5/res,-0.5/res,0.5/res)
plt.imshow(np.log10(im.image(center=(size,size))),extent=ext2,origin='lower')
plt.xlabel('l')
plt.ylabel('m')
plt.subplot(224)
plt.imshow(np.log10(im.bm_image(center=(size,size))[0]),extent=ext2,origin='lower')
plt.xlabel('l')
plt.ylabel('m')
plt.show()


# u = [u1 for [u1,v1,w1] in uvw.T]
# v = [v2 for [ud21,v2,w2] in uvw.T]
# # print len(u),len(v)
# plt.figure()
# plt.scatter(u,v)
# plt.show()
