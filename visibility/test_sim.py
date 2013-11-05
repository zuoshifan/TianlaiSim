import numpy as np
import aipy as ap
# import pwa303 as pw

def get_freqs(sdf, sfreq, nchan):
    return np.arange(nchan, dtype=np.float) * sdf + sfreq

uv = ap.miriad.UV('../ps_estimate/test.uv')
freqs = get_freqs(uv['sdf'],uv['sfreq'],uv['nchan'])
# aa = pw.get_aa(freqs)
aa = ap.cal.get_aa('ant_array',freqs)
aa.select_chans([0])
aa.set_jultime(2544480.0)
cat = ap.src.get_catalog(srcs=['vir','cen'])
cat.compute(aa)
# src = cat.get_srcs('vir')
# src = src[0]
# ra_dec =src.ra,src.dec
# xyz = ap.coord.radec2eq(ra_dec)
# xyz.shape = 3,-1
# print xyz
# aa.sim_cache(xyz)
eqs = cat.get_crds('eq',ncrd=3)
flx = cat.get_jys()
aa. sim_cache(eqs,flx)
V01 = aa.sim(0,1)
print V01

# passband = aa.passband(0,1)
# bm_resp = aa.bm_response(0,1)
#aa.sim_cache

# print src
# print src.get_params()
# print aa
# print aa.get_params()
# print 'passband: ', passband
# print 'bm_resp: ',bm_resp