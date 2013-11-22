import numpy as np
import matplotlib.pyplot as plt
import aipy as ap
import constant as cst


def data_mode(data, mode='abs'):
    if mode.startswith('phs'): data = np.angle(data.filled(0))
    elif mode.startswith('lin'):
        data = np.ma.absolute(data.filled(0))
        data = np.ma.masked_less_equal(data, 0)
    elif mode.startswith('real'): data = data.real
    elif mode.startswith('imag'): data = data.imag
    elif mode.startswith('log'):
        data = np.ma.absolute(data.filled(0))
        data = np.ma.masked_less_equal(data, 0)
        data = np.ma.log10(data)
    else: raise ValueError('Unrecognized plot mode.')
    return data


uv = ap.miriad.UV('../test.uv')
aa = ap.cal.get_aa('pwa303',uv['sdf'],uv['sfreq'],uv['nchan'])
nants = len(aa.ants)
vis = [] # save visibility data
uvw = [] # save uvw coordinate, dimension 1
try:
    pol = ap.miriad.str2pol['yy'] # select polarization
except(KeyError):
    raise ValueError('--pol argument invalid or absent')
uv.select('polarization', pol, 0) # select yy polarization
uv.select('antennae',0,1,include=True)
for (pos,t,(i,j)),data in uv.all():
    aa.set_jultime(t)
    data = aa.phs2src(data,'z',i,j) # should not change, test it
    crd = aa.gen_uvw(i,j,src='z')
    # vis.append(data.compressed())
    # uvw.append(np.squeeze(crd.compress(np.logical_not(data.mask),axis=2)))
    vis.append(data)
    uvw.append(crd)
# print len(vis),len(uvw)
# print vis[0], uvw[0]
assert len(vis) == len(uvw), 'len(vis) not equal to len(uvw)!'
V_tude = [] # delay transform of visibility
w = ap.dsp.gen_window(vis[0].shape[-1],window='blackman-harris') # generate window function
s = np.ones_like(w) # the frequency -dependent sample weight
for V in vis:
    flags = np.logical_not(V.mask).astype(np.float)
    gain = np.sqrt(np.average(flags**2))
    ker = np.fft.fft(flags*w)
    V = V.filled(0)
    dsp = np.fft.fft(V*w*s) # delay spectrum
    V_tude.append(np.fft.fftshift(dsp))
# get the delays in unit ns
delays = np.fft.fftfreq(uv['nchan'],uv['sdf'])
delays = np.fft.fftshift(delays)
# # plot delay spectrum
# plt.figure(figsize=(8,6))
# ext = (delays[0],delays[-1],0,len(V_tude))
# # plt.imshow(np.abs(V_tude),extent=ext,aspect='auto',origin='lower')
# plt.imshow(data_mode(np.ma.array(V_tude),mode='log'),extent=ext,aspect='auto',origin='lower')
# plt.title('Delay spectrum')
# plt.xlabel('delays (ns)')
# plt.ylabel('time (integrations)')
# plt.show()

# power spectrum estimate
freqs = 1.0e9*np.arange(uv['sfreq'], uv['sfreq']+uv['nchan']*uv['sdf'], uv['sdf']) # in unit Hz
wavelen = cst.c / freqs # in unit m
# Antenna parameters.
z = 8.5                     # Redshift
Omega = 0.31                # Beam area, in unit sr. See Parsons et al. 2013 Eq.(B10)
X = 1.9 * ((1+z)/10.0)**0.2 / cst.arcmin # in unit h^-1 Mpc rad^-1
Y = 17.0 * np.sqrt(0.015*(1+z)/cst.Omega_m) * 1.0e-6 # in unit h^-1 Mpc Hz^-1
XXY = X*X*Y
B = 1.0e9*uv['sdf']*uv['nchan'] # Bandwidth, in unit Hz
V_tude2 = [] # |V_tilde|^2
for index in range(len(V_tude)):
    V_tude2.append([np.abs(V)**2 for V in V_tude[index]])
P21 = []
# Calculate the Power Spectrum P_21(k).
for index in np.arange(len(V_tude2)):
    P21.append([V_tude2[index][n]*(wavelen[n]**4 * XXY) / (4 * cst.k_B**2 * Omega * B) for n in np.arange(uv['nchan'])])
# kx_ky = [uvw[0]*(2*np.pi / X),uvw[1]*(2*np.pi / X)]
kx_ky = []
for var in uvw:
    kx_ky.append([var[0]*(2*np.pi / X),var[1]*(2*np.pi / X)]) # unit: h Mpc^-1
k_perp = []
k_para = []
for var in kx_ky:
    k_perp.append(np.sqrt(var[0]**2 + var[1]**2)) # unit: h Mpc^-1
print len(k_perp)
print k_perp[0]
k_z = 1.0e-9*delays*(2*np.pi / Y) # 1.0e-9 convert the unit of self.delays from ns to s. Unit: h Mpc^-1
# k_para = [var for var in k_z]

