import numpy as np
import matplotlib.pyplot as plt
import aipy as ap
import constant as cst


def data_mode(data, mode='lin'):
    """Plot mode. Can be log (logrithmic), lin (linear), phs (phase), real, or imag."""
    if mode.startswith('phs'):
        data = np.angle(data.filled(0)) # Return the angle of the complex argument, in unit: radian
    elif mode.startswith('lin'):
        data = np.ma.absolute(data.filled(0)) # Abs of the data
        data = np.ma.masked_less_equal(data, 0)
    elif mode.startswith('real'):
        data = data.real
    elif mode.startswith('imag'):
        data = data.imag
    elif mode.startswith('log'):
        data = np.ma.absolute(data.filled(0))
        data = np.ma.masked_less_equal(data, 0)
        data = np.ma.log10(data)
    else:
        raise ValueError('Unrecognized plot mode.')
    return data


uv = ap.miriad.UV('../test.uv')
sdf = uv['sdf']
sfreq = uv['sfreq']
nchan = uv['nchan']
aa = ap.cal.get_aa('pwa303',sdf,sfreq,nchan)
nants = len(aa.ants)
vis = [] # save visibility data
uvw = [] # save uvw coordinate, dimension 1
try:
    pol = ap.miriad.str2pol['yy'] # select polarization
except(KeyError):
    raise ValueError('--pol argument invalid or absent')
uv.select('polarization', pol, 0) # select yy polarization
# uv.select('antennae',0,1,include=True)
uv.select('auto',0,1,include=False) # select all cross correlation
for (pos,t,(i,j)),data in uv.all():
    aa.set_jultime(t)
    data = aa.phs2src(data,'z',i,j) # should not change when pointint to the zenith, test it
    crd = aa.gen_uvw(i,j,src='z')
    # vis.append(data.compressed())
    # uvw.append(np.squeeze(crd.compress(np.logical_not(data.mask),axis=2)))
    vis.append(data)
    uvw.append(np.squeeze(crd))
# print len(vis),len(uvw)
# print vis[0], uvw[0]
assert len(vis) == len(uvw), 'len(vis) not equal to len(uvw)!'
V_tude = [] # delay transform of visibility
w = ap.dsp.gen_window(vis[0].shape[-1],window='blackman-harris') # generate window function
s = np.ones_like(w) # the frequency -dependent sample weight
for V in vis:
    flags = np.logical_not(V.mask).astype(np.float) # mask invalid visibility data in some frequency channels
    gain = np.sqrt(np.average(flags**2))
    ker = np.fft.fft(flags*w)
    V = V.filled(0)
    dsp = np.fft.fft(V*w*s) # delay spectrum
    V_tude.append(np.fft.fftshift(dsp))
# get the delays in unit ns
delays = np.fft.fftfreq(nchan,sdf)
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
freqs = 1.0e9*np.arange(sfreq, sfreq+nchan*sdf, sdf) # in unit Hz
wavelen = cst.c / freqs # in unit m
# Antenna parameters.
z = 8.5                     # Redshift
Omega = 0.31                # Beam area, in unit sr. See Parsons et al. 2013 Eq.(B10)
X = 1.9 * ((1+z)/10.0)**0.2 / cst.arcmin # in unit h^-1 Mpc rad^-1
Y = 17.0 * np.sqrt(0.015*(1+z)/cst.Omega_m) * 1.0e-6 # in unit h^-1 Mpc Hz^-1
XXY = X*X*Y
B = 1.0e9*sdf*nchan # Bandwidth, in unit Hz
V_tude2 = [] # |V_tilde|^2
for index in range(len(V_tude)):
    V_tude2.append([np.abs(V)**2 for V in V_tude[index]])
P21 = []
# Calculate the Power Spectrum P_21(k).
for index in np.arange(len(V_tude2)):
    P21.append([V_tude2[index][n]*(wavelen[n]**4 * XXY) / (4 * cst.k_B**2 * Omega * B) for n in np.arange(nchan)])
# kx_ky = [uvw[0]*(2*np.pi / X),uvw[1]*(2*np.pi / X)]
kx_ky = []
for var in uvw:
    kx_ky.append([var[0]*(2*np.pi / X),var[1]*(2*np.pi / X)]) # unit: h Mpc^-1
k_perp = []
k_para = []
for var in kx_ky:
    k_perp.append(np.sqrt(var[0]**2 + var[1]**2)) # unit: h Mpc^-1
# print len(k_perp)
# print k_perp[0]
k_perp = np.array(k_perp)
k_perp_max = np.max(k_perp)
k_perp_min = np.min(k_perp)
# k_perp_bins = 100
k_perp_bins = 100
k_perp_spc = (k_perp_max - k_perp_min) / (k_perp_bins - 1) # k_perp spacing
k_perp_ax = np.linspace(k_perp_min-k_perp_spc/2,k_perp_max+k_perp_spc/2,k_perp_bins) # k_perp axis
k_z = 1.0e-9*delays*(2*np.pi / Y) # 1.0e-9 convert the unit of self.delays from ns to s. Unit: h Mpc^-1
# Now transpose k_perp and P21
k_perp = k_perp.T
P21 = np.array(P21).T
# Now sort k_perp and P21
for index in range(nchan):
    k_perp_idx = np.argsort(k_perp[index])
    k_perp[index] = k_perp[index].take(k_perp_idx)
    P21[index] = P21[index].take(k_perp_idx)
# Now reshape P21 to a new matrix P21k nchan x k_perp_bins
P21k = np.zeros((nchan, k_perp_bins))
col_k_perp = (k_perp.shape)[1] # columns of k_perp
for row in range(nchan):
    col = 0
    for bins in range(k_perp_bins-1):
        # col = 0
        sums = 0.0
        count = 0
        while col < col_k_perp and k_perp_ax[bins] <= k_perp[row][col] and k_perp[row][col] < k_perp_ax[bins+1]:
            count += 1
            sums += P21[row][col]
            col += 1
        if count >1:
            P21k[row][bins] = sums / count # average over bins
# plot the 2d power spectrum
plt.figure(figsize=(8,6))
ext = (min(k_perp_ax),max(k_perp_ax),min(k_z),max(k_z))
P21k = np.ma.masked_less_equal(P21k, 0) # mask all the void data
plt.imshow(P21k,extent=ext,aspect='auto',origin='lower')
plt.colorbar(shrink=0.75)
plt.xlabel('k_perp')
plt.ylabel('k_para')
plt.show()