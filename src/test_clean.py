import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
import aipy as ap

def Vfunc(nu):
    """
    Return visibility.
    Arguments:
    - `nu`:
    """
    return np.exp(-nu**2) # Gaussian (0,1.0)

nchan = 2**10
ending = 20.0 # GHz
space = 2.0*ending / nchan # GHz

# V
nu = np.linspace(-ending,ending,nchan) # GHz
V = [np.exp(-x**2) for x in nu]
Vtilde = np.fft.fft(V)
# Vtilde = np.fft.fftshift(Vtilde)
delay = np.fft.fftfreq(nchan,space) # ns
# delay = np.fft.fftshift(delay) # ns

# masked V
mask= [0.75<x<0.85 for x in V]
mV = ma.array(V,mask=mask)
mV = mV.filled(0)

# masked V delay tansform
mVtilde = np.fft.fft(mV)
# mVtilde = np.fft.fftshift(mVtilde)

# ifft of masked mVtilde
mVi = np.fft.ifft(mVtilde)
# mVi = np.fft.fftshift(mVi)

# CLEAN masked V
w = 1.0
flags = np.logical_not(mask).astype(np.float)
gain = np.sqrt(np.average(flags**2))
ker = np.fft.fft(flags*w)
d = np.fft.fft(mV*w)
# CLEAN before or after fftshift?
# Here no fftshift
d, info = ap.deconv.clean(d,ker,tol=1e-3)
d += info['res'] / gain
mVtilde_w = d
# mVtilde_w = np.fft.fftshift(d)

# ifft of CLEANed mVtilde_w
mV_wi = np.fft.ifft(mVtilde_w)
# mV_wi = np.fft.fftshift(mV_wi)


# plot figure
plt.subplot(311)
plt.plot(nu,V)
plt.plot(nu,mV)
plt.plot(nu,np.abs(mVi),'+')
plt.plot(nu,np.abs(mV_wi),'.')
plt.xlim(-3,3)
plt.ylim(0,1.1)
plt.subplot(312)
plt.subplot(313)
plt.plot(delay,np.abs(Vtilde))
plt.plot(delay,np.abs(mVtilde))
plt.plot(delay,np.abs(mVtilde_w),'.')
plt.xlim(-10/ending,10/ending)
plt.show()