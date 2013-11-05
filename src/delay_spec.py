import aipy as ap
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import constant as cst


# Antenna parameters.
z = 8.5                     # Redshift
Omega = 0.31                # Beam area, in unit sr. See Parsons et al. 2013 Eq.(B10)


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


class Delay_trans():
    """
    A Per-Baseline, Delay-Spectrum Technique for 21 cm power spectrum estimate.
    """
    
    def __init__(self, uv_file,bl_i,bl_j,pol='xx',window='blackman-harris',tol=1e-3,mask=True):
        """
        Initialize.
        Arguments:
        - `uv_file`: The miriad UV file contains data to processing.
        - `bl_i`: Baseline i, indexed from zero;
        - `bl_j`: Baseline j, indexed from zero.
        - `pol`: Polarization, can be 'xx','yy','xy','yx'.
        - `window`: Windowing function to use in delay transform. Default is blackman-harris. Options are: 'blackman','blackman-harris','gaussian0.4','kaiser2','kaiser3','hamming','hanning','parzen','none'. See aipy.dsp.WINDOW_FUNC.keys().
        - `tol`: Specify a tolerance for termination (usually 1e-2 or 1e-3).
        - `mask`: Use masked data or not, default is mask = True.
        """
        self.uv = ap.miriad.UV(uv_file)
        self.bl_i = bl_i
        self.bl_j = bl_j
        self.pol = pol
        self.window = window
        self.tol = tol
        self.mask = mask
        self.nchan = self.uv['nchan'] # number of channels
        self.sfreq = self.uv['sfreq'] # starting frequency in GHz
        self.sdf = self.uv['sdf']     # channel width in GHz

    def set_bl(self,bl_i,bl_j):
        """
        Select baseline.
        Arguments:
        - `bl_i`: Baseline i;
        - `b;_j`: Baseline j.
        """
        self.bl_i = bl_i
        self.bl_j = bl_j

    def set_pol(self,pol):
        """
        Select polarization.
        Arguments:
        - `self`:
        - `pol`: Polarization.
        """
        self.pol = pol

    def set_window(self,window):
        """
        Set window function to use in delay transform. If you call this function, it must be called before function D_trans().
        Arguments:
        - `self`:
        - `window`: Window function.
        """
        self.window = window

    def set_tol(self,tol):
        """
        Set the tolerance for termination of the CLEAN (usually 1e-2 or 1e-3).
        Arguments:
        - `self`:
        - `tol`: Tolerance.
        """
        self.tol = tol

    def set_mask(self,mask):
        """
        Choose masked/unmasked data to use.
        Arguments:
        - `self`:
        - `mask`: True or Faulse. Use masked data or not.
        """
        self.mask = mask

    def set_samp_wgt(self,nchan):
        """
        Set the frequency-dependent sample weights that result from RFI flagging.
        Arguments:
        - `self`:
        - `nchan`: number of frequency channels.
        """
        return np.array([1.0]*nchan)

    def D_trans(self):
        """
        Per-baseline delay transform and CLEAN the delay-transformed visibility.
        Arguments:
        - `self`:
        """
        self.uvw = []
        self.times = []
        self.uncl_vis_tude = [] # save the unCLEANed delay-transformed visibility
        self.cl_vis_tude = [] # Save the CLEANed delay-transformed visibility
        self.uv.rewind()
        self.uv.select('antennae',self.bl_i,self.bl_j,include=True) # select baseline
        try:
            pol = ap.miriad.str2pol[self.pol] # select polarization
        except(KeyError):
            raise ValueError('--pol argument invalid or absent')
        self.uv.select('polarization', pol, 0)
        for (uvw,t,(i,j)),d in self.uv.all():
            # d is the observed visibility, in unit W m^-2 Hz^-1
            self.uvw.append(uvw)
            self.times.append(t)
            w = ap.dsp.gen_window(d.shape[-1],window=self.window) # generate window function
            s = self.set_samp_wgt(d.shape[-1]) # the frequency -dependent sample weight
            if self.mask:
                flags = np.logical_not(d.mask).astype(np.float)
                gain = np.sqrt(np.average(flags**2))
                ker = np.fft.fft(flags*w)
                d = d.filled(0)
            else:
                d = d.data
                ker = np.zeros_like(d)
                ker[0] = 1.0
                gain = 1.0
            d = np.fft.fft(d*w*s) # delay transform
            # after delay transform, the unit is W m^-2
            self.uncl_vis_tude.append(np.fft.fftshift(d))
            if not np.all(d==0):
                d, info =ap.deconv.clean(d,ker,tol=self.tol) # CLEAN
                d += info['res'] / gain
            self.cl_vis_tude.append(np.fft.fftshift(d))

    def get_delays(self):
        """
        Get the delays (unit: ns) of the per-baseline delay transform.
        """
        self.delays = np.fft.fftfreq(self.nchan,self.sdf)
        self.delays = np.fft.fftshift(self.delays)

    def bl_len(self,uvw):
        """
        Return the length of the baseline in unit of wavelength.
        Arguments:
        - `self`:
        - `uvw`: A tuple of (u,v,w).
        """
        return np.sqrt(uvw[0]**2+uvw[1]**2+uvw[2]**2)

    def h_limit(self,nrecord):
        """
        Get the horizon limit in unit ns.
        Arguments:
        - `self`:
        - `nrecord`: Nth (Starting from 0) record in the selected visibility data.
        """
        return self.bl_len(self.uvw[nrecord])/self.sfreq

    def plot_nth_D(self,nrecord=0,mode='log',h_limit=True,clean='both',figsize=None,o_file=''):
        """
        Plot Nth (Starting from 0) 1D delay spectrum.
        Arguments:
        - `self`:
        - `nrecord`: Nth (Starting from 0) record in the selected visibility data.
        - `mode`: Plot mode can be log (logrithmic), lin (linear), phs (phase), real, or imag. Default is log.
        - `h_limit`: Whether or not display the horizon limit, defualt is h_limit = True, i.e. display.
        - 'clean': Plot the CLEANed/unCLEANed/both 1D delay spectrum. Options are: 'both','clean','unclean'. Default is 'both'.
        - 'figsize': Width and height of the output figure, e.g. figsize=(8,4)
        - `o_file`: If provided, will save the figure to the specified file instead of popping up a window.
        """
        plt.figure(figsize=figsize)
        xlabel = 'delay (ns)'
        ylabel = 'power'
        x = self.delays
        y_cl = data_mode(np.ma.array(self.cl_vis_tude[nrecord]),mode=mode)
        y_uncl = data_mode(np.ma.array(self.uncl_vis_tude[nrecord]),mode=mode)
        if clean.startswith('clean'):
            plt.plot(x,y_cl)
            plt.title('CLEANed delay spectrum for baseline %s,%s'%(self.bl_i,self.bl_j))
        elif clean.startswith('unclean'):
            plt.plot(x,y_uncl)
            plt.title('UnCLEANed delay spectrum for baseline %s,%s'%(self.bl_i,self.bl_j))
        elif clean.startswith('both'):
            plt.plot(x,y_cl,label='CLEANed')
            plt.plot(x,y_uncl,label='UnCLEANed')
            plt.title('Delay spectrum for baseline %s,%s'%(self.bl_i,self.bl_j))
            plt.legend()
        else:
            raise ValueError('Unrecognized plot type.')
        plt.xlabel(xlabel);plt.ylabel(ylabel)
        # Save to a file or pop up a window
        if o_file != '':
            plt.savefig(o_file)
        else:
            plt.show()

    def plot_D_spec(self,mode='log',clean='both',figsize=None,interp='nearest',o_file=''):
        """
        Plot the delay spectrum.
        Arguments:
        - `self`:
        - 'mode','clean','figsize','o_file': See corresponding argument in function plot_nth_D.
        - 'interp': Acceptable values are *None*, 'none', 'nearest', 'bilinear','bicubic', 'spline16', 'spline36', 'hanning', 'hamming','hermite', 'kaiser', 'quadric', 'catrom', 'gaussian','bessel', 'mitchell', 'sinc', 'lanczos'.
        """
        pl.figure(figsize=figsize)
        xlabel = 'delay (ns)'
        ylabel = 'time (integrations)'
        x1 = self.delays[0]
        x2 = self.delays[-1]
        t1 = 0
        t2 = len(self.times)
        extent = (x1,x2,t2,t1)
        y_cl = data_mode(np.ma.array(self.cl_vis_tude),mode=mode)
        y_uncl = data_mode(np.ma.array(self.uncl_vis_tude),mode=mode)
        if clean.startswith('clean'):
            pl.imshow(y_cl,extent=extent,aspect='auto',interpolation=interp)
            pl.title('CLEANed delay spectrum for baseline %s,%s'%(self.bl_i,self.bl_j))
            pl.xlabel(xlabel); pl.ylabel(ylabel)
            pl.colorbar(shrink=0.5)
        elif clean.startswith('unclean'):
            pl.imshow(y_uncl,extent=extent,aspect='auto',interpolation=interp)
            pl.title('UnCLEANed delay spectrum for baseline %s,%s'%(self.bl_i,self.bl_j))
            pl.xlabel(xlabel); pl.ylabel(ylabel)
            pl.colorbar(shrink=0.5)
        elif clean.startswith('both'):
            pl.subplot(121)
            pl.imshow(y_cl,extent=extent,aspect='auto',interpolation=interp)
            pl.title('CLEANed delay spectrum \n %s,%s'%(self.bl_i,self.bl_j))
            pl.xlabel(xlabel); pl.ylabel(ylabel)
            pl.colorbar(shrink=0.5)
            pl.subplot(122)
            pl.imshow(y_uncl,extent=extent,aspect='auto',interpolation=interp)
            pl.title('UnCLEANed delay spectrum \n %s,%s'%(self.bl_i,self.bl_j))
            pl.xlabel(xlabel); pl.ylabel(ylabel)
            pl.colorbar(shrink=0.5)
        else:
            raise ValueError('Unrecognized plot type.')
        # Save to a file or pop up a window
        if o_file != '':
            plt.savefig(o_file)
        else:
            pl.show()

    def sqr_cl_vis_tude(self):
        """
        Return cl_vis_tude*conjugate(cl_vis_tude), a real float array.
        Arguments:
        - `self`:
        """
        vis_tude_2 = []
        for index in range(len(self.cl_vis_tude)):
            vis_tude_2.append([np.abs(vis)**2 for vis in self.cl_vis_tude[index]])
        return vis_tude_2 # unit: W^2 m^-4

    def X(self):
        """
        Return the conversion factor X from angle to comoving distance.
        Arguments:
        - `self`:
        """
        return 1.9 * ((1+z)/10.0)**0.2 / cst.arcmin # in unit h^-1 Mpc rad^-1

    def Y(self):
        """
        Return the conversion factor Y from frequency to comoving distance.
        Arguments:
        - `self`:
        """
        return 17.0 * np.sqrt(0.015*(1+z)/cst.Omega_m) * 1.0e-6 # in unit h^-1 Mpc Hz^-1

    def P21_est(self):
        """
        Calculate the Power Spectrum P_21(k) in unit K^2 (h^-1 Mpc)^3 according to the formula Eq.(12) given by Parsons, arXiv:1103.2135v2.
        Arguments:
        - `self`:
        """
        freqs = 1.0e9*np.arange(self.uv['sfreq'], self.uv['sfreq']+self.uv['nchan']*self.uv['sdf'], self.uv['sdf']) # in unit Hz
        wavelen = cst.c / freqs # in unit m
        # XXY = X^2*Y in unit h^-3 Mpc^3 sr^-1 Hz^-1
        XXY = self.X()**2 * self.Y()
        B = 1.0e9*self.uv['sdf']*self.uv['nchan'] # Bandwidth, in unit Hz
        vis_tude_sqr = self.sqr_cl_vis_tude()
        self.P21 = []
        # Calculate the Power Spectrum P_21(k).
        for index in np.arange(len(self.cl_vis_tude)):
            self.P21.append([vis_tude_sqr[index][n]*(wavelen[n]**4 * XXY) / (4 * cst.k_B**2 * Omega * B) for n in np.arange(self.uv['nchan'])])

    def get_kx_ky(self):
        """
        Get the k_x and k_y components of k vector. Unit: h Mpc^-1.
        Arguments:
        - `self`:
        """
        self.kx_ky = []
        for index in np.arange(len(self.uvw)):
            self.kx_ky.append([self.uvw[index][0], self.uvw[index][1]])
        self.kx_ky = np.array(self.kx_ky)
        self.kx_ky *= (2*np.pi / self.X())

    def get_k_perpend(self):
        """
        Get the k_perpendicular component of k vector. Unit: h Mpc^-1.
        Arguments:
        - `self`:
        """
        self.k_perpend = [np.sqrt(self.kx_ky[index][0]**2 + self.kx_ky[index][1]**2) for index in np.arange(len(self.kx_ky))]
        self.k_perpend = np.array(self.k_perpend)

    def get_kz(self):
        """
        Get the k_z component of k vector. Unit: h Mpc^-1.
        Arguments:
        - `self`:
        """
        self.kz = 1.0e-9*self.delays *(2*np.pi / self.Y()) # 1.0e-9 convert the unit of self.delays from ns to s
        # self.kz -= np.min(self.kz) # minimem k_z starts from zero

    def plot_ps2k_perpend(self):
        """
        Plot power spectra vs k_perpendicular by times sequence.
        Arguments:
        - `self`:
        """
        xlabel = 'k_perpen [h Mpc^-1]'
        ylabel = 'time (integrations)'
        x1 = self.kz[0]
        x2 = self.kz[-1]
        t1 = 0
        t2 = len(self.times)
        extent = (x1,x2,t2,t1)
        y = np.array(self.P21)
        pl.imshow(y,extent=extent,aspect='auto')
        pl.colorbar(shrink=0.5)
        pl.xlabel(xlabel); pl.ylabel(ylabel)
        pl.show()

    def plot_2d_ps(self):
        """
        Plot the two-dimensional power spectrum.
        Arguments:
        - `self`:
        """
        xlabel = 'k_perpen [h Mpc^-1]'
        ylabel = 'k_par [h Mpc^-1]'
        y = np.array(self.P21).T
        pl.imshow(y,aspect='auto')
        pl.colorbar(shrink=0.5)
        pl.xlabel(xlabel); pl.ylabel(ylabel)
        pl.show()

    def plot_ps(self):
        """
        Plot the power spectrum.
        Arguments:
        - `self`:
        """
        plt.figure(figsize=None)
        xlabel = 'k [h Mpc^-1]'
        ylabel = 'P(k) [mK^2 (h^-1 Mpc)^3  ]'
        k =[]    # |k|
        Pk = []  # P(k)
        for n1 in np.arange(self.kx_ky.shape[0]):
            for n2 in np.arange(self.kz.shape[0]):
                k.append(np.sqrt(self.kx_ky[n1][0]**2 + self.kx_ky[n1][1]**2 + self.kz[n2]**2)) # unit: h Mpc^-1
                Pk.append(self.P21[n1][n2] * 1.0e6) # unit: mK^2 (h^-1 Mpc)^3
        
        plt.plot(k,Pk,'o')
        plt.xlabel(xlabel);plt.ylabel(ylabel)
        plt.show()
        

# Dt = Delay_trans('test.uv',0,2)
# Dt.get_delays()
# Dt.D_trans()
# Dt.get_kx_ky()
# Dt. get_k_perpend()
# print len(Dt.k_perpend)
# print Dt.k_perpend
# Dt.plot_nth_D(o_file='figure/test.png')
# Dt.plot_D_spec(o_file='figure/delay_sp.png')
# Dt.plot_D_spec()
# Dt.P21_est()
# Dt.get_kx_ky()
#Dt.get_kz()
# Dt.plot_2d_ps()
# Dt.plot_ps()
# Dt.plot_ps2k_perpend()