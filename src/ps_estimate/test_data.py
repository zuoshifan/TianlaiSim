import aipy as ap
import delay_spec as ds
import numpy as np
# import matplotlib.pyplot as plt
import pylab as pl
# import sys

# sys.argv = 'plot_uv.py -a 0 -p yy -d --chan_axis=phys -c -200_200 -x 3 test.uv'.split()
# execfile(r'scripts/plot_uv.py')

# sys.argv = 'srclist.py -P * -s all'.split()
# execfile(r'scripts/srclist.py')

# sys.argv = 'uvlist.py test.uv'.split()
# execfile(r'scripts/uvlist.py')

# don't run the following command or mk_img.py, it is very slow.
# sys.argv = 'mk_img.py -a 0_1 -p yy -C pwa303 test.uv'.split()
# execfile(r'scripts/mk_img.py')


def sort2arrays(a,b):
    """
    Sort a and the corresponding element of b .
    Arguments:
    - `a`: A array;
    - `b`: A array.
    """
    a = [var for var in a]
    b = [var for var in b]
    temp = dict(zip(a,b))
    a.sort()
    b = [temp[key] for key in a]
    return a,b

def binning(k_perpend,ps,bins,start=None,end=None):
    """
    Binning the power spectra along the k_perpendicular axes. Bins that have no corresponding K_perpendicular are filled with 0.
    Arguments:
    - `k_perpend`: The k_perpendicular array.
    - `ps`: The power spectra to be binned.
    - `bins`: Number of bins to be binned.
    - `start`: Starting point of bins, if None (the default) it will set k_perpend[0] the central value of the first bin.
    - `end`: Ending point of bins, if None (the default) it will set k_perpend[-1] the central value of the last bin.
    """
    if start == None and end == None:
        bin_width = (k_perpend[-1] - k_perpend[0]) / (bins - 1)
        start = k_perpend[0] - 0.5*bin_width
        end = k_perpend[-1] + 0.5*bin_width
    else:
        bin_width = (end - start) / bins
    bin_k_perpend = [start+n*bin_width for n in np.arange(bins+1)]
    bin_ps = [np.zeros_like(ps[0]) for n in np.arange(len(bin_k_perpend))]
    for n1 in np.arange(bins):
        temp = []
        for n2 in np.arange(len(k_perpend)):
            if bin_k_perpend[n1] <= k_perpend[n2] < bin_k_perpend[n1+1]:
                temp.append(n2)
        if temp != []:
            for n3 in temp:
                bin_ps[n1] += np.array(ps[n3])
            bin_ps[n1] /= len(temp)
    return bin_k_perpend, bin_ps

uvfile = 'test.uv'
uv = ap.miriad.UV(uvfile)
bl = []
pol = ap.miriad.pol2str[uv['pol']]
for (uvw,t,(i,j)),data in uv.all():
    if i==j or (i,j) in bl:
        continue
    else:
        bl.append((i,j))

k_perpend = []
k_paral = []
ps = []
for (i,j) in bl:
    # if (i,j) != (0,2): continue
    delay = ds.Delay_trans(uvfile,i,j,pol=pol)
    delay.get_delays()
    delay.D_trans()
    delay.plot_nth_D(o_file='figure/delay_sp_1D_%s_%s.png'%(i,j))
    delay.plot_D_spec(o_file='figure/delay_sp_%s_%s.png'%(i,j),figsize=None)
    delay.P21_est()
    delay.get_kx_ky()
    delay. get_k_perpend()
    k_perpend.append(delay.k_perpend[0])
    delay.get_kz()
    if k_paral==[]:
        k_paral.append(delay.kz)
    ps.append((delay.P21)[0]) # maybe better use the average value
k_perpend, ps = sort2arrays(k_perpend,ps)
k_perpend,ps = binning(k_perpend,ps,100)
ps = np.array(ps).T
ps = [var for var in ps]
ps.reverse()
pl.figure()
x1 = k_perpend[0]
x2 = k_perpend[-1]
y1 = k_paral[0][0]
y2 = k_paral[0][-1]
extent = (x1,x2,y1,y2)
pl.imshow(ps,extent=extent,aspect='auto')
pl.title('$P(k) [K^2 (h^{-1}Mpc)^3]$')
pl.xlabel("$k_{\perp} [hMpc^{-1}]$")
pl.ylabel("$k_{\parallel} [hMpc^{-1}]$")
pl.colorbar(shrink=0.5)
pl.savefig('figure/ps_2D.png')