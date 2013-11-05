"""
Module for 21cm power spectrum estimation.
"""

import aipy
import numpy as np
import pylab as pl  #, math, sys, optparse
import math


def convert_arg_range(arg):
    """Split apart command-line lists/ranges into a list of numbers."""
    arg = arg.split(',')
    return [map(float, option.split('_')) for option in arg]

def gen_chans(chanopt, uv, coords, is_delay):
    """Return an array of active channels and whether or not a range of
    channels is selected (as opposed to one or more individual channels)
    based on command-line arguments."""
    is_chan_range = True
    if chanopt == 'all': chans = np.arange(uv['nchan'])
    else:
        chanopt = convert_arg_range(chanopt)
        if coords != 'index':
            if is_delay:
                def conv(c):
                    return int(np.round(c * uv['sdf'] * uv['nchan'])) \
                        + uv['nchan']/2
            else:
                def conv(c): return int(np.round((c - uv['sfreq']) / uv['sdf']))
        else:
            if is_delay:
                def conv(c): return int(c) + uv['nchan']/2
            else:
                def conv(c): return c
        chanopt = [map(conv, c) for c in chanopt]
        if len(chanopt[0]) != 1: 
            chanopt = [np.arange(x,y, dtype=np.int) for x,y in chanopt]
        else: is_chan_range = False
        chans = np.concatenate(chanopt)
    return chans.astype(np.int), is_chan_range

def gen_times(timeopt, uv, coords, decimate, is_fringe):
    is_time_range = True
    if timeopt == 'all' or is_fringe:
        def time_selector(t, cnt): return True
    else:
        timeopt = convert_arg_range(timeopt)
        if len(timeopt[0]) != 1:
            def time_selector(t, cnt):
                if coords == 'index': t = cnt
                for opt in timeopt:
                    if (t >= opt[0]) and (t < opt[1]): return True
                return False
        else:
            is_time_range = False
            timeopt = [opt[0] for opt in timeopt]
            inttime = uv['inttime'] / aipy.const.s_per_day * decimate
            def time_selector(t, cnt):
                if coords == 'index': return cnt in timeopt
                for opt in timeopt:
                    if (t >= opt) and (t < opt + inttime): return True
                return False
    return time_selector, is_time_range

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



uvfile = 'test.uv'
ant_str = '0_1'
pol_str = 'yy'
chan_axis = 'phys'
chan = 'all'
time = '-200_200'
is_delay = True
# 'Windowing function to use in delay transform.  Default is kaiser3.  Options are: ' + ', '.join(a.dsp.WINDOW_FUNC.keys()))
window = 'kaiser3'
unmask = False # data is masked
clean = 1e-3
cal = 'pwa303'
cmap = 'jet'
out_file = ''
time_axis = 'index'
decimate =1
is_fringe = False
mode = 'log'

cmap = pl.get_cmap(cmap)
uv = aipy.miriad.UV(uvfile)
aipy.scripting.uv_selector(uv, ant_str, pol_str)
chans, is_chan_range = gen_chans(chan, uv, chan_axis, is_delay)
freqs = np.arange(uv['sfreq'], uv['sfreq']+uv['nchan']*uv['sdf'], uv['sdf'])
freqs = freqs.take(chans)
delays = np.arange(-.5/uv['sdf'], .5/uv['sdf'], 1/(uv['sdf']*uv['nchan']))
delays = delays.take(chans)
time_sel, is_time_range = gen_times(time, uv, time_axis, 
    decimate, is_fringe)
inttime = uv['inttime'] * decimate
#del(uv)

# Loop through UV files collecting relevant data
plot_x = {}
plot_t = {'jd':[], 'lst':[], 'cnt':[]}
times = []

# Hold plotting handles
plots = {}
plt_data = {}

aa = aipy.cal.get_aa(cal, uv['sdf'], uv['sfreq'], uv['nchan'])
for (uvw,t,(i,j)),d in uv.all():
        bl = '%d,%d' % (i,j)
        # Implement Decimation
        if len(times) == 0 or times[-1] != t:
            times.append(t)
            use_this_time = ((len(times) - 1) % decimate) == 0
            use_this_time &= time_sel(t, (len(times)-1) / decimate)
            if use_this_time:
                if aa == None: lst = uv['lst']
                else:
                    aa.set_jultime(t)
                    lst = aa.sidereal_time()
                plot_t['lst'].append(lst)
                plot_t['jd'].append(t)
                plot_t['cnt'].append((len(times)-1) / decimate)
        # Do delay transform if required
        if is_delay:
            w = aipy.dsp.gen_window(d.shape[-1], window=window)
            if unmask:
                d = d.data   # return unmasked array
                ker = np.zeros_like(d)
                ker[0] = 1.
                gain = 1.
            else:
                flags = np.logical_not(d.mask).astype(np.float)
                gain = np.sqrt(np.average(flags**2))
                ker = np.fft.ifft(flags*w)
                # numpy.ma.filled(a, fill_value=None)
                # Return input as an array with masked data replaced by a fill value.
                d = d.filled(0)
            d = np.fft.ifft(d*w)
            if not clean is None and not np.all(d == 0):
                d, info = aipy.deconv.clean(d, ker, tol=clean)
                d += info['res'] / gain
            d = np.ma.array(d)
            d = np.ma.concatenate([d[d.shape[0]/2:], d[:d.shape[0]/2]], 
                axis=0)
        # Extract specific channels for plotting
        d = d.take(chans)
        d.shape = (1,) + d.shape
        if not plot_x.has_key(bl): plot_x[bl] = []
        plot_x[bl].append(d)
del(uv)

bls = plot_x.keys()

# Generate all the plots
dmin,dmax = None, None
for cnt, bl in enumerate(bls):
    d = np.ma.concatenate(plot_x[bl], axis=0)
    plt_data[cnt+1] = d
    d = data_mode(d, mode)
    if time_axis == 'index':
        t1,t2 = plot_t['cnt'][0], plot_t['cnt'][-1]
        ylabel = 'Time (integrations)'
    else:
        t1,t2 = plot_t['jd'][0], plot_t['jd'][-1]
        ylabel = 'Time (Julian Date)'
    if is_delay:
        if chan_axis == 'index':
            c1,c2 = len(chans)/2 - len(chans), len(chans)/2
            xlabel = 'Delay (bins)'
        else:
            c1,c2 = delays[0], delays[-1]
            xlabel = 'Delay (ns)'
    dmax = d.max()
    dmin = d.min()
    plots[cnt+1] = pl.imshow(d, extent=(c1,c2,t2,t1), 
        aspect='auto', interpolation='nearest', 
        vmax=dmax, vmin=dmin, cmap=cmap)
    pl.colorbar(shrink=0.5)
    pl.xlabel(xlabel); pl.ylabel(ylabel)


    pl.legend(loc='best')

# Save to a file or pop up a window
if out_file != '': pl.savefig(out_file)
else:
    def click(event):
        print [event.key]
        if event.key == 'm':
            mode = raw_input('Enter new mode: ')
            for k in plots:
                try:
                    d = data_mode(plt_data[k], mode)
                    plots[k].set_data(d)
                except(ValueError):
                    print 'Unrecognized plot mode'
            pl.draw()
        elif event.key == 'd':
            max = raw_input('Enter new max: ')
            try: max = float(max)
            except(ValueError): max = None
            drng = raw_input('Enter new drng: ')
            try: drng = float(drng)
            except(ValueError): drng = None
            for k in plots:
                _max,_drng = max, drng
                if _max is None or _drng is None:
                    d = plots[k].get_array()
                    if _max is None: _max = d.max()
                    if _drng is None: _drng = _max - d.min()
                plots[k].set_clim(vmin=_max-_drng, vmax=_max)
            print 'Replotting...'
            pl.draw()
    pl.connect('key_press_event', click)
    pl.show()
