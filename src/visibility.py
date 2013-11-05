import numpy as np
# import matplotlib.pyplot as plt
import aipy as ap
import constant as cst
# import lm

# nchan = 64
# # nchan = 4
# sdf = 1.0 / nchan # GHz
# sfreq = 1.0 # GHz
# aa = ap.cal.get_aa('ant_array',sdf,sfreq,nchan)
# aa.set_ephemtime('2013/6/1 12:00') # observing time
# aa.select_chans([0]) # Select which channels are used in computations
# src = ap.fit.RadioFixedBody('250:00','40:00',mfreq=1.42,name='center') # ra,dec
# src.compute(aa) # Update coords relative to the provided observer/antenna array
# nants = len(aa.ants)
# # bl = [(i,j) for i in range(nants) for j in range (i+1,nants)] # all the baseline
# bl = [(i,j) for i in range(nants) for j in range (nants)]
# print len(bl)
# uvw = []
# for i,j in bl:
#     uvw.append(np.squeeze(aa.gen_uvw(i,j,src=src)))
# u = [u1 for [u1,v1,w1] in uvw]
# v = [v2 for [u2,v2,w2] in uvw]
# plt.figure(figsize=(8,6))
# plt.scatter(u,v)
# plt.axes().set_aspect('equal', 'datalim')
# plt.xlabel('u (wavelength)')
# plt.ylabel('v (wavelength)')
# plt.title('Instantaneous uv coverage')
# # plt.savefig('figure/png/AntennaArray.png')
# # plt.savefig('figure/eps/AntennaArray.eps')
# plt.show()






# # const
# c = 3.0e8 # m/s
# d = 2     # m antenna diameter
# freqs = np.array([1.4])
# w = c/(d*freqs[len(freqs)/2]*1.0e9) # beam width in angular coord, lambda/d
# beam = ap.fit.Beam2DGaussian(freqs,w,w)
# ants = []
# ants.append(ap.fit.Antenna(0,0,0,beam,phsoff=[0,0]))
# ants.append(ap.fit.Antenna(0,100,0,beam,phsoff=[0,0]))
# aa = ap.fit.AntennaArray(ants=ants,location=("+42:40:54.57","+81:05:39.09"))
# aa.set_ephemtime('2013/10/1 12:00') # observing time
# ref_src = ap.fit.RadioFixedBody('250:00','40:00',mfreq=1.42,name='center') # ra,dec
# src = ap.fit.RadioFixedBody('260:00','50:00',mfreq=1.42,name='src') # ra,dec
# ref_src.compute(aa)
# src.compute(aa)
# print lm.get_lm(ref_src,ref_src,aa)
# print lm.get_lm(src,ref_src,aa)
# print aa.gen_uvw(0,1,src=src)



def deg2str(deg):
    """
    Conver the deg in unit degree to string represent, e.g. '40:25:34.75'.
    Arguments:
    - `deg`:
    """
    deg = float(deg)
    int_deg = int(deg)
    minite = 60*(deg - int_deg)
    int_min = int(minite)
    second = 60*(minite - int_min)
    return str(int_deg) + ':' + str(int_min) + ':' + str(second)

def gen_name(i,j,max_i,max_j):
    """
    Generate a unique name for each Radio Baddies in a sky map.
    Arguments:
    - `i`: First axes index;
    - `j`: Second axes index;
    - `max_i`: max(i);
    - `max_j`: max(j).
    """
    assert i <= max_i and j <= max_j, "i must less equal than max_i, and j must less equal than max_j."
    name = ''
    for n in range(len(str(max_i)) - len(str(i))):
        name += '0'
    name += str(i)
    name += '_'
    for n in range(len(str(max_j)) - len(str(j))):
        name += '0'
    name += str(j)
    return name


# freqs = np.array([0.8]) # GHz
# nchan = (freqs.shape)[0]
# sdf = 0.0
# sfreq = freqs[0]
# aa = ap.cal.get_aa('ant_array',freqs)
sdf = 0.01
sfreq = 0.8
nchan = 2
aa = ap.cal.get_aa('ant_array',sdf,sfreq,nchan)
Tsky = np.load('/home/zuoshifan/21cm_foreground_noise/T_total_mK.npy') # mK
Tsky = Tsky[0] # only the required freq
# Isky = 2*cst.k_B * (Tsky*1.0e-3) * (freqs[0]*1.0e9)**2 / cst.c**2 # W m^-2 Hz^-1 sr^-1
Isky = 2*cst.k_B * (Tsky*1.0e-3) * (sfreq*1.0e9)**2 / cst.c**2 # W m^-2 Hz^-1 sr^-1
##################### Omega_s ##########################
Omega_s = 0.1 # the solid angle of the source, in sr
Jys = 1.0e26 * Isky * Omega_s # in Jy
ras = np.linspace(235.0,265.0,360) # ra, in degree
decs = np.linspace(25.0,55.0,360) # dec, in degree
ras = [deg2str(ra) for ra in ras]
decs = [deg2str(dec) for dec in decs]
nra,ndec = len(ras),len(decs)
nra_ctr,ndec_ctr = nra/2,ndec/2 # center index
cat = ap.fit.SrcCatalog([])
for i in range(nra):
    for j in range(ndec):
        if i == nra_ctr and j == ndec_ctr:
            # ref_src = ap.fit.RadioFixedBody(ras[i],decs[j],jys=Jys[i][j],mfreq=freqs[0],name='center') # the center reference source
            ref_src = ap.fit.RadioFixedBody(ras[i],decs[j],jys=Jys[i][j],mfreq=sfreq,name='center') # the center reference source
            cat.add_srcs([ref_src])
        else:
            # cat.add_srcs([ap.fit.RadioFixedBody(ras[i],decs[j],jys=Jys[i][j],mfreq=freqs[0],name=gen_name(i,j,nra,ndec))])
            cat.add_srcs([ap.fit.RadioFixedBody(ras[i],decs[j],jys=Jys[i][j],mfreq=sfreq,name=gen_name(i,j,nra,ndec))])
# aa.select_chans([0]) # Select which channels are used in computations
aa.set_ephemtime('2013/6/1 12:00') # observing time
startjd = aa.get_jultime() # Julian Date to start observation
aa.set_ephemtime('2013/6/1 12:08') # observing time
endjd = aa.get_jultime() # Julian Date to end observation

# ---------------------------------------------------------
# options for simulation
mode = 'sim'
flag = False
noiselev = 0.
pol = 'xx'
src = 'sky map ra: 235:00:00 - 265:00:00, dec: 25:00:00 - 55:00:00'
inttime = 10 # s

no_data = np.zeros(nchan, dtype=np.complex64)
no_flags = np.zeros(nchan, dtype=np.int32)
mfq = cat.get('mfreq')
a1s,a2s,ths = cat.get('srcshape')
dras, ddecs = cat.get('ionref')

curtime = None
def mdl(uv, p, d, f):
    global curtime, eqs
    uvw, t, (i,j) = p
    if i == j: return p, d, f
    if curtime != t:
        curtime = t
        aa.set_jultime(t)
        cat.compute(aa)
        eqs = cat.get_crds('eq', ncrd=3)
        flx = cat.get_jys()
        aa.sim_cache(eqs, flx, mfreqs=mfq, ionrefs=(dras,ddecs), srcshapes=(a1s,a2s,ths))
    sd = aa.sim(i, j, pol=ap.miriad.pol2str[uv['pol']])
    if mode.startswith('sim'):
        d = sd
        if not flag: f = no_flags
    elif mode.startswith('sub'):
        d -= sd
    elif mode.startswith('add'):
        d += sd
    else:
        raise ValueError('Mode "%s" not supported.' % mode)
    if noiselev != 0:
        # Add on some noise for a more realistic experience
        noise_amp = np.random.random(d.shape) * noiselev
        noise_phs = np.random.random(d.shape) * 2*np.pi * 1j
        noise = noise_amp * np.exp(noise_phs)
        d += noise * aa.passband(i, j)
    return p, np.where(f, 0, d), f

# Initialize a new UV file
pols = pol.split(',')
uv = ap.miriad.UV('../sim_data/sim3.uv', status='new')
uv._wrhd('obstype','mixed-auto-cross')
uv._wrhd('history','MDLVIS: created file.\nMDLVIS: srcs=%s mode=%s flag=%s noise=%f\n' % (src, mode, flag, noiselev))
uv.add_var('telescop','a'); uv['telescop'] = 'AIPY'
uv.add_var('operator','a'); uv['operator'] = 'AIPY'
uv.add_var('version' ,'a'); uv['version'] = '0.0.1'
uv.add_var('epoch'   ,'r'); uv['epoch'] = 2000.
uv.add_var('source'  ,'a'); uv['source'] = 'sky map ra: 235:00:00 - 265:00:00, dec: 25:00:00 - 55:00:00'
uv.add_var('latitud' ,'d'); uv['latitud'] = aa.lat
uv.add_var('dec'     ,'d'); uv['dec'] = aa.lat
uv.add_var('obsdec'  ,'d'); uv['obsdec'] = aa.lat
uv.add_var('longitu' ,'d'); uv['longitu'] = aa.long
uv.add_var('npol'    ,'i'); uv['npol'] = len(pols)
uv.add_var('nspect'  ,'i'); uv['nspect'] = 1
uv.add_var('nants'   ,'i'); uv['nants'] = len(aa)
uv.add_var('antpos'  ,'d')
antpos = np.array([ant.pos for ant in aa], dtype=np.double)
uv['antpos'] = antpos.transpose().flatten()
uv.add_var('sfreq'   ,'d'); uv['sfreq'] = sfreq
uv.add_var('freq'    ,'d'); uv['freq'] = sfreq
uv.add_var('restfreq','d'); uv['restfreq'] = sfreq
uv.add_var('sdf'     ,'d'); uv['sdf'] = sdf
uv.add_var('nchan'   ,'i'); uv['nchan'] = nchan
uv.add_var('nschan'  ,'i'); uv['nschan'] = nchan
uv.add_var('inttime' ,'r'); uv['inttime'] = float(inttime)
# These variables just set to dummy values
uv.add_var('vsource' ,'r'); uv['vsource'] = 0.
uv.add_var('ischan'  ,'i'); uv['ischan'] = 1
uv.add_var('tscale'  ,'r'); uv['tscale'] = 0.
uv.add_var('veldop'  ,'r'); uv['veldop'] = 0.
# These variables will get updated every spectrum
uv.add_var('coord'   ,'d')
uv.add_var('time'    ,'d')
uv.add_var('lst'     ,'d')
uv.add_var('ra'      ,'d')
uv.add_var('obsra'   ,'d')
uv.add_var('baseline','r')
uv.add_var('pol'     ,'i')

# Now start generating data
times = np.arange(startjd, endjd, inttime/ap.const.s_per_day)
for cnt,t in enumerate(times):
    print 'Timestep %d / %d' % (cnt+1, len(times))
    aa.set_jultime(t)
    uv['lst'] = aa.sidereal_time()
    uv['ra'] = aa.sidereal_time()
    uv['obsra'] = aa.sidereal_time()
    for i,ai in enumerate(aa):
        for j,aj in enumerate(aa):
            if j < i: continue
            # crd = ai.pos - aj.pos # the MIRIAD convention is aj.pos - ai.pos
            crd = aa.get_baseline(i,j) # accommodate MIRIAD convention
            preamble = (crd, t, (i,j))
            for pol in pols:
                uv['pol'] = ap.miriad.str2pol[pol]
                preamble,data,flags = mdl(uv, preamble, None, None)
                if data is None:
                    data = no_data
                    flags = no_flags
                uv.write(preamble, data, flags)
del(uv)

