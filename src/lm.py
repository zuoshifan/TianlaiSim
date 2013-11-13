import numpy as np
import matplotlib.pyplot as plt
import aipy as ap
import utils as ut
# import constant as cst

def get_hadec(FixedBody,ant_array):
    """
    Get the (ha,dec) of a Radio FixedBody at the location of the antenna array ant_array and its observing time.
    Arguments:
    - `FixedBody`: a radio FixedBody;
    - `ant_array`: the observing antenna array.
    """
    # hour angle = local sidereal time - ra
    return ant_array.sidereal_time() - FixedBody.ra, FixedBody.dec

def get_lm(FixedBody,ref_src,ant_array):
    """
    Conver the position of a radio FixedBody (ra,ded) to (l,m,sqrt(1-l^2-m^2)). After conversion, the ref_src becomes (0,0,1).
    Arguments:
    - `FixedBody`: a radio FixedBody;
    - `ref_src`: the source of the phase reference point, also a radio FixedBody;
    - `ant_array`: the observing antenna array.
    """
    h,d = get_hadec(FixedBody,ant_array)
    H,delta = get_hadec(ref_src,ant_array)
    lm = np.array(
       [np.cos(d)*np.sin(H-h),
        np.sin(d)*np.cos(delta) - np.cos(d)*np.sin(delta)*np.cos(H-h),
        np.sin(d)*np.sin(delta) + np.cos(d)*np.cos(delta)*np.cos(H-h)])
    return lm[0],lm[1]

# if __name__ == '__main__':
    '''
    # const
    # c = 3.0e8 # m/s
    d = 2     # m antenna diameter
    freqs = np.array([1.4])
    w = cst.c/(d*freqs[len(freqs)/2]*1.0e9) # beam width in angular coord, lambda/d
    beam = ap.fit.Beam2DGaussian(freqs,w,w)
    ants = []
    ants.append(ap.fit.Antenna(0,0,0,beam,phsoff=[0,0]))
    ants.append(ap.fit.Antenna(0,100,0,beam,phsoff=[0,0]))
    aa = ap.fit.AntennaArray(ants=ants,location=("+42:40:54.57","+81:05:39.09"))
    aa.set_ephemtime('2013/10/1 12:00') # observing time
    ref_src = ap.fit.RadioFixedBody('16:40','40:00',mfreq=1.42,name='center') # ra,dec
    src = ap.fit.RadioFixedBody('17:00','50:00',mfreq=1.42,name='src') # ra,dec
    ref_src.compute(aa)
    src.compute(aa)
    print get_lm(ref_src,ref_src,aa)
    print get_lm(src,ref_src,aa)
    '''

if __name__ == '__main__':
    sdf = 0.01
    sfreq = 0.8
    nchan = 2
    aa = ap.cal.get_aa('ant_array',sdf,sfreq,nchan)
    ras = np.linspace(235.0,265.0,360) # ra, in degree
    decs = np.linspace(25.0,55.0,360) # dec, in degree
    ras = [ut.deg2hstr(ra) for ra in ras] # now in hh:mm:ss
    decs = [ut.deg2str(dec) for dec in decs] # now in deg:mm:ss
    nra,ndec = len(ras),len(decs)
    nra_ctr,ndec_ctr = nra/2,ndec/2 # center index
    cat = ap.fit.SrcCatalog([])
    for i in range(nra):
        for j in range(ndec):
            if i == nra_ctr and j == ndec_ctr:
                ref_src = ap.fit.RadioFixedBody(ras[i],decs[j],mfreq=sfreq,name='center') # the center reference source
                cat.add_srcs([ref_src])
            else:
                cat.add_srcs([ap.fit.RadioFixedBody(ras[i],decs[j],mfreq=sfreq,name=ut.gen_name(i,j,nra,ndec))])
    # aa.select_chans([0]) # Select which channels are used in computations
    aa.set_ephemtime('2013/6/1 12:00') # observing time
    cat.compute(aa)
    lms = []
    # for src in cat.get_srcs(): # bugs in aipy.phs.SrcCatalog.get_srcs function. No return value for case len(srcs) == 0.
    for src in [cat[s] for s in cat.keys()]:
        lms.append(get_lm(src,ref_src,aa))
    print len(lms)
    ls = [l1 for (l1,m1) in lms]
    ms = [m2 for (l2,m2) in lms]
    plt.figure(figsize=(8,6))
    plt.scatter(ls,ms)
    plt.show()