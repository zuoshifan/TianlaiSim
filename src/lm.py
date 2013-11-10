import numpy as np
import aipy as ap

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

if __name__ == '__main__':
    # const
    c = 3.0e8 # m/s
    d = 2     # m antenna diameter
    freqs = np.array([1.4])
    w = c/(d*freqs[len(freqs)/2]*1.0e9) # beam width in angular coord, lambda/d
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