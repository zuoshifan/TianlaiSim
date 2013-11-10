import numpy as np
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt
import aipy as ap
import ary as ay
import constant as cst


# const
d = 6     # m antenna diameter

def xyz2XYZ_m(lat):
    """
    Matrix of coordinates conversion through xyz to XYZ.
    xyz coord: z toward zenith, x toward East, y toward North, xy in the horizon plane;
    XYZ coord: Z toward north pole, X in the local meridian plane, Y toward East, XY plane parallel to equatorial plane.
    Arguments:
    - `lat`: latitude of the observing position.
    """
    sin_a, cos_a = np.sin(lat), np.cos(lat)
    zero = np.zeros_like(lat)
    one = np.ones_like(lat)
    map =  np.array([[  zero,   -sin_a,   cos_a  ],
                     [   one,     zero,    zero  ],
                     [  zero,    cos_a,   sin_a  ]])
    if len(map.shape) == 3: map = map.transpose([2, 0, 1])
    return map

def latlong_conv(lat):
    """
    Covert the string represent latitude/longitude to radian.
    Arguments:
    - `lat`: string represent latitude
    """
    str_lat = lat.split(":")
    lat = 0.0
    for n in range(len(str_lat)):
        lat += float(str_lat[n])/(60.0**n)
    return lat*cst.deg2rad


beam = ap.fit.Beam2DGaussian # a 2D Gaussian beam pattern
# antenna position in the horizon plan, x toward East, y toward North, z toward Zenith
ant_pos = np.array([
         [0.,0.,  0.],[100./6,0.,  0.],[200./6,0.,  0.],[300./6,0.,  0.],
         [0.,100./6,0.],[100./6,100./6,0.],[200./6,100./6,0.],[300./6,100./6,0.],
         [0.,200./6,0.],[100./6,200./6,0.],[200./6,200./6,0.],[300./6,200./6,0.],
         [0.,300./6,0.],[100./6,300./6,0.],[200./6,300./6,0.],[300./6,300./6,0.]
    ]) # 16 antennas in square array, spacing approx 30m. unit: ns

# locate = ('+42:40:54.47', '+81:05:39.09') # (lat,long) of  Zhaosu, Sinkiang
locate = ('+44.160811', '+91.919494') # (lat,long)

prms = {
    'loc': locate,
    # 'antpos': np.array([
    #      [0.,0.,  0.],[0.,100.,  0.],[0.,200.,  0.],[0.,300.,  0.],
    #      [0.,0.,100.],[0.,100.,100.],[0.,200.,100.],[0.,300.,100.],
    #      [0.,0.,200.],[0.,100.,200.],[0.,200.,200.],[0.,300.,200.],
    #      [0.,0.,300.],[0.,100.,300.],[0.,200.,300.],[0.,300.,300.]
    # ]), # 16 antennas in square array, spacing approx 30m. unit: ns
    'antpos': np.dot(xyz2XYZ_m(latlong_conv(locate[0])),ant_pos.T).T,
    'delays': [0.] * 16, # zero delays for all antennas
    'offsets': [0.] * 16, # zero offsets for all antennas
    'amps': [1.] * 16,
    'passbands': np.ones(16),
    'beam': beam,
}

def get_aa(freqs):
    '''Return the AntennaArray to be used fro simulation.'''
    w = cst.c/(d*freqs[len(freqs)/2]*1.0e9) # beam width in angular coord, lambda/d
    beam = prms['beam'](freqs,w,w)
    try: beam.set_params(prms)
    except(AttributeError): pass
    location = prms['loc']
    antennas = []
    nants = len(prms['antpos'])
    assert(len(prms['delays']) == nants and len(prms['offsets']) == nants \
        and len(prms['amps']) == nants and len(prms['passbands']) == nants)
    for pos, dly, off, amp, bp in zip(prms['antpos'], prms['delays'],
            prms['offsets'], prms['amps'], prms['passbands']):
        antennas.append(
            ap.fit.Antenna(pos[0],pos[1],pos[2], beam, delay=dly, offset=off,
                amp=amp, bp=bp)
        )
    aa = ay.AntennaArray(location, antennas)
    # aa = ap.fit.AntennaArray(location, antennas)
    return aa


freqs = np.array([0.8]) # GHz
aa = get_aa(freqs) # antenna array
obs_time = '2014/5/4 12:00'
aa.set_ephemtime(obs_time)
ref_ra = '15.556' # hour
ref_dec = '39.58' # degree
ref_src = ap.fit.RadioFixedBody(ref_ra,ref_dec,name='center') # the reference source
ref_src.compute(aa)
az_alt = ref_src.get_crds('top',ncrd=2) # get (az, alt)
print az_alt
nants = len(aa.ants)
bl = [(i,j) for i in range(nants) for j in range (nants) if i != j] # all the baseline
uvw = []
for i,j in bl:
    uvw.append(np.squeeze(aa.gen_uvw(i,j,src=ref_src))) # relative to ref_src
    # uvw.append(np.squeeze(aa.gen_uvw(i,j,src='z'))) # relative to the zenith
u = [u1 for [u1,v1,w1] in uvw]
v = [v2 for [u2,v2,w2] in uvw]
plt.figure(figsize=(8,6))
plt.scatter(u,v)
plt.axes().set_aspect('equal', 'datalim')
plt.xlabel('u (wavelength)')
plt.ylabel('v (wavelength)')
plt.title('Instantaneous uv coverage')
# plt.savefig('figure/png/AntennaArray.png')
# plt.savefig('figure/eps/AntennaArray.eps')
plt.show()
