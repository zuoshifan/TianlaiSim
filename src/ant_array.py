import numpy as np
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt
import aipy as ap
import ary as ay
import constant as cst


# const
d = 2     # m antenna diameter



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
         [0.,0.,  0.],[100.,0.,  0.],[200.,0.,  0.],[300.,0.,  0.],
         [0.,100.,0.],[100.,100.,0.],[200.,100.,0.],[300.,100.,0.],
         [0.,200.,0.],[100.,200.,0.],[200.,200.,0.],[300.,200.,0.],
         [0.,300.,0.],[100.,300.,0.],[200.,300.,0.],[300.,300.,0.]
    ]) # 16 antennas in square array, spacing approx 30m. unit: ns

locate = ('+42:40:54.47', '+81:05:39.09') # (lat,long) of  Zhaosu, Sinkiang
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


# plot the beam, antenna array, uv coverage
if __name__ == '__main__':
    plt_beam = True
    plt_antArray = False
    plt_uvCov = False
    if plt_beam == True:
       freqs = np.array([0.150,0.160,0.170]) # GHz
       w = cst.c/(d*freqs[len(freqs)/2]*1.0e9) # beam width in angular coord, lambda/d
       beam = prms['beam'](freqs,w,w)
       beam.select_chans([1])
       grids = 200
       x,y = np.mgrid[-w:w:1j*grids,-w:w:1j*grids] # generate grid data
       x1 = x.reshape(-1)
       y1 = y.reshape(-1)
       z1 = np.zeros_like(x1)
       resp = beam.response([x1,y1,z1])
       resp.shape = grids, -1
       ext = (-w,w,-w,w)
       # 2D
       plt.figure(figsize=(8,6))
       plt.imshow(resp,extent=ext,origin='lower')
       plt.colorbar(shrink=0.75)
       # plt.axes().set_aspect('equal', 'datalim')
       plt.xlabel('x (EW)')
       plt.ylabel('y (NS)')
       # plt.xlim(-w,w)
       plt.savefig('../figure/vis/png/beam2d.png')
       plt.savefig('../figure/vis/eps/beam3d.eps')
       plt.show()
       # 3D
       # ax = plt.subplot(111,projection='3d')
       # ax.plot_surface(x,y,resp,cmap=plt.cm.jet)
       # ax.set_xlabel('x (EW)')
       # ax.set_ylabel('y (NS)')
       # ax.set_zlabel('z')
       # plt.savefig('figure/png/beam3d.png')
       # plt.savefig('figure/eps/beam3d.eps')
       # plt.show()
    if plt_antArray:
        x = [x1 for [x1,y1,z1] in ant_pos]
        y = [y2 for [x2,y2,z2] in ant_pos]
        plt.figure(figsize=(8,6))
        plt.scatter(x,y)
        plt.axes().set_aspect('equal', 'datalim')
        plt.xlabel('East-West Antenna Position (ns)')
        plt.ylabel('North-South Antenna Position (ns)')
        plt.savefig('../figure/vis/png/AntennaArray.png')
        plt.savefig('../figure/vis/eps/AntennaArray.eps')
        plt.show()
    if plt_uvCov:
        nants = (ant_pos.shape)[0]
        # Not consider conjugate baseline
        bl1 = [(i,j) for i in range(nants) for j in range(i+1,nants)]
        uvCov1 = []
        for i,j in bl1:
            uvCov1.append(ant_pos[j]-ant_pos[i])
        u_x1 = [x1 for [x1,y1,z1] in uvCov1]
        u_y1 = [y2 for [x2,y2,z2] in uvCov1]
        plt.figure(figsize=(8,6))
        plt.scatter(u_x1,u_y1)
        plt.axes().set_aspect('equal', 'datalim')
        plt.xlabel('u (ns)')
        plt.ylabel('v (ns)')
        plt.savefig('figure/png/uv_coverage1.png')
        plt.savefig('figure/eps/uv_coverage1.eps')
        plt.show()
        # Now consider conjugate baseline
        bl2 = [(i,j) for i in range(nants) for j in range(nants)]
        uvCov2 = []
        for i,j in bl2:
            uvCov2.append(ant_pos[j]-ant_pos[i])
        u_x2 = [x1 for [x1,y1,z1] in uvCov2]
        u_y2 = [y2 for [x2,y2,z2] in uvCov2]
        plt.figure(figsize=(8,6))
        plt.scatter(u_x2,u_y2)
        plt.axes().set_aspect('equal', 'datalim')
        plt.xlabel('u (ns)')
        plt.ylabel('v (ns)')
        plt.savefig('../figure/vis/png/uv_coverage2.png')
        plt.savefig('../figure/vis/eps/uv_coverage2.eps')
        plt.show()


