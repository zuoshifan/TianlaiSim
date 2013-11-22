import numpy as np
import matplotlib.pyplot as plt
import aipy as ap
import constant as cst
import utils as ut

# antenna parameters
W = 40 # cylinder width in unit m

class BeamCylinder(ap.fit.Beam):
    """
    Represent the cylinder parabolic antenna beam.
    """
    
    def __init__(self,freqs,width):
        """
        Arguments:
        - `freqs`: frequencies (in GHz) at bin centers across spectrum.;
        - `width`: the cylinder width, in unit m.
        """
        ap.fit.Beam.__init__(self,freqs)
        self.width = width

    def response(self,xyz):
        """
        Return the beam response as a function of direction vector. This is given by Shaw et al. 2013, Eq.29.
        Arguments:
        - `xyz`: unit direction vector, x toward east, y toward north and toward the zenith.
        """
        vec_n = np.array(xyz)
        vec_z = np.array([0.0,0.0,1.0]) # unit vector pointing to the zenith
        nz = np.dot(vec_n,vec_z)
        if nz <= 0:
            return 0.0
        else:
            vec_u = np.array([1.0,0.0,0.0]) # unit vector pointing East in the ground-plane
            nu = np.dot(vec_n,vec_u)
            wavelen = 1.0e-9*cst.c / np.array(self.freqs) # wavelenth in unit m
            return np.abs(np.sinc(self.width*nu / wavelen))*np.sqrt(nz)


beam = BeamCylinder # Cylinder parabolic antenna beam
# antenna position in the horizon plan, x toward East, y toward North, z toward Zenith
fv = 50.0/3 # 5m
ft = 400.0/3 # 40m
ant_pos = np.array([
          [0.0,   0.0, 0.0],[ft,   0.0, 0.0],[2*ft,   0.0, 0.0],
          [0.0,    fv, 0.0],[ft,    fv, 0.0],[2*ft,    fv, 0.0],
          [0.0,  2*fv, 0.0],[ft,  2*fv, 0.0],[2*ft,  2*fv, 0.0],
          [0.0,  3*fv, 0.0],[ft,  3*fv, 0.0],[2*ft,  3*fv, 0.0],
          [0.0,  4*fv, 0.0],[ft,  4*fv, 0.0],[2*ft,  4*fv, 0.0],
          [0.0,  5*fv, 0.0],[ft,  5*fv, 0.0],[2*ft,  5*fv, 0.0],
          [0.0,  6*fv, 0.0],[ft,  6*fv, 0.0],[2*ft,  6*fv, 0.0],
          [0.0,  7*fv, 0.0],[ft,  7*fv, 0.0],[2*ft,  7*fv, 0.0],
          [0.0,  8*fv, 0.0],[ft,  8*fv, 0.0],[2*ft,  8*fv, 0.0],
          [0.0,  9*fv, 0.0],[ft,  9*fv, 0.0],[2*ft,  9*fv, 0.0],
          [0.0, 10*fv, 0.0],[ft, 10*fv, 0.0],[2*ft, 10*fv, 0.0],
          [0.0, 11*fv, 0.0],[ft, 11*fv, 0.0],[2*ft, 11*fv, 0.0],
          [0.0, 12*fv, 0.0],[ft, 12*fv, 0.0],[2*ft, 12*fv, 0.0],
          [0.0, 13*fv, 0.0],[ft, 13*fv, 0.0],[2*ft, 13*fv, 0.0],
          [0.0, 14*fv, 0.0],[ft, 14*fv, 0.0],[2*ft, 14*fv, 0.0],
          [0.0, 15*fv, 0.0],[ft, 15*fv, 0.0],[2*ft, 15*fv, 0.0],
          [0.0, 16*fv, 0.0],[ft, 16*fv, 0.0],[2*ft, 16*fv, 0.0],
          [0.0, 17*fv, 0.0],[ft, 17*fv, 0.0],[2*ft, 17*fv, 0.0],
          [0.0, 18*fv, 0.0],[ft, 18*fv, 0.0],[2*ft, 18*fv, 0.0],
          [0.0, 19*fv, 0.0],[ft, 19*fv, 0.0],[2*ft, 19*fv, 0.0],
          [0.0, 20*fv, 0.0],[ft, 20*fv, 0.0],[2*ft, 20*fv, 0.0],
          [0.0, 21*fv, 0.0],[ft, 21*fv, 0.0],[2*ft, 21*fv, 0.0],
          [0.0, 22*fv, 0.0],[ft, 22*fv, 0.0],[2*ft, 22*fv, 0.0],
          [0.0, 23*fv, 0.0],[ft, 23*fv, 0.0],[2*ft, 23*fv, 0.0],
       ]) # 3 cylinder parabolic antenna array, EW spacing 40m, NS spacing 5m, unit: ns.
nants = ant_pos.shape[0]

locate = ('+42:40:54.47', '+81:05:39.09') # (lat,long) of  Zhaosu, Sinkiang
prms = {
    'loc': locate,
    'antpos': np.dot(ut.xyz2XYZ_m(ut.latlong_conv(locate[0])),ant_pos.T).T,
    'delays': [0.] * nants, # zero delays for all antennas
    'offsets': [0.] * nants, # zero offsets for all antennas
    'amps': [1.] * nants,
    'passbands': np.ones(nants),
    'beam': beam,
}


# plot the beam, antenna array, uv coverage
if __name__ == '__main__':
    plt_beam = False
    plt_antArray = False
    plt_uvCov = True
    if plt_beam == True:
       freqs = np.array([0.8]) # GHz
       beam = prms['beam'](freqs,W)
       beam.select_chans([0])
       bound = 1.0
       grids = 200
       x1 = np.linspace(-bound,bound,grids)
       y1 = np.linspace(-bound,bound,grids)
       vec_ns = [[nx,ny,np.sqrt(1-nx**2-ny**2) if (1-nx**2-ny**2)>=0 else -np.sqrt(nx**2+ny**2-1)] for ny in y1 for nx in x1] # construct unit dirction victor n, some may not invalid for nz<0. Note the data arrangement, data in x axis changes faster
       resp = [beam.response(n) for n in vec_ns]
       resp = np.array(resp)
       resp.shape = grids, -1
       ext = (-bound,bound,-bound,bound)
       # 2D
       plt.figure(figsize=(8,6))
       plt.imshow(resp,extent=ext,origin='lower')
       plt.colorbar(shrink=0.75)
       # plt.axes().set_aspect('equal', 'datalim')
       plt.xlabel('x (EW)')
       plt.ylabel('y (NS)')
       # plt.xlim(-w,w)
       # plt.savefig('../figure/vis/png/beam2d.png')
       # plt.savefig('../figure/vis/eps/beam3d.eps')
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
        # plt.savefig('../figure/vis/png/AntennaArray.png')
        # plt.savefig('../figure/vis/eps/AntennaArray.eps')
        plt.show()
    if plt_uvCov:
        nants = (ant_pos.shape)[0]
        # # Not consider conjugate baseline
        # bl1 = [(i,j) for i in range(nants) for j in range(i+1,nants)]
        # uvCov1 = []
        # for i,j in bl1:
        #     uvCov1.append(ant_pos[j]-ant_pos[i])
        # u_x1 = [x1 for [x1,y1,z1] in uvCov1]
        # u_y1 = [y2 for [x2,y2,z2] in uvCov1]
        # plt.figure(figsize=(8,6))
        # plt.scatter(u_x1,u_y1)
        # plt.axes().set_aspect('equal', 'datalim')
        # plt.xlabel('u (ns)')
        # plt.ylabel('v (ns)')
        # plt.savefig('figure/png/uv_coverage1.png')
        # plt.savefig('figure/eps/uv_coverage1.eps')
        # plt.show()
        
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
        # plt.savefig('../figure/vis/png/uv_coverage2.png')
        # plt.savefig('../figure/vis/eps/uv_coverage2.eps')
        plt.show()


