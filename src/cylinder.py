import numpy as np
import matplotlib.pyplot as plt
import aipy as ap
import constant as cst
import utils as ut
from matplotlib.ticker import MultipleLocator


class BeamCylinder(ap.fit.Beam):
    """
    Represent the cylinder parabolic antenna beam.
    """
    
    def __init__(self,freqs,width,length=0.0):
        """
        Arguments:
        - `freqs`: frequencies (in GHz) at bin centers across spectrum.;
        - `width`: the cylinder width (EW direction), in unit m;
        - `length`: the spacing between too feeds (NS direction), in unit m.
        """
        ap.fit.Beam.__init__(self,freqs)
        self.width = width
        self.length = length

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
            vec_v = np.array([0.0,1.0,0.0]) # unit vector pointing North in the ground-plane
            nu = np.dot(vec_n,vec_u)
            nv = np.dot(vec_n,vec_v)
            wavelen = 1.0e-9*cst.c / np.array(self.freqs) # wavelenth in unit m
            # return np.abs(np.sinc(self.width*nu / wavelen)*np.sinc(self.length*nv / wavelen)) * np.sqrt(nz)
            return (np.sinc(self.width*nu / wavelen)*np.sinc(self.length*nv / wavelen))**2 * np.sqrt(nz)



beam = BeamCylinder # Cylinder parabolic antenna beam
# antenna parameters
W = 15 # cylinder width in unit m
Ox = 7.5  # m
Oy = 1.75 # m
Ly = 1.5  # m
# antenna position in the horizon plan, x toward East, y toward North, z toward Zenith
ant_pos_m = np.array([
    [Ox,Oy,      0.0],[Ox+W,Oy,      0.0],[Ox+2*W,Oy,      0.0],
    [Ox,Oy+   Ly,0.0],[Ox+W,Oy+   Ly,0.0],[Ox+2*W,Oy+   Ly,0.0],
    [Ox,Oy+ 2*Ly,0.0],[Ox+W,Oy+ 2*Ly,0.0],[Ox+2*W,Oy+ 2*Ly,0.0],
    [Ox,Oy+ 3*Ly,0.0],[Ox+W,Oy+ 3*Ly,0.0],[Ox+2*W,Oy+ 3*Ly,0.0],
    [Ox,Oy+ 4*Ly,0.0],[Ox+W,Oy+ 4*Ly,0.0],[Ox+2*W,Oy+ 4*Ly,0.0],
    [Ox,Oy+ 5*Ly,0.0],[Ox+W,Oy+ 5*Ly,0.0],[Ox+2*W,Oy+ 5*Ly,0.0],
    [Ox,Oy+ 6*Ly,0.0],[Ox+W,Oy+ 6*Ly,0.0],[Ox+2*W,Oy+ 6*Ly,0.0],
    [Ox,Oy+ 7*Ly,0.0],[Ox+W,Oy+ 7*Ly,0.0],[Ox+2*W,Oy+ 7*Ly,0.0],
    [Ox,Oy+ 8*Ly,0.0],[Ox+W,Oy+ 8*Ly,0.0],[Ox+2*W,Oy+ 8*Ly,0.0],
    [Ox,Oy+ 9*Ly,0.0],[Ox+W,Oy+ 9*Ly,0.0],[Ox+2*W,Oy+ 9*Ly,0.0],
    [Ox,Oy+10*Ly,0.0],[Ox+W,Oy+10*Ly,0.0],[Ox+2*W,Oy+10*Ly,0.0],
    [Ox,Oy+11*Ly,0.0],[Ox+W,Oy+11*Ly,0.0],[Ox+2*W,Oy+11*Ly,0.0],
    [Ox,Oy+12*Ly,0.0],[Ox+W,Oy+12*Ly,0.0],[Ox+2*W,Oy+12*Ly,0.0],
    [Ox,Oy+13*Ly,0.0],[Ox+W,Oy+13*Ly,0.0],[Ox+2*W,Oy+13*Ly,0.0],
    [Ox,Oy+14*Ly,0.0],[Ox+W,Oy+14*Ly,0.0],[Ox+2*W,Oy+14*Ly,0.0],
    [Ox,Oy+15*Ly,0.0],[Ox+W,Oy+15*Ly,0.0],[Ox+2*W,Oy+15*Ly,0.0],
    [Ox,Oy+16*Ly,0.0],[Ox+W,Oy+16*Ly,0.0],[Ox+2*W,Oy+16*Ly,0.0],
    [Ox,Oy+17*Ly,0.0],[Ox+W,Oy+17*Ly,0.0],[Ox+2*W,Oy+17*Ly,0.0],
    [Ox,Oy+18*Ly,0.0],[Ox+W,Oy+18*Ly,0.0],[Ox+2*W,Oy+18*Ly,0.0],
    [Ox,Oy+19*Ly,0.0],[Ox+W,Oy+19*Ly,0.0],[Ox+2*W,Oy+19*Ly,0.0],
    [Ox,Oy+20*Ly,0.0],[Ox+W,Oy+20*Ly,0.0],[Ox+2*W,Oy+20*Ly,0.0],
    [Ox,Oy+21*Ly,0.0],[Ox+W,Oy+21*Ly,0.0],[Ox+2*W,Oy+21*Ly,0.0],
    [Ox,Oy+22*Ly,0.0],[Ox+W,Oy+22*Ly,0.0],[Ox+2*W,Oy+22*Ly,0.0],
    [Ox,Oy+23*Ly,0.0],[Ox+W,Oy+23*Ly,0.0],[Ox+2*W,Oy+23*Ly,0.0],
    [Ox,Oy+24*Ly,0.0],[Ox+W,Oy+24*Ly,0.0],[Ox+2*W,Oy+24*Ly,0.0],
    [Ox,Oy+25*Ly,0.0],[Ox+W,Oy+25*Ly,0.0],[Ox+2*W,Oy+25*Ly,0.0],
    [Ox,Oy+26*Ly,0.0],[Ox+W,Oy+26*Ly,0.0],[Ox+2*W,Oy+26*Ly,0.0],
    [Ox,Oy+27*Ly,0.0],[Ox+W,Oy+27*Ly,0.0],[Ox+2*W,Oy+27*Ly,0.0],
    [Ox,Oy+28*Ly,0.0],[Ox+W,Oy+28*Ly,0.0],[Ox+2*W,Oy+28*Ly,0.0],
    [Ox,Oy+29*Ly,0.0],[Ox+W,Oy+29*Ly,0.0],[Ox+2*W,Oy+29*Ly,0.0],
    [Ox,Oy+30*Ly,0.0],[Ox+W,Oy+30*Ly,0.0],[Ox+2*W,Oy+30*Ly,0.0],
    [Ox,Oy+31*Ly,0.0],[Ox+W,Oy+31*Ly,0.0],[Ox+2*W,Oy+31*Ly,0.0]
]) # 3x32 feeds, unit: m
nants = ant_pos_m.shape[0]
ant_pos_ns = cst.m2ns * ant_pos_m

locate = ('+42:40:54.47', '+81:05:39.09') # (lat,long) of  Zhaosu, Sinkiang
prms = {
    'loc': locate,
    'antpos': np.dot(ut.xyz2XYZ_m(ut.latlong_conv(locate[0])),ant_pos_ns.T).T, # unit:ns
    'delays': [0.] * nants, # zero delays for all antennas
    'offsets': [0.] * nants, # zero offsets for all antennas
    'amps': [1.] * nants,
    'passbands': np.ones(nants),
    'beam': beam,
}


# plot the beam, antenna array, uv coverage
if __name__ == '__main__':
    LMorDegree = False # Axis type. True for l,m, False for degree
    # what to plot
    plt_antArray = False
    plt_uvCov = False
    plt_beam = False
    plt_synBeam = False
    plt_AsynB = True

    # contral paramter
    freqs = np.array([0.8]) # GHz
    chan = 0 # frequency channel
    freq = freqs[chan]
    grids = 301
    ###-----------------------------------------------------
    if LMorDegree:
        bound = 1.0 # max(l) and max(m)
        # bound = 0.3
        L = np.linspace(-bound,bound,grids)
        M = L[:]
    ###----------------------------------------------------
    else:
        # bound = 90 # max(E) and max(N) in unit deg
        bound = 20
        EW = np.linspace(-bound,bound,grids) # deg
        L = [np.sin(ew*cst.deg2rad) for ew in EW]
        M = L[:]
    ###--------------------------------------------------
    ext = (-bound,bound,-bound,bound)
    # Consider conjugate baseline, also auto-correlation
    bl = [(i,j) for i in range(nants) for j in range(nants)]
    nbl = len(bl)

    ###----------------------------------------------------
    if plt_antArray == True:
        x = [x1 for [x1,y1,z1] in ant_pos_m]
        y = [y2 for [x2,y2,z2] in ant_pos_m]
        plt.figure(figsize=(8,6))
        plt.scatter(x,y)
        plt.axes().set_aspect('equal', 'datalim')
        plt.xlabel('East-West Antenna Position (m)')
        plt.ylabel('North-South Antenna Position (m)')
        plt.savefig('../figure/cylinder3x32/png/CylinderArray.png')
        plt.savefig('../figure/cylinder3x32/eps/CylinderArray.eps')
        # plt.show()
    ###---------------------------------------------------------
    if plt_uvCov == True:
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
        
        # Now consider conjugate baseline, also auto-correlation
        m2wavelen = 1.0e9*freq / cst.c
        uvCov2 = []
        for i,j in bl:
            uvCov2.append(m2wavelen*(ant_pos_m[j]-ant_pos_m[i])) # convert to u,v ,unit: 1
        u_x2 = [x1 for [x1,y1,z1] in uvCov2]
        u_y2 = [y2 for [x2,y2,z2] in uvCov2]
        plt.figure(figsize=(8,6))
        plt.scatter(u_x2,u_y2)
        plt.axes().set_aspect('equal', 'datalim')
        plt.xlabel('u (wavelenth)')
        plt.ylabel('v (wavelenth)')
        plt.savefig('../figure/cylinder3x32/png/uv_coverage.png')
        plt.savefig('../figure/cylinder3x32/eps/uv_coverage.eps')
        # plt.show()
    ###-------------------------------------------------------    
    if plt_beam == True or plt_AsynB == True:
        beam = prms['beam'](freqs,0.5*W,0.15*Ly)
        beam.select_chans([chan])
        vec_ns = [[nx,ny,np.sqrt(1-nx**2-ny**2) if (1-nx**2-ny**2)>=0 else -np.sqrt(nx**2+ny**2-1)] for ny in M for nx in L] # construct unit dirction victor n, some may not invalid for nz<0. Note the data arrangement, data in x axis changes faster
        resp = [beam.response(n) for n in vec_ns]
    if plt_beam == True:
        resp1 = np.array(resp)
        resp1.shape = grids, -1
        # 2D
        plt.figure(figsize=(8,6))
        plt.imshow(resp1,extent=ext,origin='lower')
        plt.colorbar(shrink=0.75)
        if LMorDegree == True:
            xlabel = 'l (EW)'
            ylabel = 'm (NS)'
            figname_sfx = 'lm'
        else:
            xlabel = 'EW (degree)'
            ylabel = 'NS (degree)'
            figname_sfx = 'deg'
            ax = plt.gca()
            ax.xaxis.set_major_locator(MultipleLocator(20))
            ax.yaxis.set_major_locator(MultipleLocator(20))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        plt.savefig('../figure/cylinder3x32/png/cylinderBeam_%s.png'%figname_sfx)
        plt.savefig('../figure/cylinder3x32/eps/cylinderBeam_%s.eps'%figname_sfx)
        # plt.show()  
        # 3D
        # ax = plt.subplot(111,projection='3d')
        # ax.plot_surface(x,y,resp,cmap=plt.cm.jet)
        # ax.set_xlabel('x (EW)')
        # ax.set_ylabel('y (NS)')
        # ax.set_zlabel('z')
        # plt.savefig('figure/png/beam3d.png')
        # plt.savefig('figure/eps/beam3d.eps')
        # plt.show()
    ###------------------------------------------------------
    if plt_synBeam == True or plt_AsynB == True:
        vec_b = []
        for i,j in bl:
            vec_b.append(ant_pos_m[j]-ant_pos_m[i])
        LM = [(l,m) for m in M for l in L]
        synBeam = []
        k = 2*np.pi*freq*1.0e9 / cst.c # wave vector
        for lm in LM:
            if lm[0]**2 + lm[1]**2 > 1.0:
                synBeam.append(0.0+0.0j)
            else:
                sum = 0.0 + 0.0j
                for n in range(nbl):
                    sum += np.exp(1.0j*k*(vec_b[n][0]*lm[0] + vec_b[n][1]*lm[1]))
                synBeam.append(sum)
        abs_synBeam = [np.absolute(var) / nbl for var in synBeam] # divide nbl for normalization
    if plt_synBeam == True:
        abs_synBeam1 = np.array(abs_synBeam)
        abs_synBeam1.shape = grids, -1
        plt.figure(figsize=(8,6))
        plt.imshow(abs_synBeam1, extent=ext, origin='lower')
        plt.colorbar(shrink=0.75)
        plt.title('Synthesized beam of cylinder array')
        if LMorDegree == True:
            xlabel = 'l (EW)'
            ylabel = 'm (NS)'
            figname_sfx = 'lm'
        else:
            xlabel = 'EW (degree)'
            ylabel = 'NS (degree)'
            figname_sfx = 'deg'
            ax = plt.gca()
            ax.xaxis.set_major_locator(MultipleLocator(20))
            ax.yaxis.set_major_locator(MultipleLocator(20))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig('../figure/cylinder3x32/png/synBeam_%s.png'%figname_sfx)
        plt.savefig('../figure/cylinder3x32/eps/synBeam_%s.eps'%figname_sfx)
        # plt.show() 
    ###--------------------------------------------------------
    if plt_AsynB == True:
        AsynB = [resp[index]*abs_synBeam[index] for index in range(len(resp))]
        AsynB1 = np.array(AsynB)
        AsynB1.shape = grids, -1
        plt.figure(figsize=(8,6))
        plt.imshow(AsynB1, extent=ext, origin='lower')
        plt.colorbar(shrink=0.75)
        if LMorDegree == True:
            xlabel = 'l (EW)'
            ylabel = 'm (NS)'
            figname_sfx = 'lm'
        else:
            xlabel = 'EW (degree)'
            ylabel = 'NS (degree)'
            figname_sfx = 'deg'
            ax = plt.gca()
            ax.xaxis.set_major_locator(MultipleLocator(20))
            ax.yaxis.set_major_locator(MultipleLocator(20))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # plt.savefig('../figure/cylinder3x32/png/AxSynBeam_%s.png'%figname_sfx)
        # plt.savefig('../figure/cylinder3x32/eps/AxSynBeam_%s.eps'%figname_sfx)
        plt.savefig('../figure/cylinder3x32/png/AxSynBeamCenter_%s.png'%figname_sfx)
        plt.savefig('../figure/cylinder3x32/eps/AxSynBeamCenter_%s.eps'%figname_sfx)
        # plt.show()


