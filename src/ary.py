'''
An antenna array class inherited from aipy.fit.AntennaArray, fixed some bugs of aipy.sim.AntennaArray and aipy.phs.AntennaArray.
'''

import aipy as ap
import numpy as np


class AntennaArray(ap.fit.AntennaArray):
    """Representation of location and time of observation, and response of array of antennas as function of pointing and frequency.
    """

    def resolve_src(self, u, v, srcshape=(0,0,0)):
        """Adjust amplitudes to reflect resolution effects for a uniform 
        elliptical disk characterized by srcshape:
        srcshape = (a1,a2,th) where a1,a2 are angular sizes along the 
            semimajor, semiminor axes, and th is the angle (in radians) of
            the semimajor axis from E."""
        a1,a2,th = srcshape
        try:
            if len(u.shape) > len(a1.shape): 
                a1.shape += (1,); a2.shape += (1,); th.shape += (1,)
        except(AttributeError): pass
        ru = a1 * (u*np.cos(th) - v*np.sin(th))
        rv = a2 * (u*np.sin(th) + v*np.cos(th))
        x = 2 * np.pi * np.sqrt(ru**2 + rv**2)
        # Use first Bessel function of the first kind (J_1)
        # return n.where(x == 0, 1, 2 * _cephes.j1(x)/x).squeeze()
        return np.where(x == 0, 1, 2 * ap._cephes.j1(x)/x)

    def sim(self, i, j, pol='xx'):
        """Simulate visibilites for the specified (i,j) baseline and 
        polarization.  sim_cache() must be called at each time step before 
        this will return valid results."""
        assert(pol in ('xx','yy','xy','yx'))
        if self._cache is None:
            raise RuntimeError('sim_cache() must be called before the first sim() call at each time step.')
        elif self._cache == {}:
            return np.zeros_like(self.passband(i,j))
        s_eqs = self._cache['s_eqs']
        # u,v,w computed, but never used, so I comment it out
        # u,v,w = self.gen_uvw(i, j, src=s_eqs)
        I_sf = self._cache['jys']
        Gij_sf = self.passband(i,j)
        Bij_sf = self.bm_response(i,j,pol=pol)
        if len(Bij_sf.shape) == 2: Gij_sf = np.reshape(Gij_sf, (1, Gij_sf.size))
        # Get the phase of each src vs. freq, also does resolution effects
        E_sf = np.conjugate(self.gen_phs(s_eqs, i, j, mfreq=self._cache['mfreq'],
            srcshape=self._cache['s_shp'], ionref=self._cache['i_ref'],
            resolve_src=True))
        try: E_sf.shape = I_sf.shape
        except(AttributeError): pass
        # Combine and sum over sources
        GBIE_sf = Gij_sf * Bij_sf * I_sf * E_sf
        Vij_f = GBIE_sf.sum(axis=0)
        return Vij_f
        