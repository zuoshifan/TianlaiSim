import numpy as np
import matplotlib.pyplot as plt
# import aipy as ap
import constant as cst
# import utils as ut

# antenna parameters
W = 40.0 # cylinder width in unit m


m100 = 1000.0/3 # 100m
L31 = m100/31 # 100/31 m
L32 = m100/32 # 100/32 m
L33 = m100/33 # 100/33 m
ft = 10*W/3 # 40m
ant_pos = np.array([
          [0.0,    0.0, 0.0],[ft,    0.0, 0.0],[2*ft,    0.0, 0.0],
          [0.0,    L31, 0.0],[ft,    L32, 0.0],[2*ft,    L33, 0.0],
          [0.0,  2*L31, 0.0],[ft,  2*L32, 0.0],[2*ft,  2*L33, 0.0],
          [0.0,  3*L31, 0.0],[ft,  3*L32, 0.0],[2*ft,  3*L33, 0.0],
          [0.0,  4*L31, 0.0],[ft,  4*L32, 0.0],[2*ft,  4*L33, 0.0],
          [0.0,  5*L31, 0.0],[ft,  5*L32, 0.0],[2*ft,  5*L33, 0.0],
          [0.0,  6*L31, 0.0],[ft,  6*L32, 0.0],[2*ft,  6*L33, 0.0],
          [0.0,  7*L31, 0.0],[ft,  7*L32, 0.0],[2*ft,  7*L33, 0.0],
          [0.0,  8*L31, 0.0],[ft,  8*L32, 0.0],[2*ft,  8*L33, 0.0],
          [0.0,  9*L31, 0.0],[ft,  9*L32, 0.0],[2*ft,  9*L33, 0.0],
          [0.0, 10*L31, 0.0],[ft, 10*L32, 0.0],[2*ft, 10*L33, 0.0],
          [0.0, 11*L31, 0.0],[ft, 11*L32, 0.0],[2*ft, 11*L33, 0.0],
          [0.0, 12*L31, 0.0],[ft, 12*L32, 0.0],[2*ft, 12*L33, 0.0],
          [0.0, 13*L31, 0.0],[ft, 13*L32, 0.0],[2*ft, 13*L33, 0.0],
          [0.0, 14*L31, 0.0],[ft, 14*L32, 0.0],[2*ft, 14*L33, 0.0],
          [0.0, 15*L31, 0.0],[ft, 15*L32, 0.0],[2*ft, 15*L33, 0.0],
          [0.0, 16*L31, 0.0],[ft, 16*L32, 0.0],[2*ft, 16*L33, 0.0],
          [0.0, 17*L31, 0.0],[ft, 17*L32, 0.0],[2*ft, 17*L33, 0.0],
          [0.0, 18*L31, 0.0],[ft, 18*L32, 0.0],[2*ft, 18*L33, 0.0],
          [0.0, 19*L31, 0.0],[ft, 19*L32, 0.0],[2*ft, 19*L33, 0.0],
          [0.0, 20*L31, 0.0],[ft, 20*L32, 0.0],[2*ft, 20*L33, 0.0],
          [0.0, 21*L31, 0.0],[ft, 21*L32, 0.0],[2*ft, 21*L33, 0.0],
          [0.0, 22*L31, 0.0],[ft, 22*L32, 0.0],[2*ft, 22*L33, 0.0],
          [0.0, 23*L31, 0.0],[ft, 23*L32, 0.0],[2*ft, 23*L33, 0.0],
          [0.0, 24*L31, 0.0],[ft, 24*L32, 0.0],[2*ft, 24*L33, 0.0],
          [0.0, 25*L31, 0.0],[ft, 25*L32, 0.0],[2*ft, 25*L33, 0.0],
          [0.0, 26*L31, 0.0],[ft, 26*L32, 0.0],[2*ft, 26*L33, 0.0],
          [0.0, 27*L31, 0.0],[ft, 27*L32, 0.0],[2*ft, 27*L33, 0.0],
          [0.0, 28*L31, 0.0],[ft, 28*L32, 0.0],[2*ft, 28*L33, 0.0],
          [0.0, 29*L31, 0.0],[ft, 29*L32, 0.0],[2*ft, 29*L33, 0.0],
          [0.0, 30*L31, 0.0],[ft, 30*L32, 0.0],[2*ft, 30*L33, 0.0],
                             [ft, 31*L32, 0.0],[2*ft, 31*L33, 0.0],
                                               [2*ft, 32*L33, 0.0]
      ])



if __name__ == '__main__':
    plt_synBeam = True
    if plt_synBeam:
        nants = (ant_pos.shape)[0]
        # Not consider conjugate baseline
        # bl = [(i,j) for i in range(nants) for j in range(i+1,nants)]
        # consider conjugate baseline
        bl = [(i,j) for i in range(nants) for j in range(nants) if i != j]
        nbl = len(bl)
        vec_b = []
        for i,j in bl:
            vec_b.append(ant_pos[j]-ant_pos[i])
        grids = 301
        L = np.linspace(-1,1,grids)
        M = np.linspace(-1,1,grids)
        LM = [(l,m) for m in M for l in L]
        synBeam = []
        k = 2*np.pi*0.8e9 / cst.c # wave vector
        for lm in LM:
            if lm[0]**2 + lm[1]**2 > 1.0:
                synBeam.append(0.0+0.0j)
            else:
                sum = 0.0 + 0.0j
                for n in range(nbl):
                    sum += np.exp(1.0j*k*(vec_b[n][0]*lm[0] + vec_b[n][1]*lm[1]))
                synBeam.append(sum)
        abs_synBeam = [np.absolute(var) for var in synBeam]
        abs_synBeam = np.array(abs_synBeam)
        abs_synBeam.shape = grids, -1
        plt.figure(figsize=(8,6))
        plt.imshow(abs_synBeam, extent=(-1,1,-1,1)) # , origin='lower')
        plt.colorbar(shrink=0.75)
        plt.title('Synthesized beam of cylinder array')
        plt.xlabel('l')
        plt.ylabel('m')
        plt.show()