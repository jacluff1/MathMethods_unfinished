import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors as colors

# length of theta (longitudinal) Array
N       =   1000j

# theta,phi arrays
# PHI         =   np.linspace(0,1,N/2) * np.pi    # longitudinal
# THETA       =   np.linspace(0,2,N) * np.pi      # colatitudinal
# theta,phi   =   np.meshgrid(THETA,PHI)

theta,phi   =   np.mgrid[0:np.pi:N, 0:2*np.pi:N/2]

def harmonics(m,n):

    r   = sp.sph_harm(m,n,theta,phi).real

    # convert to x,y,z
    # X   = np.abs(r) * np.sin(phi) * np.cos(theta)
    # Y   = np.abs(r) * np.sin(phi) * np.sin(theta)
    # Z   = np.abs(r) * np.cos(phi)
    X   = r * np.sin(phi) * np.cos(theta)
    Y   = r * np.sin(phi) * np.sin(theta)
    Z   = r * np.cos(phi)

    return r,X,Y,Z

def SH_surface_plots(n_max=6,figsize=(15,15),fs=15,saveA=True,show=False,dpi=400,vis_type='real'):
    """
    plots spherical harmonics as surface plots
    n       -> degree:          n >= 0
    m       -> order:           -n >= m >= n
    theta   -> colatitudinal:   0 >= theta > 2 pi
    phi     -> longitudinal:    0 >= phi > pi

    Parameters
    ----------
    n_max:      ** max n                - default = 6
    figsize:    ** size of figure       - default = (15,15)
    fs:         ** fontsize             - default = 15
    saveA:      ** will save iff True   - default = True
    show:       ** will show iff True   - default = False
    dpi:        ** resolution           - default = 400
    vis_type:   ** type of visual       - default = 'real'

    """

    N = 100j

    for n in range(n_max+1):
        for m in range(n+1):
            plt.close('all')
            print("working on Y_%s^%s" % (n,m) )

            PHI,THETA   = np.mgrid[0:2*np.pi:N*2, 0:np.pi:N]
            if vis_type == 'real':
                R   = sp.sph_harm(m,n,PHI,THETA).real
            if vis_type == 'modulus':
                r   = sp.sph_harm(m,n,PHI,THETA)
                R   = r * r.conjugate()
            if vis_type == 'unit':
                R   = sp.sph_harm(m,n,PHI,THETA).real + 1

            X       = np.abs(R) * np.sin(THETA) * np.cos(PHI)
            Y       = np.abs(R) * np.sin(THETA) * np.sin(PHI)
            Z       = np.abs(R) * np.cos(THETA)

            norm    = colors.Normalize()
            fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(14,10))
            sm      = cm.ScalarMappable(cmap=cm.seismic)
            ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=cm.seismic(norm(R)))
            ax.set_title('real$(Y^%s_%s)$' % (m,n), fontsize=fs)
            ax.set_aspect(1)
            sm.set_array(R)
            fig.colorbar(sm, shrink=0.8)

            if saveA:
                fig.savefig('images/%s/%s_%s.png' % (vis_type,n,m), dpi=dpi)
            if show:
                plt.show()

    # print("\n only +m values are used.")
    # for n in range(n_max+1):
    #     for m in range(n+1):
    #         plt.close('all')
    #         print("\n n,m = %s,%s" % (n,m) )
    #
    #         R,X,Y,Z   =   harmonics(m,n)
    #
    #         fig = plt.figure(figsize=figsize)
    #         ax  = plt.subplot(projection='3d')
    #         ax.set_aspect(1)
    #         ax.set_title("n: %s   m: %s" % (n,m), fontsize=fs+2)
    #         ax.plot_surface(X,Y,Z,\
    #             cmap    =   cm.seismic,
    #             norm    =   colors.Normalize( vmin=np.min(R),vmax=np.max(R) )\
    #             )
    #
    #         if saveA:
    #             fig.savefig('images/%s_%s.png' % (n,m), dpi=dpi)
    #         if show:
    #             plt.show()
