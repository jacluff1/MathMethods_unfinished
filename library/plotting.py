import matplotlib as mpl

def set_mpl_params():
    mpl.rcParams['xtick.labelsize']=15
    mpl.rcParams['ytick.labelsize']=15

    mpl.rcParams['legend.loc']='best'
    mpl.rcParams['legend.numpoints']=1
    mpl.rcParams['legend.fontsize']=15

    mpl.rcParams['lines.linewidth']=3
    mpl.rcParams['lines.markersize']=20

    mpl.rcParams['font.size']=10

    mpl.rcParams['figure.figsize']=(15,15)
    mpl.rcParams['figure.titlesize']=27
    mpl.rcParams['figure.titleweight']='bold'

    mpl.rcParams['axes.titlesize']=25
    mpl.rcParams['axes.titleweight']='bold'
    mpl.rcParams['axes.linewidth']=1
    mpl.rcParams['axes.labelsize']=20
    mpl.rcParams['axes.labelpad']=10
