"""
This script is used to extract C_l and compute C(theta), 
as well as to compute anomaly feature statistics for a bunch of maps. 

The general outline of this file is:
- utility functions for handling healpix maps (downgrading maps & masks, 
    filename handling, 2pt function measurement, map simulation using synfast, etc)
- functions for measuring features associated with anomalies
- functions for obtaining & handling feature meas. from many realizations
- functions for plotting 1d hists and 2d triangle scatter
- functions for measuring covariance matrix and doing PCA analysis

For the Muir, Adhikari, and Huterer 2018 anomaly paper this code was used as follows:
- The scripts in the main() function (at the bottom of this file) was used to generate &
  analyze simulated maps to get anomaly measurements for all desired realizations. There's 
  a hack there, where bool switches "if 0:" or "if 1" are used to manually turn off 
  and on parts of the calculation (if e.g. only one part needs to be rerun).
  -> The workhorse function used by that script is run_manystats which, given a list of stat names
     calls the necessary functions to compute them for all realizations, and handles file-naming
- Once the stat data is generated, a jupyter notebook (not included in this repo) was used to 
  analyze data and do plotting. It imports this python file as a module, and so uses
  some of the covariance calculation and plotting functions. 

The script is set up to expect specific filename structures, and
expects folders to be in a specific arrangement (as theya are on 
Jessie's computer). Because of this, it won't work out of the box as a 
self-contained script, but the functions defined in it could be useful,
either if it is imported as a module in another pythons script, or simply 
for reference to see how we've computed the various anomaly features. 

Script maintained by Jessie Muir (jlmuir@umich.edu or jlynnmuir@gmail.com).
Docstring last updated June 6, 2018. 
"""
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import multiprocessing
import subprocess
import os
import glob
import time
from scipy import stats,ndimage
import mpd_decomp #craig copi's multipole vector code
# can be obtained athttps://github.com/cwru-pat/cmb-multipole_vectors

import imp
#figure out where polspice is, depending on which machine code is running on
kobaspice = '/Users/jlmuir/bin/PolSpice_v03-02-00/src/ispice.py'
jmlaptopspice = '/Users/jessicamuir/bin/PolSpice_v03-02-00/src/ispice.py'
fastspice ='/opt/local/PolSpice_v03-03-01/src/ispice.py'
if os.path.isfile(kobaspice):
    polsp = imp.load_source('ispice',kobaspice)
elif os.path.isfile(jmlaptopspice):
    polsp = imp.load_source('ispice',jmlaptopspice)
elif os.path.isfile(fastspice):
    polsp = imp.load_source('ispice',fastspice)

#some default values
NSIDEfid=64
UseNcore = 5
#these values are from a table in the Planck 2015 Isotropy and Statistics paper
NSIDEtoFWHMarcmin = {2048:5, 1024:10, 512:20, 256:40, 128:80, 64:160,\
                     32:320, 16:640}
def arcmin2rad(angle):
    """
    Given angle in arcmin, convert into radians
    """
    return angle*np.pi/(60.*180.)

texdict = {'S12':r'$\log{S_{\frac{1}{2}}}$','R10':r'$R_{10}$','R27':r'$R_{27}$','R50':r'$R_{50}$','Cl2':r'$C_2$','Cl3':r'$C_3$','Ct180':r'$C(\pi)$','align23S':r'$S_{QO}$','ALV':r'$A_{LV}$','s16':r'$\sigma^2_{16}$'}

##################################################################
#UTILITY FUNCTIONS FOR EXTRACTING STATS FROM MASKS
##################################################################
def downgrade_map(inmap,NSIDEout):
    """
    Downgrades map, scaling by appropriate beam and pixel window
    functions, as discussed in Planck isotropy paper.
    """
    #get coefficent to covolve with beam and pixel window func
    plout = hp.sphtfunc.pixwin(NSIDEout)
    lmax = plout.size-1
    #print "LMAX is ",lmax
    NSIDEin = hp.get_nside(inmap)
    plin = hp.sphtfunc.pixwin(NSIDEin)[:lmax+1]
    fwhmin = arcmin2rad(NSIDEtoFWHMarcmin[NSIDEin])
    blin = hp.sphtfunc.gauss_beam(fwhmin,lmax=lmax)
    fwhmout = arcmin2rad(NSIDEtoFWHMarcmin[NSIDEout])
    blout = hp.sphtfunc.gauss_beam(fwhmout,lmax=lmax)
    multby = blout*plout/(blin*plin) #one number per ell

    #turn map to spherical harmonics, colvolve, then turn back into map
    alm = hp.sphtfunc.map2alm(inmap,lmax)
    alm = hp.almxfl(alm,multby)  #colvolve w/window funcs
    outmap = hp.sphtfunc.alm2map(alm,NSIDEout,verbose=False)
    return outmap

def downgrade_mask(maskmap,NSIDEout, threshold=0.9):
    """
    Downgrade and smooth mask to NSIDEout, then threshold it
    to get a binary mask again.
    """
    return (downgrade_map(maskmap,NSIDEout)>threshold).astype(bool)
    
def get_filename_testcase(datadir,mapbase,number,stattype = 'map', maskname='',extratag = ''):
    if stattype == 'map':
        ending = '.fits'
    elif stattype == 'cl':
        ending = '.cl.dat'
    elif stattype == 'ct':
        ending = '.ct.dat'
    elif stattype == 'lvmap':
        ending = '.{0:s}.fits'.format(extratag)
    elif stattype == 'statsummary':
        ending = '.stats.dat'
    elif stattype == 'Rall':
        ending = '.Rall.dat'
    elif stattype == 'Rall-contours':
        ending = '.Rall-contours.dat'
    elif stattype == 'Rall-contours-singletail':
        ending = '.Rall-contours-singletail.dat'
    elif stattype == 'Rall-contours-hist':
        ending = '.Rall-contours-hist.dat'
    if maskname:
        maskstr = '-'+maskname
    else:
        maskstr = ''

    return ''.join([datadir,mapbase,maskstr,'_{0:d}'.format(number),ending])

def get_filename_forstat(datadir,mapbase, rmin, rmax, stattype = 'S12', maskname =''):
    """
    Given map filename for one realization,
    returns filename for output datafile containing info for realizations
    numbered rmin through rmax
    If filename is of form datadir/mapname-maskstuff_####.type
    the output looks like datadir/statname.mapname-maskstuff_RMIN-RMAX.dat
    """
    if maskname:
        maskstr = '-'+maskname
    else:
        maskstr = ''
    return ''.join([datadir,stattype,'.',mapbase,maskstr,'_{0:d}-{1:d}'.format(rmin,rmax),'.dat'])

        
#-----------------------------------------
def gen_maps_from_cldat(cldatfile = "data/COM_PowerSpect_CMB-base-plikHM-TT-lowTEB-minimum-theory_R2.02.txt",\
                        Nmaps = 10, outdir = 'output/lcdm-map-testspace/', outbase = 'map', \
                        Nside = NSIDEfid, lmax = 200, regen = True, returnoutf=False,realmin =0):
    """
    Given Cl data filename, desired output file lcoation and name, and some other map properties,
    generates Nmaps .fits files consisteant with input C_ls
    
    If regen = False, doesn't generate maps, just returns filenames
    """
    data = np.loadtxt(cldatfile, skiprows=1)
    llist = np.arange(lmax+1)
    Clist = np.zeros(lmax+1)
    Clist[2:] = data[:lmax-1, 1]*2.*np.pi/(llist[2:]*(llist[2:] + 1))
    outfiles = []
    for i in xrange(realmin, realmin+Nmaps):
        outf = get_filename_testcase(outdir,outbase,i,'map')
        if regen:
            #print 'generate map with fwhm=',arcmin2rad(NSIDEtoFWHMarcmin[Nside])
            m = hp.sphtfunc.synfast(Clist, nside = Nside, verbose=False, fwhm = arcmin2rad(NSIDEtoFWHMarcmin[Nside]),pixwin=True)
            hp.write_map(outf,m)
            #print outf
        if returnoutf:
            outfiles.append(outf)
    return outfiles
#-----------------
def gen_manymaps_from_cldat(cldatfile = "data/COM_PowerSpect_CMB-base-plikHM-TT-lowTEB-minimum-theory_R2.02.txt",\
                        Nmaps = 10, outdir = 'output/lcdm-map-testspace/', outbase = 'map', \
                            Nside = NSIDEfid, lmax = 200, Ncore = 0,realmin =0):
    """
    Same as gen_maps_from_cldat, but parallelized for large numbers
    """
    availcore = multiprocessing.cpu_count()
    if not Ncore or (availcore<Ncore):
        Ncore = availcore
    print "Using {0} cores to make maps.".format(Ncore) 
    edges = np.linspace(realmin,realmin + Nmaps -1,num=Ncore+1,dtype=int)
    print "Splitting realizations into chunks with edges:\n",edges 
    rmins = edges[:-1]
    rmaxs = edges[1:]-1
    rmaxs[-1]+=1
    #start processes for each chunk of realizations
    jobs = []
    for i in xrange(Ncore):
        Nmapi = rmaxs[i] - rmins[i] +1
        p = multiprocessing.Process(target = gen_maps_from_cldat, args=(cldatfile,Nmapi, outdir, outbase, Nside, lmax, True, False, rmins[i]))
        jobs.append(p)
        print "Starting map-making for rlzns {0:d}-{1:d}".format(rmins[i],rmaxs[i])
        p.start()
    #wait until all are done before moving on
    for j in jobs:
        j.join()
#-----------------
def make_DQcorr_maps(dqcorr=np.array([1.46,0.28-2.64j,-1.16-0.25j]),indir = 'output/lcdm-map-testspace/', outdir = None, inbase = 'map',addtag='+DQ',Nside=NSIDEfid,Nmaps=10,realmin=0,regen=True, returnoutf=False):
    """
    Looks for a set of maps generated using naming conventions from 
    gen_manymaps_from_cldat, adds dipole quadurople correction where 
    either dqcorr is a map of matching Nside to maps, or 
    dqcorr = (a20,a21,a22). Default values are form Table 3 of 
    Copi  et al arxiv:1311.4562
    """
    if outdir is None:
        outdir = indir
    if dqcorr.size==3:
         #set up healpy alm array with lmax=2, put in dqcorr
        a20 = dqcorr[0]
        a21 = dqcorr[1]
        a22 = dqcorr[2]
        Nalm = hp.sphtfunc.Alm.getsize(2)
        dqalm = np.zeros(Nalm)
        dqalm[hp.sphtfunc.Alm.getidx(2,2,0)] = a20
        dqalm[hp.sphtfunc.Alm.getidx(2,2,1)] = a21
        dqalm[hp.sphtfunc.Alm.getidx(2,2,2)] = a22
        corrmap = hp.alm2map(dqalm,Nside,fwhm = arcmin2rad(NSIDEtoFWHMarcmin[Nside]),pixwin=True) #use same settings as gen_maps_from_cldat
    else:
        corrmap = dqcorr
    outfiles = []
    for i in xrange(realmin, realmin+Nmaps):
        uncorrf = get_filename_testcase(indir,inbase,i,'map')
        outf =  get_filename_testcase(outdir,inbase+addtag,i,'map')
        if regen:
            #print 'generate map with fwhm=',arcmin2rad(NSIDEtoFWHMarcmin[Nside])
            inm = hp.read_map(uncorrf)
            hp.write_map(outf,inm+corrmap)
            #print outf
        if returnoutf:
            outfiles.append(outf)
    return outfiles

def make_manyDQcorr_maps(dqcorr=np.array([1.46,0.28-2.64j,-1.16-0.25j]),indir = 'output/lcdm-map-testspace/', outdir = None, inbase = 'map',addtag='+DQ',Nside=NSIDEfid,Nmaps=10,realmin=0,Ncore=0):
    """
    Looks for a set of maps generated using naming conventions from 
    gen_manymaps_from_cldat, adds dipole quadurople correction where 
    dqcorr = (a20,a21,a22). Default values are form Table 3 of 
    Copi et al arxiv:1311.4562
    """
    if outdir is None:
        outdir = indir
    #set up healpy alm array with lmax=2, put in dqcorr
    a20 = dqcorr[0]
    a21 = dqcorr[1]
    a22 = dqcorr[2]
    Nalm = hp.sphtfunc.Alm.getsize(2)
    dqalm = np.zeros(Nalm,dtype=complex)
    dqalm[hp.sphtfunc.Alm.getidx(2,2,0)] = a20
    dqalm[hp.sphtfunc.Alm.getidx(2,2,1)] = a21
    dqalm[hp.sphtfunc.Alm.getidx(2,2,2)] = a22
    corrmap = hp.alm2map(dqalm,Nside,fwhm = arcmin2rad(NSIDEtoFWHMarcmin[Nside]),pixwin=True) #use same settings as gen_maps_from_cldat

    availcore = multiprocessing.cpu_count()
    if not Ncore  or (availcore<Ncore):
        Ncore = availcore
    print "Using {0} cores to apply DQ to maps.".format(Ncore) 
    edges = np.linspace(realmin,realmin + Nmaps -1,num=Ncore+1,dtype=int)
    print "Splitting realizations into chunks with edges:\n",edges 
    rmins = edges[:-1]
    rmaxs = edges[1:]-1
    rmaxs[-1]+=1
    #start processes for each chunk of realizations
    jobs = []
    for i in xrange(Ncore):
        Nmapi = rmaxs[i] - rmins[i] +1
        p = multiprocessing.Process(target = make_DQcorr_maps, args=(corrmap,indir,outdir,inbase,addtag,Nside,Nmapi, rmins[i],True, False))
        jobs.append(p)
        print "Starting DQ corrs for for rlzns {0:d}-{1:d}".format(rmins[i],rmaxs[i])
        p.start()
    #wait until all are done before moving on
    for j in jobs:
        j.join()
##################################################################
def getCl_fromSpice(Tmapfile,maskfile,outclfile='output/lcdm-map-testspace/testmasked.cl.dat',overwrite=True):
    # use spice to get C_l for masked map
    #print 'saving outclfile',outclfile
    if (overwrite or not os.path.isfile(outclfile)) or (not os.stat(outclfile).st_size):
        #print 'calling spice'
        polsp.ispice(Tmapfile, clout = outclfile, maskfile1 = maskfile, subav='YES', subdipole='YES', fits_out='NO', apodizesigma="NO",pixelfile="NO")
        #Print 'done with spice'
    else:
        print "file already exists, reading",outclfile
        pass
    
    cl = readSpice_clfile(outclfile) #array of Cl 
    return cl
#-----------------------------------------
def readSpice_clfile(infile):
    #take in name of spice output .cl file, return array of cl
    x = np.loadtxt(infile,skiprows=1)
    return x[:,1]
#-----------------------------------------
def getCl_anafast(Tmapfile,outclfile = 'output/lcdm-map-testspace/testmap.cl.dat', overwrite = True ):
    if (overwrite or not os.path.isfile(outclfile)) or (not os.stat(outclfile).st_size):
        m = hp.read_map(Tmapfile,verbose= False)
        cl = hp.sphtfunc.anafast(m)
        ell = np.arange(cl.size)
        outdat = np.zeros((cl.size,2))
        outdat[:,0] = ell
        outdat[:,1] = cl
        np.savetxt(outclfile,outdat,header = 'ell  C_l')
        #this should be in a format so it can be read the same as readSpce_clfile
    else:
        print "file already exists, reading",outclfile
        cl = readSpice_clfile(outclfile)
    return cl
##################################################################
def getCtheta(cl, outctfile = 'output/lcdm-map-testspace/testmap.ct.dat',overwrite = True,LMAX=100):
    """
    given array of cl data, returns 2xn array with
    first column being theta values, second C(theta)
    """
    if (overwrite or not os.path.isfile(outctfile)) or (not os.stat(outctfile).st_size):
        if not LMAX:
            LMAX = cl.size -1
        cl = cl[:LMAX+1]
        ell = np.arange(LMAX+1)
        Pcoef = cl*(2.*ell + 1.)*.25/np.pi
        dtheta = 1.
        Ntheta = int(180./dtheta)+1
        theta = np.linspace(0,180,Ntheta)
        costheta = np.cos(theta*np.pi/180.)
        ct = np.polynomial.legendre.legval(costheta,Pcoef)

        outdat = np.zeros((theta.size,2))
        outdat[:,0] = theta
        outdat[:,1] = ct
        np.savetxt(outctfile,outdat, header = 'theta  C(theta)')
    else:
        print "file already exists, reading",outctfile
        outdat = np.loadtxt(outctfile,skiprows=1)
    return outdat

def getCtheta_cosmvar(cl, outctfile = 'output/lcdm-map-testspace/testmap.ct.dat',overwrite = True,LMAX=100):
    """
    given array of cl data, returns 2xn array with
    first column being theta values, second C(theta)
    """
    if (overwrite or not os.path.isfile(outctfile)) or (not os.stat(outctfile).st_size):
        if not LMAX:
            LMAX = cl.size -1
        cl = cl[:LMAX+1]
        ell = np.arange(LMAX+1)
        #clcosmvar = cl*cl*2./(2.*ell+1)
        
        Pcoef = cl*cl*(2.*ell + 1.)/(8.*np.pi*np.pi)
        Pcoef = np.diag(Pcoef)
        #print Pcoef[:5,:5]
        dtheta = 1.
        Ntheta = int(180./dtheta)+1
        theta = np.linspace(0,180,Ntheta)
        costheta = np.cos(theta*np.pi/180.)
        ct = np.polynomial.legendre.legval2d(costheta,costheta,Pcoef)

        outdat = np.zeros((theta.size,2))
        outdat[:,0] = theta
        outdat[:,1] = ct
        np.savetxt(outctfile,outdat, header = 'theta  var[C(theta)]')
    else:
        print "file already exists, reading",outctfile
        outdat = np.loadtxt(outctfile,skiprows=1)
    return outdat

##################################################################
# FUNCTIONS FOR MEASURING FEATUERSS ASSOC. WITH ANOMALIES
##################################################################

#---------------------------------------------------------------
def getSmeas(clin,x=0.5,LMAX=100,Itab = np.array([])): 
    #computes measure of large scale power S(x)
    #cl = array of C_l's, x= upper bound on cos(theta) integral
    if (not LMAX) or (LMAX>clin.size -1):
        LMAX = clin.size -1
        cl = clin
    else:
        cl = clin[:LMAX+1]
    #print '  Computing S('+str(x)+') with LMAX=',LMAX
    if not Itab.shape==(LMAX+1,LMAX+1):
        Itab = tabulate_Ifunc(x = x, LMAX = LMAX)
        
    cldat = (2.*np.arange(LMAX+1) + 1)*cl
    clascol = np.tile(cldat.reshape((cldat.size,1)),cldat.size) 
    clasrow = clascol.T
    Sval = np.sum((clascol*clasrow*Itab)[2:,2:])/(16.*np.pi*np.pi)
    return Sval

def tabulate_Ifunc(x,LMAX):
    """
    This is the matrix you get when integrating two legengre 
    polynomials with l=m,n from -1 to x

    Will return array of shape (LMAX+1)x(LMAX+1)
    """
    legPx = np.zeros(LMAX+2)
    for i in xrange(LMAX+2):
        Pcoef = np.zeros(i+1)
        Pcoef[-1] = 1
        legPx[i] = np.polynomial.legendre.legval(x,Pcoef)
    Imat = np.zeros((LMAX+2,LMAX+2))
    # need the LMAX+1 index for last diagonal entry, will
    # slice of last row and column before returning
    for m in xrange(LMAX+1):
        for n in xrange(m+1,LMAX+1):
            #do off diagonals first, as diagonals depend on them
            if m==0:
                A = 0.
            else:
                A = m*legPx[n]*(legPx[m-1]-x*legPx[m])
            if n==0:
                B = 0.
            else:
                B = -n*legPx[m]*(legPx[n-1]-x*legPx[n])
            Imat[n,m] = (A+B)/(n*n+n - m*m-m)
            Imat[m,n] = Imat[n,m] #symmetric
    for m in xrange(LMAX+1):

        if m==0:
            Imat[m,m] = x+1.
        elif m==1:
            Imat[m,m] = (x**3+1.)/3.
        else:
            A = (legPx[m+1]-legPx[m-1])*(legPx[m]-legPx[m-2])
            B = -(2*m-1)*Imat[m+1,m-1]+(2*m+1)*Imat[m,m-2]
            C = (2*m-1)*Imat[m-1,m-1]
            Imat[m,m] = (A+B+C)/(2*m+1)
    
    return Imat[:-1,:-1]

#---------------------------------------------------------------
def get_Rassymstat(cl,lmax=27,clstartsat=0):
    """
    Given C_ell data and lmax, computes R stat as on page 25 of
    https://arxiv.org/abs/1506.07135, which measures amount of parity
    assymetry. If 1, no assymetry, >1 even parity pref, <1 odd parity pref
    """
   
    LMIN=2
    ell = np.arange(LMIN,lmax+1)
    isodd = (ell%2).astype(bool)
    iseven = np.logical_not(isodd)

    Dl = ell*(ell+1.)*cl[LMIN-clstartsat:lmax+1-clstartsat]
    R = np.mean(Dl[iseven])/np.mean(Dl[isodd])
    
    return R


def get_Rassym_allell(cl,lmax=60,clstartsat=0):
    """
    Given C_ell data and lmax, computes R stat as on page 25 of
    https://arxiv.org/abs/1506.07135, which measures amount of parity
    assymetry. If 1, no assymetry, >1 even parity pref, <1 odd parity pref
    """
    #NSIDE=NSIDEfid
    #NSIDEin = NSIDEfid
    LMIN=2
    startind = LMIN - clstartsat
    endind = lmax+1 - clstartsat
    ell = np.arange(LMIN,lmax+1 )
    #print ell
    Nell = ell.size
    isodd = (ell%2).astype(bool)
    iseven = np.logical_not(isodd)

    #print iseven.size,ell.size,cl[LMIN:lmax+1].shape
    Devenarray = iseven*ell*(ell+1)*cl[startind:endind]
    Doddarray = isodd*ell*(ell+1)*cl[startind:endind]

    #each entry i will have Deven and Dodd for lmax=ell[i]
    Deven= np.cumsum(Devenarray)/np.cumsum(2*np.pi*iseven)
    Dodd = np.cumsum(Doddarray)/np.cumsum(2*np.pi*isodd)
    R = Deven/Dodd

    return R
#---------------------------------------------------------------
def get_l23_alignment_S(mapf):
    """
    Basing this on Saroj's script in alignment.py
    """
    LMAX=3
    m = hp.read_map(mapf, verbose=False)
    m = hp.remove_dipole(m, verbose=False)
    
    hpalm = hp.map2alm(m,lmax=LMAX)
    #get alm in format needed by mpd_decomp
    a2mpos = get_mpd_alms(hpalm, l=2) #only has positive m
    a3mpos = get_mpd_alms(hpalm, l=3)

    #gives ell x 3 array, where each row is a unit multipole vector (there are ell +1)
    v2, N2 = mpd_decomp.mpd_decomp_full_fit(a2mpos) #3x3
    v3, N3 = mpd_decomp.mpd_decomp_full_fit(a3mpos) #4x3

    # get the 4 cross products
    w212 = np.cross(v2[0], v2[1])
    w312 = np.cross(v3[0], v3[1])
    w323 = np.cross(v3[1], v3[2])
    w331 = np.cross(v3[2], v3[0])

    # # divide by the norms to get the unit vectors
    # w212 = w212 #/ np.linalg.norm(w212)
    # w312 = w312 #/ np.linalg.norm(w312)
    # w323 = w323 #/ np.linalg.norm(w323)
    # w331 = w331 #/ np.linalg.norm(w331)

    D1 = np.dot(w212, w312)
    D2 = np.dot(w212, w323)
    D3 = np.dot(w212, w331)
    meas= (np.abs(D1)+np.abs(D2)+np.abs(D3))/3.0 
    return meas

def get_mpd_alms(alm, l=2):
    """
    Basing this on Saroj's script in alignment.py
    input: alm - array of alm 
    output: alms upto l specified in the format of mpd_decomp
    """
    lmax = hp.sphtfunc.Alm.getlmax(alm.size)
    tempalm = np.array([ alm[hp.sphtfunc.Alm.getidx(lmax,l,m) ] for m in xrange(0,l+1)])
    almr = [tempalm[0].real ]
    for a in tempalm[1:]:
        almr.append(a.real)
        almr.append(a.imag)
    return np.array(almr)
#---------------------------------------------------------------
def get_lowres_mapvar(mapf,maskfile='',Nsideout = 16):
    """
    Given input map and mask, read them in, downgrade to Nsideout,
    and report variance of unmasked pixels. 
    """
    inmap = hp.read_map(mapf, verbose=False)
    NSIDEin = hp.get_nside(inmap)
    if NSIDEin>Nsideout:
        usemap = downgrade_map(inmap,Nsideout)
    else:
        print "Can't downgrade NSIDE from {0} to {1}. Sticking with original, {0}.".format(NSIDEin,Nsideout)
        usemap = inmap
    if maskfile:
        #check if low res version of the mask already exists
        # (note that how it was downgraded matters!
        #  going 2048->16 gives a smaller mask than 2048->64->16.)
        lowresmaskf = maskfile.replace('{0:04d}'.format(NSIDEin),'{0:04d}'.format(Nsideout))
        if os.path.isfile(lowresmaskf):
            #print 'using mask from ',lowresmaskf
            maskmap = hp.read_map(lowresmaskf, verbose=False)
        else:
            maskmap = hp.read_map(lowresmaskf, verbose=False)
            if NSIDEin>Nsideout:
                maskmap = downgrade_mask(maskmap,Nsideout)
        usemap = hp.ma(usemap)
        usemap.mask = np.logical_not(maskmap)
    usemap = hp.remove_dipole(usemap, verbose = False) #remove dipole before measuring variance
    return np.var(usemap)    

#============================================================
# local variance map stuff
#============================================================
def get_ALV_onemap_externalmean(datadir, mapbase, meanmapfile, varmapfile, lvmapdir = '', maskfile='', maskname='', disksizedeg = 8, NSIDELV = 16, overwrite = False):
    """
    Make LV map for a single map, then use externally saved mean and
    variance maps to measure ALV.
    (e.g. When measuring ALV for planck, need to get mean and variance
    from some set of simulations)

    datadir, mapbase, maskfile, maskname - info about input map
    lvmapdir - where to put local variance maps; defaults to datadir
    meanmapfile - filename of mean LV map to use when measuring LV
    variancemapfile - filename of variance LV map to use when measuring LV

    if overwrite, remeasure LV map, otherwise, read in file if it exists
    """
    if not lvmapdir:
        lvmapdir = datadir
    
    extractLVmap_forlist(0,0, datadir = datadir, lvmapdir = lvmapdir, mapbase = mapbase, maskfile = maskfile, maskname = maskname, disksizedeg = 8, NSIDELV = 16, overwrite = overwrite)

    lvbasestr ='LVd{0:02d}n{1:02d}'.format(int(disksizedeg),NSIDELV)
    lvmapf = get_filename_testcase(datadir = lvmapdir, mapbase = mapbase, number =0, stattype = 'lvmap', maskname = maskname,extratag = lvbasestr)
    lvmap = hp.read_map(lvmapf, verbose=False)
    if maskname:
        lvmaskfile= "{0:s}{1:s}_for{2:s}.fits".format(lvmapdir,maskname,lvbasestr)
        if  os.path.isfile(lvmaskfile): #make lv mask file if we don't have it
            lvmask = hp.read_map(lvmaskfile, verbose = False)
        else:
            mask = hp.read_map(maskfile, verbose = False)
            diskpixlist = get_diskpixlist(NSIDEfid,NSIDELV,disksizedeg,mask)
            lvmask = getLVmap_mask(mask,disksizedeg , NSIDELV,diskpixlist)
            hp.write_map(lvmaskfile,lvmask)
    else:
        lvmask = None
    
    #use mean and var from set of simulations
    meanmap = hp.read_map(meanmapfile, verbose=False)
    varmap = hp.read_map(varmapfile, verbose=False)
    zerovar = np.where(varmap==0)[0] #pixel indices
    varmap[zerovar] = 1
    weightmap = (meanmap**2)/varmap #meanmap factor is to make this dimensionless
    if lvmask is None:
        meanweights =  np.mean(weightmap)
    else:
        meanweights = np.mean(weightmap[lvmask==1])
    weightmap = weightmap/meanweights #avg weight is 1
    weightmap[zerovar] = 0
    
    ALV = get_ALVstat_foronemap(lvmap,meanmap,weightmap,lvmask)
    #note, if we save this, keep track of where the mean came from
    return ALV

#--------------------------------------------
def get_manyALVstats(realmin=0,realmax=100, mapbase = 'map',\
                     maskfile='',maskname='', \
                     datadir = 'output/lcdm-map-testspace/',\
                     lvmapdir = 'output/lvmap-testspace/', \
                     statdir = 'output/stat-testspace/', \
                     disksizedeg = 8, NSIDELV=16,redoLVmaps = True, Ncore=0):
    """
    Given parameters re. input names, output dir, etc.,
    runs procedures necessary to create local variance maps
    and measure A_LV statistics from them. 
    """
    #if lvmapdir or statdir aren't given, default to put them in datadir
    if not lvmapdir:
        lvmapdir = datadir
    if not statdir:
        stadir = datadir
    
    #first, make the LV maps, measuring mean and variance
    meanfile, varfile = get_LVmaps_formaplist(realmin = realmin, \
                                              realmax = realmax,\
                                              maskfile=maskfile,\
                                              maskname=maskname, \
                                              datadir = datadir,\
                                              lvmapdir = lvmapdir , \
                                              mapbase = mapbase, \
                                              disksizedeg = disksizedeg, \
                                              NSIDELV= NSIDELV,\
                                              overwrite = redoLVmaps,\
                                              Ncore = Ncore)

    #then parallelize and measure stats
    print "Computing ALV for list of existing LV maps; parallelizing."
    availcore = multiprocessing.cpu_count()
    if not Ncore or (availcore<Ncore):
        Ncore = availcore
    print "Using {0} cores.".format(Ncore) 
    edges = np.linspace(realmin,realmax,num=Ncore+1,dtype=int)
    print "Splitting realizations into chunks with edges:\n",edges 
    rmins = edges[:-1]
    rmaxs = edges[1:]-1
    rmaxs[-1]+=1
    #start processes for each chunk of realizations
    jobs = []
    for i in xrange(Ncore):
        p = multiprocessing.Process(target = computeALV_forlist, args=(rmins[i],rmaxs[i], lvmapdir, statdir, mapbase, maskname, disksizedeg, NSIDELV, meanfile, varfile))
        jobs.append(p)
        print "Starting ALV meas for rlzns {0:d}-{1:d}".format(rmins[i],rmaxs[i])
        p.start()
    #wait until all are done before moving on
    for j in jobs:
        j.join()



#-----------------------------
def computeALV_forlist(realmin,realmax,\
                       lvmapdir = '', statdir = '',\
                       mapbase='map', maskname='',\
                       disksizedeg = 8, NSIDELV=16, \
                       meanmapfile='',varmapfile = '',lvmaskfile=''):
    """
    Given range of realizations and info about input LV map filenames,
    reads in local variance maps, and uses meanmap and varmap to extract
    ALV.

    lvmapdir is where the local variance maps are stored
    statdir is where to put the extracted ALV files

    diskpixlist - array of arrays; computed if emtpy, but can be passed
            to save time

    if meanmapfile ,varmapfile, lvmaskfile string are passed, reads from them
           otherwise assumes their format matches the LV maps
    meanmapfile contains mean map of set of simulations
    varmap file contains variance map of set of simulations
    lvmaskfile contains mask for LV map where pixels are masked if less than 
       10% of pixels in its associated disk were unmasked
    """
    if not statdir:
        statdir = datadir
    lvbasestr = 'LVd{0:02d}n{1:02d}'.format(int(disksizedeg),NSIDELV)

    #get the lv mask
    if maskname and (not lvmaskfile):
        lvmaskfile= "{0:s}{1:s}_for{2:s}.fits".format(lvmapdir,maskname,lvbasestr)
    if lvmaskfile:
        print "Reading in LV mask file",lvmaskfile
        lvmask = hp.read_map(lvmaskfile, verbose = False)
    else:
        lvmask = None
        
    #get the mean and variance maps, as well as the LV map mask
    if not meanmapfile:
        meantag = '{2:s}_MEANr{0:d}-{1:d}'.format(realmin,realmax,lvbasestr)
        vartag = '{2:s}_VARr{0:d}-{1:d}'.format(realmin,realmax,lvbasestr)
        meanmapfile = get_filename_testcase(datadir = lvmapdir, mapbase = mapbase, number =0, stattype = 'lvmap', maskname = maskname,extratag = meantag)
        varmapfile = get_filename_testcase(datadir = lvmapdir, mapbase = mapbase, number =0, stattype = 'lvmap', maskname = maskname,extratag = vartag)
    print "Getting mean and variance of LV maps from",meanmapfile,' and ', varmapfile
    meanmap = hp.read_map(meanmapfile, verbose = False)
    varmap = hp.read_map(varmapfile, verbose = False)
    zerovar = np.where(varmap==0)[0] #pixel indices
    varmap[zerovar] = 1
    weightmap = (meanmap**2)/varmap #should be dimensionless
    # ^this is equivalent to taking the variance of map/meanmap
    if lvmask is None:
        meanweights = np.mean(weightmap)
    else:
        meanweights = np.mean(weightmap[lvmask==1])
    weightmap = weightmap/meanweights #avg weight is 1
    weightmap[zerovar] = 0    

    #set up output file and data structures
    rlzns = np.arange(realmin,realmax+1)
    Nmap = rlzns.size
    ALVf = get_filename_forstat(datadir = statdir, mapbase = mapbase, rmin = realmin, rmax = realmax, stattype = 'ALV', maskname = maskname)
    ALVdat = np.ones((Nmap,2))*np.nan
    ALVdat[:,0]=rlzns
    lvbasestr = 'LVd{0:02d}n{1:02d}'.format(int(disksizedeg),NSIDELV)
    inmaplist = [get_filename_testcase(datadir = lvmapdir, mapbase = mapbase, number =i, stattype = 'lvmap', maskname = maskname,extratag = lvbasestr) for i in rlzns]
    for i in xrange(Nmap):
        m = hp.read_map(inmaplist[i], verbose = False)
        ALVdat[i,1] = get_ALVstat_foronemap(m,meanmap,weightmap,lvmask)
    print "Saving ",ALVf
    np.savetxt(ALVf,ALVdat,header = 'realization, ALV for '+lvbasestr)
               
#--------------------------------
def get_ALVstat_foronemap(lvmap,meanmap,weights,lvmask = None):
    normlvmap = ((lvmap - meanmap)/meanmap)*weights
    # normelvmap should be dimensionless
    if (lvmask is not None):
        normlvmap = hp.ma(normlvmap)
        normlvmap.mask = np.logical_not(lvmask)
    dipolevec = hp.remove_dipole(normlvmap,fitval = True, verbose=False)[2]
    ALV = np.linalg.norm(dipolevec)
    return ALV

#--------------------------------
def get_LVmaps_formaplist(realmin,realmax,maskfile='',maskname='', \
                          datadir = 'output/lcdm-map-testspace/',\
                          lvmapdir = '',  mapbase = 'map',\
                          disksizedeg = 8, NSIDELV=16, overwrite=False,\
                          Ncore = 0):

    """
    Measures dipole amplitude of local variance maps. Steps:
    - remove monopole and dipole from masked sky
    - for each realization, make NSIDE=lvnside map, for each pixel, 
      store value of variance of unmasked pixels within a disk of size 
      disksizedeg (units=degrees) (save these in lvmapdir)
    - given all simulated LV maps of a set, find mean and variance maps 
      (avg over realizations)
    - subtract mean map from all simulated and observed LV maps
    - measure dipole amplidude from each  map, weighting pixels with inverse
      of variance accross simulated realizations

    If maskfile is empty string, assumes full sky.
    maskname is a shorter string used to indicate which mask was used
    in the output filenames

    datadir is where to find input maps
    lvmapdir is where to put output files for local variance maps; if empty
            is made equal to datadir
    disksizedeg - is the size of disks to use when measuring local variance
            in units of degrees
    NSIDELV - is the NSIDE paramter to use when making local variance masks
    overwrite - if true, redo all maps; if false
                only do if files don't already exist
    """
    
    lvbasestr = 'LVd{0:02d}n{1:02d}'.format(int(disksizedeg),NSIDELV)
    if not lvmapdir:
        lvmapdir = datadir

    #check input NSIDE
    firstmap = hp.read_map(get_filename_testcase(datadir = datadir, mapbase = mapbase, number =realmin, stattype = 'map'), verbose = False)
    NSIDEIN = hp.get_nside(firstmap)
    
    #set up mask if we have one, get low res version
    if maskfile:
        #check that the file exists
        assert os.path.isfile(maskfile), "Mask file doesn't exist!"
        #get mask and use it to get disk pixel lists
        mask = hp.read_map(maskfile, verbose = False)
        diskpixlist = get_diskpixlist(NSIDEIN,NSIDELV,disksizedeg,mask)
        #if we haven't already done so, get low res mask for LV maps
        if not maskname:
            maskname = maskfile[maskfile.rfind('/')+1:]
        maskstr = '-'+maskname

        lowresmaskf = "{0:s}{1:s}_for{2:s}.fits".format(lvmapdir,maskname,lvbasestr)
        if overwrite or not os.path.isfile(lowresmaskf):
            lowresmask = getLVmap_mask(mask,disksizedeg , NSIDELV,diskpixlist)
            hp.write_map(lowresmaskf,lowresmask)
            print "Saving mask for LV maps to",lowresmaskf
        else:
            print "Reading in mask for LV maps from",lowresmaskf
            lowreskmask = hp.read_map(lowresmaskf, verbose = False)
        
    else:
        maskstr = ''
        lowresmaskf = ''
        diskpixlist = get_diskpixlist(NSIDEIN,NSIDELV,disksizedeg)

    #using multiprocessing, go make all non-norm LV maps
    #split up realization numbers into chunks
    availcore = multiprocessing.cpu_count()
    if not Ncore or (availcore<Ncore):
        Ncore = availcore
    print "Using {0} cores for LV map making.".format(Ncore) 
    edges = np.linspace(realmin,realmax,num=Ncore+1,dtype=int)
    print "Splitting realizations into chunks with edges:\n",edges 
    rmins = edges[:-1]
    rmaxs = edges[1:]-1
    rmaxs[-1]+=1
    #start processes for each chunk of realizations
    jobs = []
    print "LV map extraction, creating QUEUE"
    queue = multiprocessing.Queue()
    print "LV map extraction, Starting processe"
    for i in xrange(Ncore):
        p = multiprocessing.Process(target = extractLVmap_forlist, args=(rmins[i], rmaxs[i], datadir, lvmapdir, mapbase, maskfile, maskname, disksizedeg, NSIDELV, overwrite, diskpixlist, NSIDEIN, queue))
        jobs.append(p)
        print "LV map making: Starting rlzns {0:d}-{1:d}".format(rmins[i],rmaxs[i])
        p.start()

   
    print "map-making jobs done, getting mean and variance info from queue"
    meanmaps = []
    varmaps = []
    counts = []
    for j in jobs:
        dat = queue.get()
        #print '  dat=',dat
        meanmaps.append(dat[0])
        varmaps.append(dat[1])
        counts.append(dat[2])
    #wait until all are done before moving on
    print "waiting until all jobs are done"
    for j in jobs:
        j.join()
        
    # go from mean and variances of subsets to toal mean and variance
    meanmaps = np.array(meanmaps)
    varmaps = np.array(varmaps)
    counts = np.array(counts)
    for i in xrange(counts.size):
        meanmaps[i,:]*=counts[i]
        varmaps[i,:]*=counts[i]
    totalmean = np.sum(meanmaps,axis=0)/np.sum(counts)
    totalvar = np.sum(varmaps,axis=0)/np.sum(counts)

    #save mean and variance
    meantag = '{2:s}_MEANr{0:d}-{1:d}'.format(realmin,realmax,lvbasestr)
    vartag = '{2:s}_VARr{0:d}-{1:d}'.format(realmin,realmax,lvbasestr)
    meanfile = get_filename_testcase(datadir = lvmapdir, mapbase = mapbase, number =0, stattype = 'lvmap', maskname = maskname,extratag = meantag)
    varfile = get_filename_testcase(datadir = lvmapdir, mapbase = mapbase, number =0, stattype = 'lvmap', maskname = maskname,extratag = vartag)
    print "Writing mean LV map to", meanfile
    print totalmean.shape
    hp.write_map(meanfile,totalmean)
    print "Writing variance LV map to", varfile
    hp.write_map(varfile,totalvar)
    
    return meanfile,varfile
        
#-----------------------------
def extractLVmap_forlist(realmin,realmax,\
                         datadir= 'output/lcdm-map-testspace/', \
                         lvmapdir = '',mapbase='map', maskfile='', maskname='',\
                         disksizedeg = 8, NSIDELV=16, overwrite = False,\
                         diskpixlist=[],NSIDEIN=None, queue= None):
    """
    Given range of realizations and info about input map filenames,
    goes through and makes local variance maps, storing them in lvmapdir.

    datadir - where the input maps are
    lvmapdir - where to put LV maps; if empty, matches datadir
    diskpixlist - array of arrays; computed if emtpy, but can be passed
            to save time

    if queue != put (meanmap,varmap,Nmap) into queue;
    otherwise just return that tuple

    if not overwrite, only make maps if files for them don't already exist
    """
    print 'In extractLVmap_forlist, rlzn=',realmin,' - ',realmax
    if not lvmapdir:
        lvmapdir = datadir
    rlzns = np.arange(realmin,realmax+1)

    print '  getting input map filenames'
    inmaplist = [get_filename_testcase(datadir = datadir, mapbase = mapbase, number =i, stattype = 'map') for i in rlzns]

    lvbasestr ='LVd{0:02d}n{1:02d}'.format(int(disksizedeg),NSIDELV)
    print 'getting output map filenames'
    outmaplist = [get_filename_testcase(datadir = lvmapdir, mapbase = mapbase, number =i, stattype = 'lvmap', maskname = maskname,extratag = lvbasestr) for i in rlzns]
    print '  reading in mask'
    if maskname:
        inmask = hp.read_map(maskfile , verbose = False)
    else:
        inmask = None
    if not len(diskpixlist):
        #print "Getting diskpixlist for rlzns",realmin,' - ',realmax
        if NSIDEIN is None:
            firstmap = hp.read_map(get_filename_testcase(datadir = datadir, mapbase = mapbase, number =realmin, stattype = 'map'), verbose = False)
            NSIDEIN = hp.get_nside(firstmap)
        diskpixlist = get_diskpixlist(NSIDEIN,NSIDELV,disksizedeg,inmask)

    Nmap = rlzns.size
    outmapdat = np.zeros((Nmap,len(diskpixlist)))
    print "  Getting LV map data for rlzns",realmin,' - ',realmax
    t0 = time.time()
    for i in xrange(Nmap):
        if overwrite or not os.path.isfile(outmaplist[i]):
            if i%100==0:
                print "   ...extracting map",i
            outmapdat[i,:] = extractLVmap(inmaplist[i],diskpixlist, outmaplist[i], inmask)
        else:
            if i%100==0:
                print "   ...reading map",i
            outmapdat[i,:] = hp.read_map(outmaplist[i], verbose = False)
    t1 = time.time()
    print "  ...took {0} sec".format(t1-t0)
    print '  getting mean and variance for rlzns',realmin,' - ',realmax
    #lvmask = getLVmap_mask(inmask,disksizedeg,NSIDELV,diskpixlist)
    meanmap = hp.ma(np.mean(outmapdat,axis=0))
    #meanmap.mask = np.logical_not(lvmask)
    #hp.mollview(meanmap,title = 'mean map')
    #plt.show()
    varmap = hp.ma(np.var(outmapdat,axis=0))
    #varmap.mask = np.logical_not(lvmask)
    #hp.mollview(varmap,title = 'variance map')
    #plt.show()
    if queue is None:
        print '  no queue, returning'
        return  meanmap,varmap,Nmap
    else:
        print '  putting mean and variance maps in queue'
        queue.put((meanmap,varmap,Nmap))
    return 
        
#-----------------------------                         
def extractLVmap(inmapf,diskpixlist=[],outmapf='',mask = None, NSIDELV = 16, disksizedeg = 8):
    """
    inmapf - string filename of input map
    mask - healpy array of mask to use (of same NSIDE as inmap)
           used here jsut for dipole subtraction, if diskpixlist given
    diskpixlist - list of all unmasked pixels in disks around NSIDELV pix
           if diskpixlist passed, NSIDELV and disksizedeg not used
    
    if outfile given, save the map there
    """
    
    inmap = hp.read_map(inmapf, verbose = False)
    #subtract monopole and dipole
    if (mask is not None):
        inmap = hp.ma(inmap)
        inmap.mask = np.logical_not(mask)

    inmap = hp.remove_dipole(inmap, verbose=False)

    if not len(diskpixlist):
        NSIDEIN = hp.get_nside(inmap)
        diskpixlist = get_diskpixlist(NSIDEIN,NSIDELV,disksizedeg,mask)
    NPIXLV = len(diskpixlist)

    plt.show()
    outmap = np.zeros(NPIXLV)
    for i in xrange(NPIXLV):
        if len(diskpixlist[i]):
            outmap[i] = np.var(inmap[diskpixlist[i]])
    if outmapf:
        print '    Saving ',outmapf
        hp.write_map(outmapf,outmap)
    return outmap
    
#-----------------------------                  
def get_diskpixlist(NSIDEIN,NSIDELV,disksizedeg,mask=None):
    """
    returns array of arrays; 
    each entry in outer array corresponds to a pixel in an healpy map of 
    resolution NSIDELV, and contains an array of the unmasked
    pixels in a map of resolution NSIDEIN within a disk of radius
    disksizedeg centered on that NSIDELV pixel.

    inmask - expects NSIDEIN map with 1 = unmasked
    """
    NpixLV = hp.nside2npix(NSIDELV)
    NpixIN = hp.nside2npix(NSIDEIN)
    if mask is not None:
        unmasked =np.where(mask)[0] #indices of unmasked pixels
    else:
        unmasked = np.arange(NpixIN)
    #get list of unmasked pixels in disk centered on each lower res pixel
    diskpixlist = []
    for p in xrange(NpixLV):
        alldisk = hp.query_disc(nside=NSIDEIN, vec=hp.pix2vec(NSIDELV, p), radius=np.deg2rad(disksizedeg))
        unmaskeddisk = np.intersect1d(alldisk,unmasked,assume_unique=True)
        diskpixlist.append(unmaskeddisk)
            
    return diskpixlist

#-----------------------------                  
def getLVmap_mask(inmask,disksizedeg = 8, NSIDELV=16,diskpixlist=[]):
    """
    Return mask for LV masks, where for a pixel to be unmasked, more than
    10% of the disk centered on it must be unmasked in original map.
    """
    NpixLV = hp.nside2npix(NSIDELV)
    NSIDEIN = hp.get_nside(inmask)
    if not len(diskpixlist):
        #list of unmasked pixels in disk corresponding to each NSIDELV disk
        diskpixlist = get_diskpixlist(NSIDEIN,NSIDELV,disksizedeg,inmask)
    #1 means unmasked in inmask, 0 means masked
    # if at least 1 percent of pixels are unmasked, leave unmasked
    fracunmasked = np.array([float(diskpixlist[i].size)/float(hp.query_disc(nside=NSIDEIN, vec=hp.pix2vec(NSIDELV, i), radius=np.deg2rad(disksizedeg)).size) for i in xrange(NpixLV)])
    #ratio of pixels in each disk with and without mask
    
    lvmask = (fracunmasked > 0.1).astype(bool)

    return lvmask

##################################################################
# FUNCTIONS FOR READING IN STAT DATA FROM FILES
# AND MANIPULATING ARRAYS OF ANOMALY STAT DATA FOR MANY REALIZATIONS
##################################################################

#============================================================
def get_realnum_from_statfname(fname):
    """
    Given filename for datafile with info about stats (e.g. S12),
    pulls out min and max realization number included in that realization
    """
    start = fname.rfind('_')+1
    end = fname.rfind('.')
    numstr = fname[start:end]
    ends = numstr.split('-')
    rmin = int(ends[0])
    rmax = int(ends[1])
    return rmin,rmax

def collect_data_forstat(statname = 'S12',datadir = 'output/stat-testspace/',\
                         mapbase = 'map', maskname ='',realmin=0,realmax = -1):
    """
    Given stat name and filename info, reads in data from files
    that have been split into chunks, returns 2xN array, first column
    is realization number, second is stat for each realization.

    if realmin ==0 and realmax ==-1, will find all data matching that filenaming convention.
    otherwise will collect all available data in given range.
    """
    print "Collecting stat data for",statname
    if maskname:
        maskstr = '-'+maskname
    else:
        maskstr = ''
    print statname,datadir,mapbase,maskname
    flist = glob.glob('{0:s}{1:s}.{2:s}{3:s}_*.dat'.format(datadir,statname,mapbase,maskstr))
    #print flist
    if not len(flist): #no files for this stat
        print "  >>No files for this stat, returning NaN"
        return np.nan
    rmins = []
    rmaxs = []
    for f in flist:
        rmin,rmax = get_realnum_from_statfname(f)
        rmins.append(rmin)
        rmaxs.append(rmax)
    #print rmins,rmaxs
    if (realmax<0) or (realmax>max(rmaxs)):
        realmax = max(rmaxs)
    if realmin < min(rmins):
        realmin = min(rmins) #don't try to read in data you don't have
    rmins = np.array(rmins)
    #print 'rmins:',rmins
    sortmin = np.argsort(rmins)
    #print 'rmaxs:',rmaxs
    rmaxs = np.array(rmaxs)
    sortmax = np.argsort(rmaxs)
    if not np.all(sortmin==sortmax):
        print " ***WARNING; overlapping realization ranges; may be duplicating data "
        
    outdat = np.zeros(( realmax - realmin +1,2))
    startonrow=0
    for i in sortmin:
        #print flist[i]
        dati = np.loadtxt(flist[i],skiprows=1)
        Ni = dati.shape[0]
        outdat[startonrow:startonrow+Ni] = dati
        startonrow += Ni
    #print outdat
    return outdat

def mix_statdats(uselist,fullstatdat,cutstatdat,qmlstatdat=None):
    """
    Given uselist containing one string of 'full','cut',or 'qml' for each stat,
    and lists of arrays (statdats) from collect_data_forstat (one entry per stat)
    return a statdat list where the array from the appropriate statdat. (for mixing full and cut stats)
    """
    Nstat = len(uselist)
    outdat = []
    for i in xrange(Nstat):
        if uselist[i]=='full':
            outdat.append(fullstatdat[i])
        elif uselist[i]=='cut':
            outdat.append(cutstatdat[i])
        elif uselist[i]=='qml':
            outdat.append(qmlstatdat[i])
    return outdat

def collect_data_forRall(datadir = 'output/stat-testspace/',\
                         mapbase = 'map', maskname ='',realmin=0,realmax = -1):
    """
    Given stat name and filename info, reads in data from files
    that have been split into chunks, returns 2xN array, first column
    is realization number, second is stat for each realization.

    if realmin ==0 and realmax ==-1, will find all data matching that filenaming convention.
    otherwise will collect all available data in given range.
    """
    print "Collecting stat data for Rall"
    statname = 'Rall'
    if maskname:
        maskstr = '-'+maskname
    else:
        maskstr = ''
    print statname,datadir,mapbase,maskname
    flist = glob.glob('{0:s}{1:s}.{2:s}{3:s}_*.dat'.format(datadir,statname,mapbase,maskstr))
    #print flist
    rmins = []
    rmaxs = []
    for f in flist:
        rmin,rmax = get_realnum_from_statfname(f)
        rmins.append(rmin)
        rmaxs.append(rmax)
    #print rmins,rmaxs
    if (realmax<0) or (realmax>max(rmaxs)):
        realmax = max(rmaxs)
    if realmin < min(rmins):
        realmin = min(rmins) #don't try to read in data you don't have
    rmins = np.array(rmins)
    #print 'rmins:',rmins
    sortmin = np.argsort(rmins)
    #print 'rmaxs:',rmaxs
    rmaxs = np.array(rmaxs)
    sortmax = np.argsort(rmaxs)
    if not np.all(sortmin==sortmax):
        print " WARNING; overlapping realization ranges; may be duplicating data "
        
    outdat = np.zeros(( realmax - realmin +1,60))
    startonrow=0
    for i in sortmin:
        #print flist[i]
        dati = np.loadtxt(flist[i],skiprows=1)
        Ni = dati.shape[0]
        outdat[startonrow:startonrow+Ni,:] = dati
        startonrow += Ni
    #print outdat
    return outdat    
##################################################################
def get_stats_formaplist(realmin,realmax,maskfile='',maskname='', \
                         datadir = 'output/lcdm-map-testspace/',\
                         statdir = '',\
                         mapbase = 'map',\
                         redoCl = True, redoCt = True, \
                         doS12 = True, doR10 = False, doR27 = True, doR50=True,\
                         doRall = False, doCl2 = True, doCl3 = True, doCt180=True,\
                         doalign23S = True,\
                         dos16 = True, mapnside = NSIDEfid):
    """
    Given min and max realization number,
    loops through maps with realization realmin - realmax,
    looking for .fits files, 
    extracts measured C_l and  C(theta) from that map, 
    saves to text files with same name as original
    but with .fits -> .cl.dat and .ct.dat

    If maskfile is empty string, assumes full sky.
    maskname is a shorter string used to indicate which mask was used
    in the output filenames

    If bools for stats are on, will create an ouptut file
    containing those stats for the given range of realizations
    
    datadir is where to find maps, and to put C_l and C(theta)
    statdir is where to put output files for computed stats; if empty
        is made equal to datadir

    mapnside - used to correct cls for pixel and beam window functions
    """
    rlzns = np.arange(realmin,realmax+1)
    maplist = [get_filename_testcase(datadir = datadir, mapbase = mapbase, number =i, stattype = 'map') for i in rlzns]

    if not statdir:
        statdir = datadir
    
    if maskfile:
        #check that the file exists
        assert os.path.isfile(maskfile), "Mask file doesn't exist!"
        if not maskname:
            maskname = maskfile[maskfile.rfind('/')+1:]
        maskstr = '-'+maskname
    else:
        maskstr = ''
    clfilelist = []
    ctfilelist = []
    
    Nmap = len(maplist)

    #set up files etc for stats we want to compile
    if doS12: #S_1/2 measure, tells you about low power at angles > 60 deg
        print 'setting up S12 for {0}-{1}'.format(realmin,realmax)
        Itab = np.array([])
        S12dat = np.ones((Nmap,2))*np.nan
        S12dat[:,0] = rlzns
        S12f = get_filename_forstat(datadir = statdir, mapbase = mapbase, rmin = realmin, rmax = realmax, stattype = 'S12', maskname = maskname)
    if doR10: #parity assymmetry measurement with ell_max =10 
        R10dat = np.ones((Nmap,2))*np.nan
        R10dat[:,0] = rlzns
        R10f = get_filename_forstat(datadir = statdir, mapbase = mapbase, rmin = realmin, rmax = realmax, stattype = 'R10', maskname = maskname)
    if doR27: #parity assymmetry measurement with ell_max = 28
        R27dat = np.ones((Nmap,2))*np.nan
        R27dat[:,0] = rlzns
        R27f = get_filename_forstat(datadir = statdir, mapbase = mapbase, rmin = realmin, rmax = realmax, stattype = 'R27', maskname = maskname)
    if doR50: #parity assymmetry measurement with ell_max = 50
        R50dat = np.ones((Nmap,2))*np.nan
        R50dat[:,0] = rlzns
        R50f = get_filename_forstat(datadir = statdir, mapbase = mapbase, rmin = realmin, rmax = realmax, stattype = 'R50', maskname = maskname)
    if doRall: #party assymmetry meas for many ell_max, from 2-60
        Ralldat = np.ones((Nmap,1+59))
        Ralldat[:,0] = rlzns
        Rallf = get_filename_forstat(datadir = statdir, mapbase = mapbase, rmin = realmin, rmax = realmax, stattype = 'Rall', maskname = maskname)
    if doCt180: #Ct for theta=180
        Ct180dat = np.ones((Nmap,2))*np.nan
        Ct180dat[:,0] = rlzns
        Ct180f = get_filename_forstat(datadir = statdir, mapbase = mapbase, rmin = realmin, rmax = realmax, stattype = 'Ct180', maskname = maskname)
    if doCl2: #Cl for ell=2 (quadrupole)
        Cl2dat = np.ones((Nmap,2))*np.nan
        Cl2dat[:,0] = rlzns
        Cl2f = get_filename_forstat(datadir = statdir, mapbase = mapbase, rmin = realmin, rmax = realmax, stattype = 'Cl2', maskname = maskname)
    if doCl3: #Cl for ell=3 (octupole)
        Cl3dat = np.ones((Nmap,2))*np.nan
        Cl3dat[:,0] = rlzns
        Cl3f = get_filename_forstat(datadir = statdir, mapbase = mapbase, rmin = realmin, rmax = realmax, stattype = 'Cl3', maskname = maskname)
    if doalign23S:
        if maskfile:
            print "WARNING: alignment stat computation not implemented yet for masked maps"
        else:
            align23Sdat = np.ones((Nmap,2))*np.nan
            align23Sdat[:,0] = rlzns
            align23Sf = get_filename_forstat(datadir = statdir, mapbase = mapbase, rmin = realmin, rmax = realmax, stattype = 'align23S', maskname = maskname)
    if dos16:
        s16dat = np.ones((Nmap,2))*np.nan
        s16dat[:,0] = rlzns
        s16f = get_filename_forstat(datadir = statdir, mapbase = mapbase, rmin = realmin, rmax = realmax, stattype = 's16', maskname = maskname)

    missingmaps = []
    
    for i in xrange(Nmap):
        mapf = maplist[i]
        if not os.path.isfile(mapf) and any([redoCl,doalign23S,dos16]):
            print "Missing map file, skipping:",mapf
            missingmaps.append(mapf)
            continue

        
        outclfile = get_filename_testcase(datadir = datadir, mapbase = mapbase, number =realmin+i, stattype = 'cl', maskname = maskname)
        outctfile = get_filename_testcase(datadir = datadir, mapbase = mapbase, number =realmin+i, stattype = 'ct', maskname = maskname)
        print 'calling getCl_fromSpice'
        #cl = getCl_fromSpice(mapf,maskfile,outclfile=outclfile,overwrite=redoCl)
        if maskfile:
            cl = getCl_fromSpice(mapf,maskfile,outclfile=outclfile,overwrite=redoCl)
        else:
            cl = getCl_anafast(mapf,outclfile = outclfile, overwrite = redoCl)

        # get scaling factors to correct cls for pixel and beams
        if i==0: #only get factors once
            if mapnside is not None:
                pl = hp.sphtfunc.pixwin(mapnside)[:cl.size]
                fwhm = arcmin2rad(NSIDEtoFWHMarcmin[mapnside])
                bl = hp.sphtfunc.gauss_beam(fwhm,lmax=cl.size-1)
                scaleclsby = 1./(bl*pl)**2.
            else:
                scaleclsby = 1.
            
        lmaxct = min(100,cl.size-1)
        ct = getCtheta(cl, outctfile, overwrite = redoCt,LMAX=lmaxct)
        clfilelist.append(outclfile)
        ctfilelist.append(outctfile)
        if doS12:
            #print 'doing S12 for {0}-{1}'.format(realmin,realmax)
            lmax = 100#cl.size - 1
            if not Itab.size:
                Itab = tabulate_Ifunc(0.5,lmax)
            S12dat[i,1] = getSmeas(cl,0.5,LMAX=lmax, Itab = Itab)
        if doR10:
            #R10dat[i,1] = get_Rassymstat(cl,lmax=10,NSIDE=parityNside)
            R10dat[i,1] = get_Rassymstat(cl*scaleclsby,lmax=10)
        if doR27:
            #R27dat[i,1] = get_Rassymstat(cl,lmax=28,NSIDE=parityNside)
            R27dat[i,1] = get_Rassymstat(cl*scaleclsby,lmax=27)
        if doR50:
            #R50dat[i,1] = get_Rassymstat(cl,lmax=50,NSIDE=parityNside)
            R50dat[i,1] = get_Rassymstat(cl*scaleclsby,lmax=50)
        if doRall:
            Ralldat[i,1:] = get_Rassym_allell(cl*scaleclsby,lmax=60)
            #Ralldat[i,1:] = get_Rassym_allell(cl,lmax=60,NSIDE=parityNside)
            #Ralldat[i,1:] = get_Rassym_allell(cl,lmax=60,NSIDE=NSIDEfid)
        if doCt180:
            Ct180dat[i,1] = ct[-1,1]
        if doCl2:
            Cl2dat[i,1] = cl[2]*scaleclsby[2]
        if doCl3:
            Cl3dat[i,1] = cl[3]*scaleclsby[3]
        if doalign23S and (not maskfile):
            align23Sdat[i,1] = get_l23_alignment_S(mapf)
        if dos16:
            s16dat[i,1] = get_lowres_mapvar(mapf,maskfile,Nsideout=16)
    if doS12:
        print "Saving ",S12f
        np.savetxt(S12f,S12dat,header = 'realization, S_1/2')
    if doR10:
        print "Saving ",R10f
        np.savetxt(R10f,R10dat,header = 'realization, R_10 (parity assym measure for lmax=10)')
    if doR27:
        print "Saving ",R27f
        np.savetxt(R27f,R27dat,header = 'realization, R_28 (parity assym measure for lmax=27)')
    if doR50:
        print "Saving ",R50f
        np.savetxt(R50f,R50dat,header = 'realization, R_50 (parity assym measure for lmax=50)')
    if doRall:
        print "Saving ",Rallf
        np.savetxt(Rallf,Ralldat,header = 'realization, R_lmax (parity assym measure for lmax=2-60)')
    if doCt180:
        print "Saving ",Ct180f
        np.savetxt(Ct180f,Ct180dat,header = 'realization, C(theta=180) ')
    if doCl2:
        print "Saving ",Cl2f
        np.savetxt(Cl2f,Cl2dat,header = 'realization, C_[ell=2] ')
    if doCl3:
        print "Saving ",Cl3f
        np.savetxt(Cl3f,Cl3dat,header = 'realization, C_[ell=3] ')
    if doalign23S and (not maskfile):
        print "Saving ",align23Sf
        np.savetxt(align23Sf,align23Sdat,header='realization, alignment stat S for ell=2,3')
    if dos16:
        print "Saving ",s16f
        np.savetxt(s16f,s16dat,header='realization, variance of unmasked pixels in NSIDE=16 map')
    print "Missing {0:d} maps in total".format(len(missingmaps))
    return missingmaps

               
##################################################################
# Plotting functions
##################################################################
def get_onemap_stats(datadir= 'output/lcdm-map-testspace/',\
                     mapbase = 'smica_0064_full',\
                     maskf = 'data/masks/COM_Mask_CMB-IQU-UT78_0064_R2.01.fits', maskname = 'UT76',\
                     statlist =['S12','R27','R50','Cl2','Cl3','Ct180','s16'],overwrite=False , lvmapdir = '',  ALVmeanmapf ='', ALVvarmapf = '',savedat=False, mapnside = NSIDEfid):
    """
    Given a map and mask returns list of stat values for 
    stats specified in statlist. (for e.g. plotting planck datapoints)

    lvmapdir, ALVmeanmapf, ALVvarmaf only needed if ALV is in statlist
    """
    statf = get_filename_testcase(datadir = datadir, mapbase = mapbase,  maskname = maskname,number =0, stattype = 'statsummary')
    #print maskf
    if os.path.isfile(statf) and not overwrite:
        print "reading in stat summary data from",statf
        f = open(statf)
        firstline = f.readline()
        f.close()
        valsinfile = np.loadtxt(statf,skiprows=1)
        statsinfile = firstline.split()[1:]
        #print statsinfile
        outvals = []
        missingany= False
        for s in statlist:
            if s in statsinfile:
                i = statsinfile.index(s)
                outvals.append(valsinfile[i])
            else:
                missingany = True
                break
        if not missingany:
            return outvals
        else:
            print "  ...all requested stats not here, recomputing."
        
    mapf = get_filename_testcase(datadir = datadir, mapbase = mapbase, maskname = '', number =0, stattype = 'map')
    #print mapf
    clfile = get_filename_testcase(datadir = datadir, mapbase = mapbase, maskname = maskname, number =0, stattype = 'cl')
    ctfile = get_filename_testcase(datadir = datadir, mapbase = mapbase,  maskname = maskname,number =0, stattype = 'ct')


    cl = getCl_fromSpice(mapf,maskf,clfile,overwrite = overwrite)
    #if maskf:
    #    cl = getCl_fromSpice(mapf,maskf,clfile,overwrite = overwrite)
    #else:
    #    cl = getCl_anafast(mapf,outclfile = clfile,overwrite=overwrite)
    if mapnside is not None:
        pl = hp.sphtfunc.pixwin(mapnside)[:cl.size]
        fwhm = arcmin2rad(NSIDEtoFWHMarcmin[mapnside])
        bl = hp.sphtfunc.gauss_beam(fwhm,lmax=cl.size-1)
        scaleclsby = 1./(bl*pl)**2.
    else:
        scaleclsby = 1.
            
    #cl = cl*scaleclsby
    

    ct = getCtheta(cl,ctfile,overwrite= overwrite)

    outvals = [] #will have list of stat values, same order as statlist
    for stat in statlist:
        if stat=='S12':
            outvals.append(getSmeas(cl,0.5,LMAX=100))
        elif stat == 'R10':
            #outvals.append(get_Rassymstat(cl,lmax=10,NSIDE=parityNside))
            outvals.append(get_Rassymstat(cl*scaleclsby,lmax=10))
        elif stat == 'R27':
            #outvals.append(get_Rassymstat(cl,lmax=28,NSIDE=parityNside))
            outvals.append(get_Rassymstat(cl*scaleclsby,lmax=27))
        elif stat == 'R50':
            #outvals.append(get_Rassymstat(cl,lmax=50,NSIDE=parityNside))
            outvals.append(get_Rassymstat(cl*scaleclsby,lmax=50))
        elif stat == 'Cl2':
            outvals.append(cl[2]*scaleclsby[2])
        elif stat == 'Cl3':
            outvals.append(cl[3]*scaleclsby[3])
        elif stat == 'Ct180':
            outvals.append(ct[-1,1])
        elif stat == 's16':
            outvals.append(get_lowres_mapvar(mapf,maskf,Nsideout=16))
        elif stat == 'align23S':
            if maskf:
                print "WARNING: have not implemented align23S stat computation for non-fullsky maps, pulling numbers from full sky with no mask"
            outvals.append(get_l23_alignment_S(mapf))
        elif stat == 'ALV':
            ALV = get_ALV_onemap_externalmean(datadir = datadir, mapbase = mapbase, meanmapfile = ALVmeanmapf, varmapfile = ALVvarmapf, lvmapdir = lvmapdir, maskfile = maskf, maskname = maskname, disksizedeg = 8, NSIDELV = 16,overwrite = overwrite)
            outvals.append(ALV)
        else:
            print "STAT {0:s} NOT RECOGNIZED.".format(stat)
    if savedat:
        print "Saving stat summary data to ",statf
        np.savetxt(statf,outvals,header=' '.join(statlist))
    return outvals

def get_onemap_Rall(datadir= 'output/lcdm-map-testspace/',\
                     mapbase = 'smica_0064_full',\
                     maskf = 'data/masks/COM_Mask_CMB-IQU-UT78_0064_R2.01.fits', maskname = 'UT76',\
                    overwrite=False , savedat=True, mapnside = NSIDEfid):#, NSIDE=NSIDEfid,NSIDEin=NSIDEfid):
    """
    Given a map and mask returns array of R_lmax, for lmax going grom 2 to 60
    """
    statf = get_filename_testcase(datadir = datadir, mapbase = mapbase,  maskname = maskname,number =0, stattype = 'Rall')
    mapf = get_filename_testcase(datadir = datadir, mapbase = mapbase,  maskname = '',number =0, stattype = 'map')
    #print statf,mapf
    #print maskf
    if os.path.isfile(statf) and not overwrite:
        print "reading in stat summary data from",statf
        outvals  = np.loadtxt(statf,skiprows=1) # Nell sized array
        return outvals
    clfile = get_filename_testcase(datadir = datadir, mapbase = mapbase, maskname = maskname, number =0, stattype = 'cl')

    #cl = getCl_fromSpice(mapf,maskf,clfile,overwrite = overwrite)
    if maskf:
        cl = getCl_fromSpice(mapf,maskf,clfile,overwrite = overwrite)
    else:
        cl = getCl_anafast(mapf,outclfile = clfile,overwrite=overwrite)
    if mapnside is not None:
        pl = hp.sphtfunc.pixwin(mapnside)[:cl.size]
        fwhm = arcmin2rad(NSIDEtoFWHMarcmin[mapnside])
        bl = hp.sphtfunc.gauss_beam(fwhm,lmax=cl.size-1)
        scaleclsby = 1./(bl*pl)**2.
    else:
        scaleclsby = 1.
    #cl = cl*scaleclsby
    
    outvals = get_Rassym_allell(cl*scaleclsby)#,NSIDE=NSIDE,NSIDEin=NSIDEin)
        
    if savedat:
        print "Saving stat summary data to ",statf
        np.savetxt(statf,outvals,header=' R_lmax assymmetry stat for lmax = 2-60')
    return outvals

def get_stats_fortheorycl(theoryclfile =  "data/COM_PowerSpect_CMB-base-plikHM-TT-lowTEB-minimum-theory_R2.02.txt",\
                        ctfile = "'output/lcdm-map-testspace/theory.ct.dat",\
                          statlist =['S12','R27','R50','Cl2','Cl3','Ct180'],lmax=200,overwriteCt=False):
    thdat = np.loadtxt(theoryclfile,skiprows=1)
    ellth = np.arange(lmax+1,dtype=int)
    ellth[2:] = thdat[:lmax-1,0]
    clth = np.zeros(lmax+1)
    clth[2:] = thdat[:lmax-1,1]*2.*np.pi/(ellth[2:]*(ellth[2:]+1))
    ctth = getCtheta(clth,ctfile,LMAX = min(100,clth.size-1))
    #scale by appropriate window function for pixelization
    # edit 3/28/18; switchign to correct maps, not theory
    # plth = hp.sphtfunc.pixwin(NSIDEfid)[:lmax+1]
    # fwhmin = arcmin2rad(NSIDEtoFWHMarcmin[NSIDEfid])
    # blth = hp.sphtfunc.gauss_beam(fwhmin,lmax=lmax)
    # clth = clth*(plth*blth)**2
    # ctth = getCtheta(clth,ctfile,overwrite=overwriteCt)
    
    outvals = [] #will have list of stat values, same order as statlist
    for stat in statlist:
        if stat=='S12':
            outvals.append(getSmeas(clth,0.5,LMAX=100))
        elif stat == 'R10':
            #outvals.append(get_Rassymstat(clth,lmax=10,NSIDE=parityNside,NSIDEin=parityNsidein))
            outvals.append(get_Rassymstat(clth,lmax=10))
        elif stat == 'R27':
            #outvals.append(get_Rassymstat(clth,lmax=28,NSIDE=parityNside,NSIDEin=parityNsidein))
            outvals.append(get_Rassymstat(clth,lmax=27))
        elif stat == 'R50':
            #outvals.append(get_Rassymstat(clth,lmax=50,NSIDE=parityNside,NSIDEin=parityNsidein))
            outvals.append(get_Rassymstat(clth,lmax=50))
        elif stat == 'Cl2':
            outvals.append(clth[2])
        elif stat == 'Cl3':
            outvals.append(clth[3])
        elif stat == 'Ct180':
            outvals.append(ctth[-1,1])
        elif stat == 's16':
            #compute theory prediction for variance given cls
            pl = hp.sphtfunc.pixwin(16)[:clth.size]
            fwhm = arcmin2rad(NSIDEtoFWHMarcmin[16])
            bl = hp.sphtfunc.gauss_beam(fwhm,lmax=pl.size-1)
            thvar = (1./(4.*np.pi))*np.sum((bl*pl)**2*clth[:pl.size]*(2*ellth[:pl.size]+1))
            outvals.append(thvar)
        elif stat in ('align23S','ALV'):
            print "WARNING: map-based stats not defined for theory calc"
            outvals.append(np.nan)
        else:
            print "STAT {0:s} NOT RECOGNIZED.".format(stat)
    return outvals

def get_Rall_fortheorycl(theoryclfile =  "data/COM_PowerSpect_CMB-base-plikHM-TT-lowTEB-minimum-theory_R2.02.txt",lmax=60,\
                         outdir = 'output/planckstats/',baselabel='planckth',overwrite=False,NSIDE=2048,NSIDEin=2048):
    #print theoryclfile
    outf = get_filename_testcase(datadir = outdir, mapbase = baselabel,  number =0, stattype = 'Rall')

    if os.path.isfile(outf) and not overwrite:
        print "reading in data from",outf
        outvals  = np.loadtxt(outf,skiprows=1) 
    else:    
        thdat = np.loadtxt(theoryclfile,skiprows=1)
        ellth = np.arange(lmax+1,dtype=int)
        ellth[2:] = thdat[:lmax-1,0]
        clth = np.zeros(lmax+1)
        clth[2:] = thdat[:lmax-1,1]*2.*np.pi/(ellth[2:]*(ellth[2:]+1)) #starts at 0, but has 0 in first few entries

        #print clth
        outvals = get_Rassym_allell(clth,lmax)
        #outvals = get_Rassym_allell(clth,lmax,NSIDE=NSIDE,NSIDEin=NSIDEin)
        print "saving data to",outf
        np.savetxt(outf,outvals,header=' R_lmax assymmetry stat for lmax = 2-60 for NSIDE={0:d}'.format(NSIDE))
    return outvals

def get_Rall_formolinaricl(clfile='/Users/jlmuir/workspace/cmbparity/data/CLTILDE001_smica_molinari.DAT',lmax=60,outdir='output/planckstats/',baselabel='planckmolinari',overwrite=False):
    outf = get_filename_testcase(datadir = outdir, mapbase = baselabel,  number =0, stattype = 'Rall')
    if os.path.isfile(outf) and not overwrite:
        print "reading in data from",outf
        outvals  = np.loadtxt(outf,skiprows=1) 
    else:    
        dat = np.loadtxt(clfile)
        ell = np.arange(lmax+1,dtype=int)
        clth = np.zeros(lmax+1)
        clth[2:] = dat[:lmax-1,1]*2.*np.pi/(ell[2:]*(ell[2:]+1)) 
        #print clth
        outvals = get_Rassym_allell(clth,lmax)
        #outvals = get_Rassym_allell(clth,lmax,NSIDE=NSIDE,NSIDEin=NSIDEin)
        print "saving data to",outf
        np.savetxt(outf,outvals,header=' R_lmax assymmetry stat for lmax = 2-60')
    return outvals

#------------------
def plot_cl_and_ct(datadir = '/Users/Shared/lcdm_NSIDE64_maps_for_cmbparity/lcdm/',\
                   mapbase = 'map',maskname = 'UT76', realmin=0, realmax=100,\
                   maskf = 'data/masks/COM_Mask_CMB-IQU-UT78_0064_R2.01.fits',\
                   theorylabel='planck best fit theory',\
                   theoryclfile = "data/COM_PowerSpect_CMB-base-plikHM-TT-lowTEB-minimum-theory_R2.02.txt",\
                   theoryctfile = "output/lcdm-map-testspace/theory.ct.dat",\
                   plancklabel = 'smica+UT76', \
                   planckdatadir ='output/lcdm-map-testspace/',\
                   planckmapbase = 'smica_0064_full',\
                   planckmaskname = 'UT76',\
                   savefig = True, outdir = 'output/lcdm_anomalycov_plots/',\
                   outtag = ''):

    if maskname:
        maskstr = '-'+maskname
    else:
        maskstr = ''

    #get planck data
    pmapf = get_filename_testcase(datadir = datadir, mapbase = planckmapbase, maskname = '', number =0, stattype = 'map')
    pclfile = get_filename_testcase(datadir = planckdatadir, mapbase = planckmapbase, maskname = planckmaskname, number =0, stattype = 'cl')
    pcl = readSpice_clfile(pclfile) #just read it in
    print pclfile
    pctfile = get_filename_testcase(datadir = planckdatadir, mapbase = planckmapbase,  maskname = planckmaskname,number =0, stattype = 'ct')
    pct = getCtheta(pcl,pctfile,overwrite=False)

    #get theory data
    thlmax = 200
    thdat = np.loadtxt(theoryclfile,skiprows=1)
    ellth = np.arange(thlmax+1,dtype=int)
    ellth[2:] = thdat[:thlmax-1,0]
    clth = np.zeros(thlmax+1)
    clth[2:] = thdat[:thlmax-1,1]*2.*np.pi/(ellth[2:]*(ellth[2:]+1))
    # changing to correct maps, not theory
    # plth = hp.sphtfunc.pixwin(NSIDEfid)[:thlmax+1]
    # fwhmin = arcmin2rad(NSIDEtoFWHMarcmin[NSIDEfid])
    # blth = hp.sphtfunc.gauss_beam(fwhmin,lmax=thlmax)
    # clth = clth*(plth*blth)**2
    ctth = getCtheta(clth,theoryctfile,overwrite=False)
        
    # Get cl, ct for desired realizations
    real = np.arange(realmin,realmax+1)
    clfiles = ['{0:s}{1:s}{2:s}_{3:d}.cl.dat'.format(datadir,mapbase,maskstr,realmin+i) for i in real]
    ctfiles = ['{0:s}{1:s}{2:s}_{3:d}.ct.dat'.format(datadir,mapbase,maskstr,realmin+i) for i in real]
    
    cl = [readSpice_clfile(f) for f in clfiles]
    ct = [np.loadtxt(f,skiprows=1) for f in ctfiles]
    Nf = len(clfiles)
    ell = np.arange(cl[0].size)
    fig, (ax0,ax1) = plt.subplots(1,2,figsize=(11,5))
    plt.subplots_adjust( wspace = .3)
    ax1.axhline(0,color='gray')

    for i in xrange(Nf):
        ax0.plot(ell,cl[i]*ell*(2*ell+1)/(2.*np.pi),alpha=.1)
        ax1.plot(ct[i][:,0],ct[i][:,1],alpha=.1)
        
    ax0.plot(ellth[2:],clth[2:]*ellth[2:]*(2*ellth[2:]+1)/(2.*np.pi),color='red',linewidth=2,linestyle = '--',label=theorylabel)
    ax0.plot(ell,pcl*ell*(2*ell+1)/(2.*np.pi),color='black',linewidth=1,linestyle = '-',label=plancklabel)
        
    ax1.plot(ctth[2:,0],ctth[2:,1],color='red',linewidth=2,linestyle = '--',label=theorylabel)
    ax1.plot(pct[:,0],pct[:,1],color='black',linewidth=1,linestyle = '-',label=plancklabel)
    #ax0.set_ylim((-100,3500))
    ax0.set_ylim((.01,3500))
    ax0.set_yscale("log", nonposy='clip')
    ax1.set_ylim((-1000,2000))
    #ax0.set_ylim((-100,6000))
    #ax1.set_ylim((-1000,2.e6))
    ax1.set_ylabel(r'$C(\theta)$',fontsize=16)    
    ax1.set_xlabel(r'$\theta$',fontsize=16) 

        
    if maskname:
        ax0.set_title('mask:'+maskname)
    else:
        ax0.set_title('no mask')
    ax0.set_ylabel(r'$\ell(2\ell+1)C_{\ell}/2\pi$',fontsize=16)    
    ax0.set_xlabel(r'$\ell}$',fontsize=16) 
    ax0.legend()

    if savefig:
        outname ='angcorr_{0:s}{1:s}_{2:s}'.format(mapbase,maskstr,outtag)
        outf = outdir+outname+'.png'
        print "saving",outf
        plt.savefig(outf)
    else:
        plt.show()

#------------------
# following https://jakevdp.github.io/blog/2014/06/14/frequentism-and-bayesianism-4-bayesian-in-python/
def compute_sigma_level(trace1, trace2, nbins=None, smooth=False):
    """From a set of traces, bin by number of standard deviations"""
    Npoints = trace1.size
    if nbins is None:
        nbisn = int(np.sqrt(Npoints)/10.)
    L, xbins, ybins = np.histogram2d(trace1, trace2, nbins)
    if smooth:
        L =ndimage.gaussian_filter(L, smooth, mode='reflect')
    # L is count in bin
    shape = L.shape
    L = L.ravel()

    # obtain the indices to sort and unsort the flattened array
    i_sort = np.argsort(L)[::-1] #indices to sort L in decreasing order
    i_unsort = np.argsort(i_sort)

    L_cumsum = L[i_sort].cumsum()
    L_cumsum /= L_cumsum[-1]
    #each entry has fraction of total number of scatter points in that bin
    # or in bins with higher density
    
    xbins = 0.5 * (xbins[1:] + xbins[:-1])
    ybins = 0.5 * (ybins[1:] + ybins[:-1])

    return xbins, ybins, L_cumsum[i_unsort].reshape(shape)
#------------------

#------------------
def plot_1dhist_summary(statslcdm,statsffp,thvals= None,qmlvals= None,\
                        plfullvals_lvlcdm= None,plfullvals_lvffp= None,plut78vals_lvlcdm= None,plut78vals_lvffp= None,\
                        p_qml_lcdm= None,p_qml_ffp = None, p_ana_lcdm = None,p_ana_ffp= None,p_ut78_lcdm= None ,p_ut78_ffp= None ,\
                        molqmlvals = None, statsmolffp=None, p_molqml_lcdm =None, p_molqml_ffp = None,\
                        histlabel='Full- or cut-sky pseudo-$C_{\ell}$ from sims.',\
                        statlist=['S12','Cl2','Cl3','s16','R27','Ct180','align23S','ALV'],\
                        starby = ['cut','qml','qml','cut','qml', 'cut',   'full', 'cut'],\
                        whichmeas = None,outdir =  'lcdm_anomalycov_plots/', outname = 'onedim_stathists_TEST.pdf',\
                        showmol=False,skipstats = [],statlabels = None, standardlines = True,altline = False,\
                        altvals_lcdm = None, altvals_ffp = None, altpvals_lcdm = None, altpvals_ffp = None, \
                        altlabel = None, altcol = 'red',\
                        lcdmhistlabel = r'$10^5$ synfast sims',ffphistlabel = r'$10^3$ FFP8.1 sims',pstrffp = "p={0:0.1f}%"):
    histcol = 'gray'
    molhistcol = 'red'
    theorycol = 'black'
    anafastcol ='green'
    ut78col = 'darkorange'#'#66a61e'#'green'
    qmlcol = 'blue'
    molqmlcol='red'
    if statlabels is None:
        statlabels =[texdict[s] for s in statlist]
    Nstat = len(statlist)
    #f, axes = plt.subplots(Nstat,2,figsize=(8,Nstat*1.1 +1))
    f, axes = plt.subplots(Nstat,2,figsize=(7.5,8.5))
    plt.subplots_adjust(left=.1, bottom=0.05, right=.97, top=.85,wspace=0.05, hspace=.5)
    for i in xrange(Nstat):
        axlcdm = axes[i,0]
        axffp = axes[i,1]
        if i==0:
            axlcdm.set_title(lcdmhistlabel,fontsize=16)
            axffp.set_title(ffphistlabel,fontsize=16)
        s = statlist[i]
        #print 'on stat ',s
        if s in skipstats:
            continue
        axlcdm.set_ylabel(statlabels[i],fontsize=16,rotation=0,labelpad=24)
        axlcdm.get_yaxis().set_ticks([]) #hide tick marks
        axffp.get_yaxis().set_ticks([])
        
        if (s in ('S12')) and (s not in ['1','2']):
            plotdatlcdm = np.log(statslcdm[i][:,1])
            plotdatffp = np.log(statsffp[i][:,1])
            if standardlines:
                plotth = np.log(thvals[i])
                plotqml = np.log(qmlvals[i])
                if showmol:
                    plotmolqml = np.log(molqmlvals[i])
                plotpl_lvlcdm = np.log(plfullvals_lvlcdm[i]) #planck full sky from anafast
                plotpl_lvffp = np.log(plfullvals_lvffp[i]) #planck full sky from anafast
                plotplut78_lvlcdm = np.log(plut78vals_lvlcdm[i])
                plotplut78_lvffp = np.log(plut78vals_lvffp[i])

            if altline:
                plotalt_lcdm = np.log(altvals_lcdm[i])
                plotalt_ffp = np.log(altvals_ffp[i])
            if showmol and type(statsmolffp[i])==np.ndarray:
                domolhist = True
                plotdatmolffp = np.log(statsmolffp[i][:,1])
            else: 
                domolhist=False
                plotdatmolffp = np.nan*np.ones(plotdatffp.size)
        else:
            plotdatlcdm = statslcdm[i][:,1]
            plotdatffp = statsffp[i][:,1]
            if standardlines:
                plotth = thvals[i]
                plotqml = qmlvals[i]
                if showmol:
                    plotmolqml = molqmlvals[i]
                plotpl_lvlcdm = plfullvals_lvlcdm[i] #planck full sky from anafast
                plotpl_lvffp = plfullvals_lvffp[i] #planck full sky from anafast
                plotplut78_lvlcdm = plut78vals_lvlcdm[i]
                plotplut78_lvffp = plut78vals_lvffp[i]

            if altline:
                plotalt_lcdm = altvals_lcdm[i]
                plotalt_ffp = altvals_ffp[i]
            if showmol and type(statsmolffp[i])==np.ndarray:
                domolhist = True
                plotdatmolffp = statsmolffp[i][:,1]
            else:
                domolhist=False
                plotdatmolffp = np.nan*np.ones(plotdatffp.size)
        
        #set x lims
        xmax = max(np.max(plotdatlcdm),np.max(plotdatffp))
        xmin = min(np.min(plotdatlcdm),np.min(plotdatffp))
        if altline:
            xmax = max(xmax,plotalt_lcdm)
            xmax = max(xmax,plotalt_ffp)
            xmin = min(xmin,plotalt_lcdm)
            xmin = min(xmin,plotalt_ffp)
        if xmin>0:
            axlcdm.set_xlim((.9*xmin,1.1*xmax))
            axffp.set_xlim((.9*xmin,1.1*xmax))
        else:
            axlcdm.set_xlim((1.1*xmin,1.1*xmax))
            axffp.set_xlim((1.1*xmin,1.1*xmax))
        #plot lcdm hists and ines
        axlcdm.hist(plotdatlcdm,bins=50,color=histcol)
        if standardlines:
            axlcdm.axvline(plotth,color=theorycol,linestyle='--')
            axlcdm.axvline(plotqml,color=qmlcol,linestyle='-')
            if showmol:
                axlcdm.axvline(plotmolqml,color=molqmlcol,linestyle='-')
            axlcdm.axvline(plotpl_lvlcdm,color=anafastcol,linestyle='-')
            if statlist[i]!='align23S':
                axlcdm.axvline(plotplut78_lvlcdm,color=ut78col,linestyle='-')

        axffp.hist(plotdatffp,bins=50,color=histcol,label=histlabel)
        if domolhist:
            axffp.hist(plotdatmolffp,bins=50,label='QML meas of sim (from Molinari)',histtype='step',color = molhistcol)
        if standardlines:
            axffp.axvline(plotth,color=theorycol,linestyle='--',label='Planck best fit theory')
            axffp.axvline(plotqml,color=qmlcol,linestyle='-',label=r"Planck public QML $C_{\ell}$'s")
            if showmol:
                axffp.axvline(plotmolqml,color=molqmlcol,linestyle='-',label=r"Planck Molinari QML $C_{\ell}$'s")
            axffp.axvline(plotpl_lvffp,color=anafastcol,linestyle='-',label="Full-sky pseudo-$C_{\ell}$'s")
            if statlist[i]!='align23S':
                axffp.axvline(plotplut78_lvffp,color=ut78col,linestyle='-',label="UT78 pseudo-$C_{\ell}$'s")
        if altline:
            axlcdm.axvline(plotalt_lcdm,color=altcol,linestyle='--')
            axffp.axvline(plotalt_ffp,color=altcol,linestyle='--',label=altlabel)

        if i==0:
            axffp.legend(bbox_to_anchor=(-.1, 1.7), loc='lower center', ncol=2,fontsize=11)
        
        #note which meas of hists was used
        if whichmeas is not None:
            axlcdm.annotate('('+whichmeas[i]+')',xy=(.99,.02),color = histcol,\
                        horizontalalignment='right',verticalalignment='bottom',xycoords='axes fraction')
            axffp.annotate('('+whichmeas[i]+')',xy=(.99,.02),color = histcol,\
                    horizontalalignment='right',verticalalignment='bottom',xycoords='axes fraction')
    
        #annotate with pvalues
        if p_ana_lcdm is not None:
            lcdmanastr = r'$\Rightarrow$'*(starby[i]=='full')+"p={0:0.3f}%".format(100*p_ana_lcdm[i])
            axlcdm.annotate(lcdmanastr,xy=(.99,.95),color = anafastcol,\
                        horizontalalignment='right',verticalalignment='top',xycoords='axes fraction')
        if p_ana_ffp is not None:
            ffpanastr = r'$\Rightarrow$'*(starby[i]=='full')+pstrffp.format(100*p_ana_ffp[i])
            axffp.annotate(ffpanastr,xy=(.99,.95),color = anafastcol,\
                       horizontalalignment='right',verticalalignment='top',xycoords='axes fraction')
        if (p_ut78_lcdm is not None) and (statlist[i]!='align23S'):
            lcdmut78str = r'$\Rightarrow$'*(starby[i]=='cut')+"p={0:0.3f}%".format(100*p_ut78_lcdm[i])
            axlcdm.annotate(lcdmut78str,xy=(.99,.6),color = ut78col,\
                            horizontalalignment='right',verticalalignment='center',xycoords='axes fraction')
        if (p_ut78_ffp is not None) and (statlist[i]!='align23S'):
            ffput78str = r'$\Rightarrow$'*(starby[i]=='cut')+pstrffp.format(100*p_ut78_ffp[i])
            axffp.annotate(ffput78str,xy=(.99,.6),color = ut78col,\
                           horizontalalignment='right',verticalalignment='center',xycoords='axes fraction')
        if (p_qml_lcdm is not None) and (not np.isnan(p_qml_lcdm[i])):
            lcdmqmlstr = r'$\Rightarrow$'*(starby[i]=='qml')+"p={0:0.3f}%".format(100*p_qml_lcdm[i])
            axlcdm.annotate(lcdmqmlstr,xy=(.99,.33),color = qmlcol,\
                            horizontalalignment='right',verticalalignment='center',xycoords='axes fraction')
        if (p_qml_ffp is not None) and (not np.isnan(p_qml_ffp[i])):
            ffpqmlstr = r'$\Rightarrow$'*(starby[i]=='qml')+pstrffp.format(100*p_qml_ffp[i])
            axffp.annotate(ffpqmlstr,xy=(.99,.33),color = qmlcol,\
                        horizontalalignment='right',verticalalignment='center',xycoords='axes fraction')
        if showmol:
            lcdmmolqmlstr = r'$\Rightarrow$'*(starby[i]=='molqml')+"p={0:0.3f}%".format(100*p_molqml_lcdm[i])
            ffpmolqmlstr = r'$\Rightarrow$'*(starby[i]=='molqml')+pstrffp.format(100*p_molqml_ffp[i])
        if altline:
            lcdmaltstr ="p={0:0.3f}%".format(100*altpvals_lcdm[i])
            axlcdm.annotate(lcdmaltstr,xy=(.99,.95),color = altcol,\
                        horizontalalignment='right',verticalalignment='top',xycoords='axes fraction')
            ffpaltstr =pstrffp.format(100*altpvals_ffp[i])
            axffp.annotate(ffpaltstr,xy=(.99,.95),color = altcol,\
                        horizontalalignment='right',verticalalignment='top',xycoords='axes fraction')
        if standardlines:
            if showmol:
                axlcdm.annotate(lcdmmolqmlstr,xy=(.99,.01),color = molqmlcol,\
                                horizontalalignment='right',verticalalignment='bottom',xycoords='axes fraction')
                axffp.annotate(ffpmolqmlstr,xy=(.99,.01),color = molqmlcol,\
                               horizontalalignment='right',verticalalignment='bottom',xycoords='axes fraction')
                
                
    outf = outdir + outname
    print "saving to",outf
    plt.savefig(outf)

#------------------
def plot_trianglecontours(statslist,statdat,simlabel='sims',pointlist = [],\
                          savefig = True, outdir = 'lcdm_anomalycov_plots/',\
                            nbins=None, outtag = '',smooth=False,center = False,unitless = False,statlabels = None):
    """
    This is like plot_triangle_scatter, but defined more flexibly: it takes
    stat data that has already been read in, and a variable number of 
    individual poitns that can be plotted. This is the function to be used
    for making plots for the paper.

    statslist is list of strings for statnames
    simdat is Nstat lenght list of 2xNreal-dim arrays (first col is realization number, 2nd is data)
    pointlist contains tuples of (label,statlist ,color,marker,markersize,linestyle)

    If center = True, data will be centered on the mean
    if unitless = True, data for each stat will be divided by standard deviation
    """
    #histcol = '#2171b5' #dark blue
    #contourcols = ('#bdd7e7','#6baed6','#2171b5') #blues light to dark
    histcol = '#cccccc' #light/medium grey
    #contourcols = ('#525252','#969696','#cccccc') #greys dark to light
    contourcols = ('#f7f7f7','#cccccc','#969696') #greys dark to light
    
    Nstat = len(statslist)
    Nreal = statdat[0].shape[0]
    if nbins is None:
        nbins = int(np.sqrt(1.*Nreal)/10.)
    print 'nbins=',nbins
    statdict = {statslist[i]:i for i in xrange(len(statslist))}
    texdict = {'S12':r'$\log{S_{\frac{1}{2}}}$','R10':r'$R_{10}$','R27':r'$R_{27}$','R50':r'$R_{50}$',\
               'Cl2':r'$C_2$','Cl3':r'$C_3$','Ct180':r'$C(\pi)$','align23S':r'$S_{QO}$','ALV':r'$A_{LV}$',\
               's16':r'$\sigma^2_{16}$'}
    if statlabels is None:
        statlabels = [texdict[s] for s in statslist]
    
    plotdat = np.zeros((Nstat,Nreal))
    #print plotdat.shape
    dontplot = []
    for i in xrange(Nstat):
        #print statslist[i],type(statdat[i])
        if type(statdat[i])==float and np.isnan(statdat[i]):
            plotdat[i,:] = np.nan*np.ones(Nreal)
            dontplot.append(i)
            continue
        if statslist[i]=='S12':
            plotdat[i,:] = np.log(statdat[i][:,1])
        else:
            plotdat[i,:] = statdat[i][:,1]

    statmeans = np.mean(plotdat,axis=1)
    statstds = np.std(plotdat,axis=1)
    #for i in xrange(Nstat):
    #    print statslist[i],statmeans[i],statstds[i],np.max(plotdat[i,:]),np.min(plotdat[i,:])
    #working here; need to handle zeros or Nan data
    if center:
        plotdat = np.array([(plotdat[:,n] - statmeans) for n in xrange(Nreal)]).T
        subval = statmeans
    else:
        subval = np.zeros(Nstat)
    if unitless:
        plotdat = np.array([plotdat[:,n]/statstds  for n in xrange(Nreal)]).T
        normval = statstds
    else:
        normval = np.ones(Nstat)
    statmins = np.min(plotdat,axis=1)
    statmaxs = np.max(plotdat,axis=1)
    statranges = statmaxs - statmins
    #print plotdat.shape,statmins.shape,statmaxs.shape
    #for i in xrange(Nstat):
    #    print statslist[i],subval[i],normval[i],statmins[i],statmaxs[i],statranges[i]

    if Nstat<=3:
        #print "doing little subplot"
        f,axes = plt.subplots(Nstat,Nstat,figsize=(1.+Nstat*1.5,1.+Nstat*1.5))
        plt.subplots_adjust(left=.17,right=0.95,top=0.95,bottom=0.11, wspace = .05, hspace = .05)
    else:
        f,axes = plt.subplots(Nstat,Nstat,figsize=(1.+Nstat*1.5,.5+Nstat*1.5))
        plt.subplots_adjust(left=.1,right=0.95,top=0.95,bottom=0.1, wspace = .05, hspace = .05)
    addonfrac = .1
    for i in xrange(Nstat):
        if i in dontplot:
            continue
        for j in xrange(Nstat):
            if j in dontplot:
                continue
            ax = axes[i,j]
            ax.set_xlim((statmins[j]-addonfrac*statranges[j],statmaxs[j]+addonfrac*statranges[j]))
            if i==Nstat-1:
                ax.set_xlabel(statlabels[j],fontsize=16)
            else:
                ax.get_xaxis().set_ticks([])
            if j==i:
                ax.get_yaxis().set_ticks([])
                if j==0:
                    #ax.set_ylabel('counts',fontsize=16)
                    ax.scatter([],[],marker='s',color=histcol,s=200,label=simlabel)
                    for  p in pointlist:
                        ax.scatter([],[],marker=p[3],color=p[2],s=p[4],label=p[0])
                    if Nstat<=3:
                        ax.legend(bbox_to_anchor=(1.,1),fontsize=14)
                    else:
                        ax.legend(bbox_to_anchor=(2.,1),fontsize=16)
                    ax.set_ylabel('Normalized \nCounts',fontsize=14)
                bins = np.linspace(statmins[i],statmaxs[j],nbins)
                ax.hist(plotdat[j,:],bins = bins,color=histcol)
                for p in pointlist:
                    if statslist[i]=='S12':
                        ax.axvline((np.log(p[1][j])-subval[j])/normval[j],color=p[2],linestyle=p[5])
                    else:
                        ax.axvline((p[1][j]-subval[j])/normval[j],color=p[2],linestyle=p[5])
                
            elif j<i:
                ax.set_ylim((statmins[i]-addonfrac*statranges[i],statmaxs[i]+3*addonfrac*statranges[i]))
                #print rij
                if j==0:
                    ax.set_ylabel(statlabels[i],fontsize=16,rotation=0,labelpad=20)
                    ax.set_ylim((statmins[i]-addonfrac*statranges[i],statmaxs[i]+3*addonfrac*statranges[i]))
                else:
                    ax.get_yaxis().set_ticks([])
                xdat = plotdat[j,:]
                ydat = plotdat[i,:]

                xbins,ybins,sigma = compute_sigma_level(xdat,ydat,nbins=nbins,smooth=smooth)
                siglevels = [1.-2.*stats.norm.sf(n) for n in range(len(contourcols)+1)]
                ax.contourf(xbins,ybins,sigma.T,levels=siglevels,\
                            colors = contourcols,antialiased=False)
                for p in pointlist:
                    if (p[1][j]!=np.nan) and (p[1][i]!=np.nan) :
                        if statslist[j]=='S12':
                            ax.scatter((np.log(p[1][j])-subval[j])/normval[j],(p[1][i]-subval[i])/normval[i],marker=p[3],color=p[2],s=p[4])
                        elif statslist[i] == 'S12':
                            ax.scatter((p[1][j]-subval[j])/normval[j],(np.log(p[1][i])-subval[i])/normval[i],marker=p[3],color=p[2],s=p[4])
                        else:
                            ax.scatter((p[1][j]-subval[j])/normval[j],(p[1][i]-subval[i])/normval[i],marker=p[3],color=p[2],s=p[4])
                    elif p[1][j]==np.nan:
                        ax.axhline((p[1][i]-subval[i])/normval[i],color=p[2],linestyle=p[5])
                    elif p[1][i]==np.nan:
                        ax.axvline((p[1][j]-subval[j])/normval[j],color=p[2],linestyle=p[5])
                rij = np.corrcoef(xdat,ydat)[0,1]
                ax.annotate('R={0:.3f}'.format(rij), xy=(.05,0.95),  color='k',fontsize=16,\
                            xycoords='axes fraction',horizontalalignment='left', verticalalignment='top')
            else:
                ax.axis('off')
    if center or unitless:
        #prepexplain = r'Processed data:If $x_i^{(j)}$ is sample $i$ of stat $j$,'+'\n'+r'plot shows  (x_i^{(j)} - \bar{x}^{(j)})/\sigma_{(j)}$'
        if center and (not unitless):
            prepexplain = r'Processed data:If $x_i^{(j)}$ is sample $i$ of stat $j$,'+'\n'+r'plot shows  $(x_i^{(j)} - \bar{x}^{(j)})$'
            prepstr = '-0mean'
        elif (not center) and unitless:
            prepexplain = r'Processed data:If $x_i^{(j)}$ is sample $i$ of stat $j$,'+'\n'+r'plot shows  $x_i^{(j)}/\sigma_{(j)}$'
            prepstr = '-nounits'
        else:
            prepexplain = r'Processed data:If $x_i^{(j)}$ is sample $i$ of stat $j$,'+'\n'+r'plot shows  $(x_i^{(j)} - \bar{x}^{(j)})\sigma_{(j)}$'
            prepstr = '-0mean-nounits'

        ax.annotate(prepexplain,xy=(1.,.75),xycoords='figure fraction',horizontalalignment='right',verticalalignment = 'top',fontsize=16)
    else:
        prepstr = ''
                    
    if savefig:
        if outtag:
            tagstr = '_'+outtag
        else:
            tagstr = ''
        outname ='triangleplot{0:s}'.format(prepstr + tagstr)
        outf = outdir+outname+'.pdf'
        print "saving",outf
        plt.savefig(outf)
    else:
        plt.show()
        
############################################################
# ANOMALY COV AND PCA analysis
############################################################

#----------------------------------------
def get_stat_covmat(statslist,statdat=None,\
                    outdir='lcdm_anomalycov_plots/',tag='sims',\
                    overwrite = False,center=True,unitless=False,normalize=False):
    """
    Computes or reads in covmat for anomaly stats. Saves data to file
    in which the first row contains the man for each stat, the second
    row contains the standard deviation, and rows after that contain the
    covmat. The header will contain first a commented line listing the stats
    (column labels), while the second commented line will say whether
    the covmat is based on preprocessed data. 

    statslist - string list of stat labels
    statdat - list of  dim  Nrealx2 arrays, containing realization in 1st column, stat in 2nd
       if file exists can keep this empty and just read in a covmat
    outdir - where to save the covmat?
    tag - how to label covmat outpt filename?
             will be: outdir/statcov_tag.dat

    overwrite - if true, compute whether or not file exists
    center - if true, for each stat x, finds covmat of (x-xbar)
    unitless - if true, for each stat x, divides by its standard deviation
              to make everything unitless
    """
    if center or unitless:
        prepstr = '-0mean'*center+'-nounits'*unitless
    else:
        prepstr = '-rawdat'
    if normalize:
        normstr = '-normed'
    else:
        normstr = '-unnorm'
    Nstat = len(statslist)
    outf = ''.join([outdir,'covmat',prepstr,normstr,'_',tag,'.dat'])
    compute = True
    print "Getting covmat for ",outf
    if not overwrite and os.path.isfile(outf):
        #read in outf, check that stats match,
        # if so, just use that covmat; otherwise recompute
        dat = np.loadtxt(outf) #expect Nstat x Nstat matrix 
        if dat.shape[1]==Nstat: #if right shape, check stats in header
            f = open(outf)
            firstline =  f.readline()[1:] #cut out pound comment sign 
            f.close()
            fstatslist = firstline.split()
            if fstatslist == statslist:
                print "  Reading in data."
                compute = False
                meanvec = dat[0,:]
                stdvec = dat[1,:]
                covmat = dat[2:,:]
            else:
                print "  Stats don't match, need to compute."
                compute = True
    if compute:        
        if (statdat is None):
            raise ValueError("No data passed, can't compute covmat")
        else:
            print "  Computing covmat."
            headerstr = ' '.join(statslist)+'\n#centered='+str(center)+', unitless='+str(unitless)+', normalized='+str(normalize)
            outdat = np.zeros((Nstat+2,Nstat)) #to hold output data (mean, std, covmat)
            Nreal = statdat[0].shape[0]
            X = np.zeros((Nstat,Nreal)) #to use for covmat computation
            for i in xrange(Nstat):
                if statslist[i]=='S12':
                    X[i,:] = np.log(statdat[i][:,1])
                else:
                    X[i,:] = statdat[i][:,1]
            meanvec = np.mean(X,axis=1) #should be dim Nstat
            stdvec = np.std(X,axis=1) #should be dim Nstat
            outdat[0,:] = meanvec
            outdat[1,:] = stdvec
            if center:
                X = np.array([(X[:,n] - meanvec) for n in range(Nreal)]).T
            if unitless:
                X = np.array([X[:,n]/stdvec for n in range(Nreal)]).T
            #covmat = np.dot(X,X.T) #Nstat x Nstat
            if normalize:
                covmat = np.corrcoef(X)
            else:
                covmat = np.cov(X)
            outdat[2:,:] = covmat
            np.savetxt(outf,outdat,header=headerstr)
    return covmat,meanvec,stdvec


def plot_covmat(covmat,statlist,titlenote='',outdir= 'lcdm_anomalycov_plots/',filetag='',savefig=True,covmat2=None):
    """
    Make heatmat of covariance matrix, given list of labels to label rows.

    If second covmat is passed, plot covmat vals in lower triangle, covmat2 in upper triangle. 
    Don't plot diagonals. 

    If string 'blank' is passed for covmat2, just plot lower triangle of covmat
    """
    import seaborn as sns
    statlabels = [texdict[s] for s in statlist]
    plt.figure(figsize=(6,6))

    ax=plt.subplot(111)

    if covmat2 is None:    
        sns.heatmap(covmat, ax=ax,xticklabels=statlabels,\
                     yticklabels=statlabels,annot=True,square=True,cbar=False)
        plt.title(titlenote,fontsize=20)
    elif covmat2=='blank':
        #only plot lower triangle
        Nstat = covmat.shape[0]
        plotdat = np.tril(covmat)[1:,:-1]
        plotmin = np.min(plotdat)
        plotmax = np.max(plotdat)
        mask = np.triu(np.ones((Nstat-1,Nstat-1)),1)
        #print plotmin,plotmax
        sns.heatmap(plotdat,vmin=plotmin,vmax=plotmax, ax=ax,xticklabels=statlabels,mask=mask,\
                     yticklabels=statlabels,annot=True,square=True,cbar=False)
        plt.annotate(titlenote,xy=(.7,.8),horizontalalignment='center',verticalalignment='center',xycoords='axes fraction',fontsize=24)
    else:
        #plot covmat in lower
        Nstat = covmat.shape[0]
        mask = np.diag(np.ones(Nstat))
        plotdat = np.tril(covmat)+np.triu(covmat2)
        plotmin = np.min(plotdat)
        plotmax = 1.
        sns.heatmap(plotdat,vmin=plotmin,vmax=plotmax, ax=ax,xticklabels=statlabels,mask=mask,\
                     yticklabels=statlabels,annot=True,square=True,cbar=False)
        ax.plot([0,Nstat],[Nstat,0],color='k',linewidth=3)
        plt.title(titlenote,fontsize=20)
    plt.yticks(fontsize=16, rotation='horizontal')
    plt.xticks(fontsize=16)


    if savefig:
        if filetag:
            tagstr = '_'+filetag
        else:
            tagstr =''
        outname = 'covmat'+tagstr+'.pdf'
        outf = outdir+outname
        print "saving",outf
        plt.savefig(outf)
    else:
        plt.show()
    
    sns.reset_orig()

#----------------------------------------

def get_pca_fromcov(covmat):
    """
    Given Nstat x Nstat covariance matrix, find eigenvectors and eigenvalues for PCA
    """
    eiglambda, eigvec= np.linalg.eig(covmat)
    #eiglambda will be a d-dim array
    #eigvec will be a dxd array where eigverc[:,i] will be the eigen vector corresponding to eiglambda[i]

    #order eigvals from largest to smallest
    sortinds = np.argsort(eiglambda)[::-1]
    eiglambda = eiglambda[sortinds]
    eigvec = eigvec[:,sortinds]
    
    return eiglambda, eigvec

def plot_eigvec_heatmap(eigvec,statlist,titlenote='',outdir= 'lcdm_anomalycov_plots/',filetag='',savefig=True):
    """
    Given Nstat x Nstat array of format that would be returned by np.linalg.eig,
    make heatmap showing components of Covariance matrix eigenvectors.
    """
    import seaborn as sns
    Nstat = eigvec.shape[0]
    statlabels = [texdict[s] for s in statlist]
    pcaind = np.arange(1,Nstat+1)


    plt.figure(figsize=(6,6))
    plt.title(titlenote,fontsize=20)
    ax=plt.subplot(111)
    sns.heatmap(eigvec , ax=ax,xticklabels=pcaind,fmt='0.3f',\
                yticklabels=statlabels,annot=True,square=True,cbar=False,linecolor='lightgray',linewidths=.5)
    for i in xrange(1,Nstat):
        ax.axvline(i,color='k')
    
    plt.yticks(fontsize=16, rotation='horizontal')
    plt.xticks(fontsize=16)
    ax.set_xlabel("PCA index",fontsize=20)
    
    if savefig:
        if filetag:
            tagstr = '_'+filetag
        else:
            tagstr =''
        outname = 'eigvecs'+tagstr+'.pdf'
        outf = outdir+outname
        print "saving",outf
        plt.savefig(outf)
    else:
        plt.show()
    
    sns.reset_orig()
#----------------------------------------
# just exploratory, can we project out pcs from scatter data
# so we can look at what the triangle plot looks like after removing
# the first to PCs?
def projectout_vecs(statslist, statdat,veclist=[]):
    """
    given statdata, and veclist, projects out veclist, 
    returns another statdat object with components of data
    along those directions removed.

    Assumes each vector in veclist is a unit vector, 
    and that the components are for the (x-mean)/sigma preprocessed data. 

    """
    Nstat = len(statslist)
    Nreal = statdat[0].shape[0]
    #get data into array we can work with more easily
    X = np.zeros((Nstat,Nreal)) #to use for covmat computation
    for i in xrange(Nstat):
        if statslist[i]=='S12':
            X[i,:] = np.log(statdat[i][:,1])
        else:
            X[i,:] = statdat[i][:,1]
    meanvec = np.mean(X,axis=1) #should be dim Nstat
    stdvec = np.std(X,axis=1) #should be dim Nstat

    #preprocess data to match what was done to create the eigenvectors for pca
    X = np.array([(X[:,n] - meanvec)/stdvec for n in range(Nreal)]).T #Nstat x Nreal
    #X is Nstat x Nreal
    U = np.array(veclist) #Nvec x Nstat
    A = np.matmul(U,X) #Nvec x Nreal
    #each element A[i,j] is the coefficient for real i, vec j
    X = X - np.matmul(U.T,A) #x with the vectors projected out

    #put units back in and undo centering
    X = np.array([X[:,n]*stdvec + meanvec for n in range(Nreal)]).T #Nstat x Nreal
    
    #put back in statdat format
    outdat = []
    for i in xrange(Nstat):
        outdati = np.zeros((Nreal,2))
        outdati[:,0] = np.arange(Nreal)
        if statslist[i]=='S12':
            outdati[:,1] = np.exp(X[i,:])
        else:
            outdati[:,1] = X[i,:]
        outdat.append(outdati)
    return outdat

def get_pca_coords(statslist, statdat=None, othermaps = [],pcaarray=None,statdir='/Users/Shared/lcdm_NSIDE64_maps_for_cmbparity/lcdm_pca_stats/',tag='sims',overwrite=False):
    """
    Assumes PCA array is Nstat x Nvec array where columns are the PCA vectors
    (in format returned by np.linalg.eig)

    othermaps is a list of Nstat-dim arrays of stats measured from e.g. smica or theory;
    will be put in same coords as statdat
    """
    outf = statdir+'pcacoords_'+tag+'.dat'
    Nstat = len(statslist)
    if overwrite or not os.path.isfile(outf):
        if statdat is None or pcaarray is None:
            raise ValueError("Need statdat and PCA vectors to be able to get PCA coords.")

        Nreal = statdat[0].shape[0]
        Nvec = pcaarray.shape[1]
        if len(statdat)!= Nstat:
            raise ValueError("Number of stat names doesn't match number of stats in statdat")
        #get data into array we can work with more easily
        X = np.zeros((Nstat,Nreal)) #to use for covmat computation
        for i in xrange(Nstat):
            if statslist[i]=='S12':
                X[i,:] = np.log(statdat[i][:,1])
            else:
                X[i,:] = statdat[i][:,1]
        meanvec = np.mean(X,axis=1) #should be dim Nstat
        stdvec = np.std(X,axis=1) #should be dim Nstat

        #preprocess data to match what was done to create the eigenvectors for pca
        X = np.array([(X[:,n] - meanvec)/stdvec for n in range(Nreal)]).T #Nstat x Nreal
        normothers = [(om-meanvec)/stdvec for om in othermaps]
        outothers = [np.dot(pcaarray.T,no) for no in normothers]
        #X is Nstat x Nreal, pcaarray is Nstat x Nvec
        A = np.matmul(X.T,pcaarray) #Nreal x Nvec; A[i,j] = pca coord j for realization i
        np.savetxt(outf, A)
    else:
        if statdat is not None:
            Nreal = statdat[0].shape[0]
            X = np.zeros((Nstat,Nreal)) #to use for covmat computation
            normothers = [om[:] for om in othermaps]
            for i in xrange(Nstat):
                if statslist[i]=='S12':
                    X[i,:] = np.log(statdat[i][:,1])
                    for no in normothers:
                        no[i] = np.log(no[i])
                else:
                    X[i,:] = statdat[i][:,1]
            meanvec = np.mean(X,axis=1) #should be dim Nstat
            stdvec = np.std(X,axis=1) #should be dim Nstat
            normothers = [(no-meanvec)/stdvec for no in normothers]
            outothers = [np.dot(pcaarray.T,no) for no in normothers]
        else:

            outothers = [np.nan for s in statslist]
            print "Can't project other vals without statdat"
        A = np.loadtxt(outf)
        Nvec = A.shape[1]
    #put back into same statslist format
    outdat = []
    for j in xrange(Nvec):
        Nreal = A.shape[0]
        outj = np.zeros((Nreal,2))
        outj[:,0] = np.arange(Nreal)
        outj[:,1] = A[:,j]
        outdat.append(outj)
    
    return outdat,outothers
    
#----------------------------------------
# functions for doing sample and bootstrap variance for covmat entries

def subsample_stats_forcovmat(statlist,statdat=None,Nsub=100,\
                    outdir='output/lcdmcov_N1000_subsample/',tag='lcdm',\
                              overwrite = False,center=True,unitless=False,normalize=True,Nrealin = 100000):
    """
    Divide statdat into 100 equal-sized subsamples, 
      for each compute covmat and save it in outdir.
      also compute & return sample variance for covmat

    center means  subtract means
    unitless means divide out standard dev before getting cov
    norm means get correlation coefs before getting cov
    unitless and norm will give same results. 

    if statdat is given, nrealin is ignored. 
    """
    if statdat is None:
        Ntot = Nrealin
    else:
        Ntot = statdat[0].shape[0]

    Nrealpersub = Ntot/Nsub #if division isn't even, will have some leftovers

    starts = np.array([n*Nrealpersub for n in xrange(Nsub)])
    ends = np.array([(n+1)*Nrealpersub-1 for n in xrange(Nsub)])
    tags = ['{0:s}_n{1:02d}_r{2:05d}-{3:05d}'.format(tag,n,starts[n],ends[n])\
            for n in xrange(Nsub)]
    covgrid = []
    for n in xrange(Nsub):
        print "subsampling covs; on sample ",n, 'r=',starts[n],'-',ends[n]
        if statdat is None:
            statdatn = None
        else:
            statdatn = [sd[starts[n]:ends[n],:] for sd in statdat]
        cov,a,b = get_stat_covmat(statlist,statdatn,tag = tags[n],\
                                  outdir=outdir,\
                                overwrite = overwrite, center = center,\
                                unitless = unitless, normalize = normalize)
        covgrid.append(cov)
    covgrid = np.array(covgrid)
    varcov = np.var(covgrid,axis=0,ddof=1)
    # ddof calcs variance with factor of 1/(N-1) to get unbiased est
    return varcov


##################################################################
def run_manystats(realmin=0,realmax=10,maskfile='',maskname='', \
                  datadir = 'output/lcdm-map-testspace/',\
                  statdir = '', lvmapdir = '',\
                  mapbase = 'map',redoCl = True, redoCt = True, \
                  statlist =['S12','R27','R50','Cl2','Cl3','Ct180','s16'],Ncore = 0,redoLVmaps = True, mapnside = NSIDEfid):
    #, parityNside=NSIDEfid):
    """
    THIS IS THE FUNCTION TO USE TO EXTRACT STATS FROM MAPS.

    given min and max realization number, splits realizations into chunks
    and runs get_stats_formaplist on them in parallel. If Ncore passed,
    uses that many cores. If Ncore == 0 just uses however many are available.

    """
    if not statdir:
        statdir = datadir
    
    doS12 = False
    doR10 = False
    doR27 = False
    doR50 = False
    doRall = False
    doCl2 = False
    doCl3 = False
    doCt180 = False
    doalign23S = False
    doALV = False 
    dos16 = False
    for stat in statlist:
        if stat=='S12':
            doS12 = True
        elif stat == 'R27':
            doR27 = True
        elif stat == 'R10':
            doR10 = True
        elif stat == 'R50':
            doR50 = True
        elif stat == 'Rall':
            doRall = True
        elif stat == 'Cl2':
            doCl2 = True
        elif stat == 'Cl3':
            doCl3 = True
        elif stat == 'Ct180':
            doCt180 = True
        elif stat == 'align23S':
            doalign23S = True
        elif stat == 'ALV':
            doALV = True
        elif stat == 's16':
            dos16 = True
        else:
            print "STAT {0:s} NOT RECOGNIZED.".format(stat)


    if len(statlist)>1 or statlist[0]!='ALV':
        #split up realization numbers into chuncks
        availcore = multiprocessing.cpu_count()
        if not Ncore or (availcore<Ncore):
            Ncore = availcore
        if maskname:
            maskstr = maskname
        else:
            maskstr = 'fullsky'
        
        print "Using {0} cores to get stats for {1}.".format(Ncore,maskstr) 
        edges = np.linspace(realmin,realmax,num=Ncore+1,dtype=int)
        print "Splitting realizations into chunks with edges:\n",edges 
        rmins = edges[:-1]
        rmaxs = edges[1:]-1
        rmaxs[-1]+=1

        #start processes for each chunk of realizations
        jobs = []
        for i in xrange(Ncore):
            p = multiprocessing.Process(target = get_stats_formaplist, args=(rmins[i],rmaxs[i],maskfile,maskname,datadir,statdir,mapbase,redoCl,redoCt,doS12,doR10,doR27,doR50,doRall,doCl2,doCl3,doCt180,doalign23S,dos16, mapnside))
            jobs.append(p)
            print "Starting rlzns {0:d}-{1:d}".format(rmins[i],rmaxs[i])
            p.start()
        #wait until all are done before moving on
        for j in jobs:
            j.join()
               
    if doALV:
        #alv needs to run through maps twice, so needs to be handled separately
        get_manyALVstats(realmin, realmax, mapbase, maskfile, maskname, datadir, lvmapdir, statdir, Ncore = Ncore,redoLVmaps = redoLVmaps)

def get_planckstats_lists(statlist = ['S12','R27','R50','Cl2','Cl3','Ct180','s16','ALV','align23S'],\
                    masknamelist=['','UT76','UT78'],\
                    maskflist = ['','data/masks/COM_Mask_CMB-IQU-UT76_0064_R2.01.fits',\
                                 'data/masks/COM_Mask_CMB-IQU-UT78_0064_R2.01.fits'],\
                    lvlabellist = ['lcdm','ffp'],\
                    lvmeanmaplist = ['/Users/Shared/lcdm_NSIDE64_maps_for_cmbparity/lcdm_[MASKNAME]_LVmaps/map[MASKNAME2]_0.LVd08n16_MEANr0-99999.fits','/Users/Shared/planck_ffp/smicaffp_[MASKNAME]_LVmaps/smica-ffp8.1[MASKNAME2]_0.LVd08n16_MEANr0-999.fits'],\
                    inmapdir ='output/planckstats/',inmapbase = 'smica_0064_full',\
                          outdir = 'output/planckstats/',outtag='',overwrite=True, mapnside = NSIDEfid):
    """
    Loop through masks and LV sim ensembles and get a measurement of Planck stats for each
    saving each into a file.

    In any filenames, put [MASKNAME] as placeholder for maskname to be filled in.
    Output will be a list of lists, indexed by [mask choice][LV mean map choice]
    """
    #working here, need to set up mapnside scaling
    Nmask = len(masknamelist)
    NLVmean = len(lvmeanmaplist)
    outvals = []
    for i in xrange(Nmask):
        maskf = maskflist[i]
        maskname = masknamelist[i]
        outvalsi = []
        for j in xrange(NLVmean):
            outvalsij = get_planckstats(statlist,maskname, maskf,lvlabellist[j], lvmeanmaplist[j],\
                            inmapdir ,inmapbase, outdir = outdir, \
                                        outtag=outtag, overwrite=overwrite, mapnside = mapnside)

            outvalsi.append(outvalsij)
        outvals.append(outvalsi)
    return outvals

def get_planckstats(statlist = ['S12','R27','R50','Cl2','Cl3','Ct180','s16','ALV','align23S'],\
                    maskname='', maskf = '',lvlabel='lcdm',
                    lvmeanmap = '/Users/Shared/lcdm_NSIDE64_maps_for_cmbparity/lcdm_fullsky_LVmaps/map_0.LVd08n16_MEANr0-99999.fits',
                    inmapdir ='output/planckstats/',inmapbase = 'smica_0064_full',\
                    outdir = 'output/planckstats/',outtag='',overwrite=True, mapnside = NSIDEfid):
    if outtag:
        outtagstr = outtag+'_'
    else:
        outtagstr = ''
    if maskname:
        maskstr  = maskname
    else:
        maskstr = 'fullsky'
    outf = ''.join([outdir,outtagstr,inmapbase,'.',maskstr,'.',lvlabel,'.stats.dat'])
    if ((not overwrite) and (os.path.isfile(outf))):
        f = open(outf)
        header = f.readline()
        f.close()
        statsinfile = (header.replace('#','')).split()
        #print statsinfile
        if statsinfile == statlist:
            print "Reading in existing data from file",outf
            outvals= np.loadtxt(outf,ndmin=1)
            
            return outvals
        else:
            print "Recomputing; stats in outf don't match for", outf
    if maskname:
        lvmeanfile = lvmeanmap.replace('[MASKNAME]',maskname).replace('[MASKNAME2]','-'+maskname)
    else:
        lvmeanfile = lvmeanmap.replace('[MASKNAME]','fullsky').replace('[MASKNAME2]','') 
    lvvarfile = lvmeanfile.replace('MEAN','VAR')
    outvals = get_onemap_stats(inmapdir,inmapbase,maskf, maskname, statlist = statlist,  ALVmeanmapf = lvmeanfile, ALVvarmapf = lvvarfile,overwrite = overwrite, mapnside = mapnside)
    print "Saving stats to",outf
    np.savetxt(outf,outvals,header=' '.join(statlist))
    return outvals


def get_pvals(vals,statlist = ['S12','R27','R50','Cl2','Cl3','Ct180','s16','ALV','align23S'],\
              statcompdir='/Users/Shared/lcdm_NSIDE64_maps_for_cmbparity/lcdm_fullsky_stats/',maskname='',mapbase='map'):
    """
    Given stats for one map, and info about which ensemble to compare to, 
    compute pvalues
    """
    pvals = []
    if maskname:
        maskstr = maskname
    else:
        maskstr = 'fullsky'
    statcompdir = statcompdir.replace('[MASKNAME]',maskstr)
    #print statcompdir
    for i,s in enumerate(statlist):
        if (s in ['align23S']) and maskname:
            pvals.append(np.nan)
        else:
            dat = collect_data_forstat(s,statcompdir,mapbase,maskname)
            print dat[:,1].size,1./dat[:,1].size,
            fracless = float(np.sum(dat[:,1]<vals[i]))/dat[:,1].size
            fracmore = float(np.sum(dat[:,1]>vals[i]))/dat[:,1].size
            p = min(fracmore,fracless)
            pvals.append(p)
    return pvals

def get_pvals_givenarrays(vals,statdat):
    """Same as get_pvals but for when you already have the statdat arrays read in"""
    pvals = []
    for i,v in enumerate(vals):
        if np.isnan(v):
            pvals.append(np.nan)
        else:
            dat = statdat[i]
            if type(dat) not in (list,np.ndarray):
                pvals.append(np.nan)
            else:
                fracless = float(np.sum(dat[:,1]<=vals[i]))/dat[:,1].size
                fracmore = float(np.sum(dat[:,1]>=vals[i]))/dat[:,1].size
                p = min(fracmore,fracless)
                pvals.append(p)
                #print p,np.sum(dat[:,1]<=vals[i]),float(np.sum(dat[:,1]>=vals[i])),dat[:,1].size
    return np.array(pvals)

def get_pvals_forRall(Ralldat,Rallsims):
    """
    Ralldat should be a Nell dim array, Rallsims shoudl be Nreal x Nell,
    returns lower tail probability for each ell
    """
    Nreal,Nell = Rallsims.shape
    if Nell!=Ralldat.size:
        print "Ralldat size is",Ralldat.size,' but sims have Nell=',Nell
    pvals = np.zeros(Nell)
    for i in xrange(Nell):
        testval = Ralldat[i]
        simvals = Rallsims[:,i]
        fracless = float(np.sum(simvals<testval))/float(Nreal)
        pvals[i] = fracless
    #tiledat = np.tile(Ralldat,(Nreal,1))
    #fracless = (np.sum(Rallsims<tiledat,axis=0).astype(float))/Nreal
    #fracmore = (np.sum(Rallsims>tiledat,axis=0).astype(float))/Nreal
    #pvals = np.minimum(fracless,fracmore)
    #print pvals.shape
    return pvals

def sigma_to_prob(sigvals=[1,2,3]):
    """
    Returns list of probabilities associated with the sigma values given in
    sigvals. 
    """
    return np.array([1. - 2*stats.norm.sf(s) for s in sigvals])

def get_Rall_contours(Rallsims=np.array([]),sigs = np.arange(1,4),samplespacing = .001,LMIN=2,LMAX=60,overwrite=False,\
                      datadir = 'output/stat-testspace/',mapbase='map',maskname='',kdebandwidth=None):
    """
    Will either read in Rall contours, or compute and save them; warning, doesn't check for matching sigs.

    If it;s able to read in previously computed data, don't need Rallsims, otherwise needs that passed 
    as an Nrealization x Nell array

    #for each lmax, does a KDE to get probability distribution, then samples points with spacing samplespacing
    # to figure out where to put the 1, 2, 3, etc sigma contours.
    """
    fname = get_filename_testcase(datadir = datadir, mapbase = mapbase, number=0,stattype='Rall-contours')
    ell = np.arange(LMIN,LMAX+1)
    probs = sigma_to_prob(sigs)
    Nprob = probs.size
    if (not overwrite) and os.path.isfile(fname):
        print "reading in data from",fname
        indat = np.loadtxt(fname)
        Nell = ell.size
        if indat.shape != (ell.size,2*sigs.size+3):
            print "WARNING!, indat shape is",indat.shape,' but expented Nell=',Nell,' Nsig=',sigs.size
        inell = indat[:,0]
        if not np.all(inell==ell):
            print "WARNING: ells are mismatched"
        peaks = indat[:,1]
        means = indat[:,2]
        outdat = np.zeros((Nprob,2,Nell))
        for p in xrange(Nprob):
            outdat[p,0,:] = indat[:,3+2*p]
            outdat[p,1,:] = indat[:,3+2*p+1]        
    else:
        if not Rallsims.size:
            print "WARNING, to compute contours, need Rallsims"
        Nreal,Nell = Rallsims.shape
        if ell.size != Nell:
            print "WARNING, check ell ranges. LMIN,MAX=",LMIN,LMAX,' but Nell for Rall is ', Nell
        outdat = np.zeros((Nprob,2,Nell)) #[prob level][lower-upper][ellmax]
        peaks =  np.zeros(Nell)
        means = np.zeros(Nell)
        samplespacing = kdebandwidth
        samplemin = 0.+samplespacing
        samplemax = 30
        xvals = np.arange(samplemin,samplemax,samplespacing)
        means = np.mean(Rallsims,axis = 0)
        for i in xrange(Nell):
            dat = Rallsims[:,i]
            if np.any(np.isnan(dat)):
                peaks[i] = np.nan
                means[i] = np.nan
                outdat[:,:,i] = np.nan
                continue
            kde = stats.gaussian_kde(Rallsims[:,i],bw_method=kdebandwidth)
            L = kde.pdf(xvals) #prob of being at each value
            i_sort = np.argsort(L)[::-1] #indices to sort L in decreasing order
            peaks[i] = xvals[i_sort[0]]
            i_unsort = np.argsort(i_sort)

            Lcumsum = L[i_sort].cumsum()
            Lcumsum/=Lcumsum[-1] #each entry has fraction of scatter points in
            #  that bin or in bins with higher density
            Lcumsum_forx = Lcumsum[i_unsort]
            
            for p in xrange(Nprob):
                inds = np.where(Lcumsum_forx<probs[p])[0]
                outdat[p,0,i] = xvals[inds[0]]
                outdat[p,1,i] = xvals[inds[-1]]

        #save output
        saveheader = '#lmax, peakR, minR for sig1,  maxR for sig1, minR for sig2, maxR for sig2, ...\n sigs='+str(sigs)
        savedat = np.zeros((Nell,3+2*Nprob))
        savedat[:,0] = ell
        savedat[:,1] = peaks
        savedat[:,2] = means
        for p in xrange(Nprob):
            savedat[:,3+2*p] = outdat[p,0,:]
            savedat[:,3+2*p+1] = outdat[p,1,:]
        print 'saving data to ',fname
        np.savetxt(fname,savedat,header=saveheader)    
    return peaks,means,outdat


def get_Rall_contours_hist(Rallsims=np.array([]),sigs = np.arange(1,4),samplespacing = .001,LMIN=2,LMAX=60,overwrite=False,\
                           datadir = 'output/stat-testspace/',mapbase='map',maskname='',nbins=100):
    """

    Don't trust this function!.

    Will either read in Rall contours, or compute and save them; warning, doesn't check for matching sigs.

    If it;s able to read in previously computed data, don't need Rallsims, otherwise needs that passed 
    as an Nrealization x Nell array

    #for each lmax, does a KDE to get probability distribution, then samples points with spacing samplespacing
    # to figure out where to put the 1, 2, 3, etc sigma contours.
    """
    fname = get_filename_testcase(datadir = datadir, mapbase = mapbase, number=0,stattype='Rall-contours-hist')
    ell = np.arange(LMIN,LMAX+1)
    probs = sigma_to_prob(sigs)
    Nprob = probs.size
    if (not overwrite) and os.path.isfile(fname):
        print "reading in data from",fname
        indat = np.loadtxt(fname)
        Nell = ell.size
        if indat.shape != (ell.size,2*sigs.size+3):
            print "WARNING!, indat shape is",indat.shape,' but expented Nell=',Nell,' Nsig=',sigs.size
        inell = indat[:,0]
        if not np.all(inell==ell):
            print "WARNING: ells are mismatched"
        peaks = indat[:,1]
        means = indat[:,2]
        outdat = np.zeros((Nprob,2,Nell))
        for p in xrange(Nprob):
            outdat[p,0,:] = indat[:,3+2*p]
            outdat[p,1,:] = indat[:,3+2*p+1]        
    else:
        if not Rallsims.size:
            print "WARNING, to compute contours, need Rallsims"
        Nreal,Nell = Rallsims.shape
        if ell.size != Nell:
            print "WARNING, check ell ranges. LMIN,MAX=",LMIN,LMAX,' but Nell for Rall is ', Nell
        outdat = np.zeros((Nprob,2,Nell)) #[prob level][lower-upper][ellmax]
        peaks =  np.zeros(Nell)
        means = np.zeros(Nell)
        #samplemin = 0.
        #samplemax = 20
        #xvals = np.linspace(samplemin,samplemax,nbins+1)
        means = np.mean(Rallsims,axis = 0)
        for i in xrange(Nell):
            dat = Rallsims[:,i]
            if np.any(np.isnan(dat)):
                peaks[i] = np.nan
                means[i] = np.nan
                outdat[:,:,i] = np.nan
                continue
            #kde = stats.gaussian_kde(Rallsims[:,i],bw_method=kdebandwidth)
            #L,xbins = np.histogram(dat,range=(samplemin,samplemax),bins=nbins)
            L,xbins = np.histogram(dat,bins=nbins)
            #print L
            xbins = 0.5 * (xbins[1:] + xbins[:-1])
            i_sort = np.argsort(L)[::-1] #indices to sort L in decreasing order
            #print L[i_sort]
            #print (L[i_sort].cumsum()).astype(float)/np.sum(L)
            peaks[i] = xbins[i_sort[0]]
            i_unsort = np.argsort(i_sort)

            Lcumsum = (L[i_sort].cumsum()).astype(float)
            Lcumsum/=Lcumsum[-1] #each entry has fraction of scatter points in
            #  that bin or in bins with higher density
            Lcumsum_forx = Lcumsum[i_unsort]
            
            for p in xrange(Nprob):
                inds = np.where(Lcumsum_forx<probs[p])[0]
                #print p,probs[p],inds[0],inds[-1]
                outdat[p,0,i] = xbins[inds[0]]
                outdat[p,1,i] = xbins[inds[-1]]

        #save output
        saveheader = '#lmax, peakR, minR for sig1,  maxR for sig1, minR for sig2, maxR for sig2, ...\n sigs='+str(sigs)
        savedat = np.zeros((Nell,3+2*Nprob))
        savedat[:,0] = ell
        savedat[:,1] = peaks
        savedat[:,2] = means
        for p in xrange(Nprob):
            savedat[:,3+2*p] = outdat[p,0,:]
            savedat[:,3+2*p+1] = outdat[p,1,:]
        print 'saving data to ',fname
        np.savetxt(fname,savedat,header=saveheader)    
    return peaks,means,outdat


def get_Rall_contours_singletail(Rallsims=np.array([]),sigs = np.arange(1,4),LMIN=2,LMAX=60,overwrite=False,\
                      datadir = 'output/stat-testspace/',mapbase='map',maskname=''):
    """
    Will either read in Rall contours, or compute and save them; warning, doesn't check for matching sigs.

    If it;s able to read in previously computed data, don't need Rallsims, otherwise needs that passed 
    as an Nrealization x Nell array

    #for each lmax, find value of R for which the single tail prob corresponds to 
    # being 1,2,3 sigma ulikely
    # In this function cumulative stats are used to place contours.
    #    as in see e.g. "mean statsitics" describe herehttps://samreay.github.io/ChainConsumer/usage.html  
    """
    fname = get_filename_testcase(datadir = datadir, mapbase = mapbase, number=0,stattype='Rall-contours-singletail')
    ell = np.arange(LMIN,LMAX+1)
    probs = sigma_to_prob(sigs)
    Nprob = probs.size
    if (not overwrite) and os.path.isfile(fname):
        print "reading in data from",fname
        indat = np.loadtxt(fname)
        Nell = ell.size
        if indat.shape != (ell.size,2*sigs.size+3):
            print "WARNING!, indat shape is",indat.shape,' but expented Nell=',Nell,' Nsig=',sigs.size
        inell = indat[:,0]
        if not np.all(inell==ell):
            print "WARNING: ells are mismatched"
        medians = indat[:,1]
        means = indat[:,2]
        outdat = np.zeros((Nprob,2,Nell))
        for p in xrange(Nprob):
            outdat[p,0,:] = indat[:,3+2*p]
            outdat[p,1,:] = indat[:,3+2*p+1]        
    else:
        if not Rallsims.size:
            print "WARNING, to compute contours, need Rallsims"
        Nreal,Nell = Rallsims.shape
        
        if ell.size != Nell:
            print "WARNING, check ell ranges. LMIN,MAX=",LMIN,LMAX,' but Nell for Rall is ', Nell
        outdat = np.zeros((Nprob,2,Nell)) #[prob level][lower-upper][ellmax]
        medians =  np.zeros(Nell)
        means = np.mean(Rallsims,axis = 0)
        for i in xrange(Nell):
            dat = Rallsims[:,i]
            if np.any(np.isnan(dat)):
                medians[i] = np.nan
                means[i] = np.nan
                outdat[:,:,i] = np.nan
                continue
            i_sort = np.argsort(dat)#[] #indices to sort R in increasing order
            #print i_sort
            medians[i] = dat[i_sort[Nreal/2]]
            for p in xrange(Nprob):
                halfp = .5*(1.-probs[p])
                lowerpercentile = 100*halfp
                upperpercentile = (1.-halfp)*100
                outdat[p,0,i] = np.percentile(dat,lowerpercentile)
                outdat[p,1,i] = np.percentile(dat,upperpercentile)
                
                #nforprob = int(Nreal*.5*(1-probs[p])) #number for single tail prob (round down)
                #print p,probs[p],nforprob
                #lowerind = i_sort[nforprob] #nforprob are less than # at this ind
                #upperind = i_sort[-(nforprob+1)] #nforprop are greater than # at this ind
                #print lowerind,upperind
                #outdat[p,0,i] = dat[lowerind]
                #outdat[p,1,i] = dat[upperind]

        #save output
        saveheader = '#lmax, medianR, meanR, minR for sig1,  maxR for sig1, minR for sig2, maxR for sig2, ...\n sigs='+str(sigs)
        savedat = np.zeros((Nell,3+2*Nprob))
        savedat[:,0] = ell
        savedat[:,1] = medians
        savedat[:,2] = means
        for p in xrange(Nprob):
            savedat[:,3+2*p] = outdat[p,0,:]
            savedat[:,3+2*p+1] = outdat[p,1,:]
        print 'saving data to ',fname
        np.savetxt(fname,savedat,header=saveheader)    
    return medians,means,outdat

#########################################################################
#########################################################################

def main():
    #various bool switches are used to run different parts of the data generation
    if 0: #downgrade plancm maps & masks
        mapdir = 'data/maps/'
        origNsidestr = '1024'
        outNside = 32#16#64
        outNsidestr = '0032'#'0016'#'0064'
        origflist = [\
                     'COM_CMB_IQU-commander_1024_R2.02_full.fits',\
                     'COM_CMB_IQU-nilc_1024_R2.02_full.fits',\
                     'COM_CMB_IQU-sevem_1024_R2.02_full.fits',\
                     'COM_CMB_IQU-smica_1024_R2.02_full.fits',\
        ]
        for f in origflist:
            outf = f.replace(origNsidestr,outNsidestr)
            inm = hp.read_map(mapdir+f)
            outm = 1.e6*downgrade_map(inm,outNside) #K to uK
            print "saving downgraded map to",mapdir+outf
            hp.write_map(mapdir+outf,outm)
            
        maskdir = 'data/masks/'
        origNsidestr = '2048'
        origcommonmaskf = 'COM_Mask_CMB-IQU-common-field-MaskInt_2048_R2.01.fits'
        orig76m = hp.read_map(maskdir+origcommonmaskf,field=1)
        out76str = 'COM_Mask_CMB-IQU-UT76_'+outNsidestr+'_R2.01.fits'
        out76m = downgrade_mask(orig76m,outNside)
        print "saving downgraded mask to",maskdir+out76str
        hp.write_map(maskdir+out76str,out76m)
        
        orig78m = hp.read_map(maskdir+origcommonmaskf,field=0)
        out78str = 'COM_Mask_CMB-IQU-UT78_'+outNsidestr+'_R2.01.fits'
        out78m = downgrade_mask(orig78m,outNside)
        print "saving downgraded mask to",maskdir+out78str
        hp.write_map(maskdir+out78str,out78m)
        
    if 0: #small number of test maps
        t0 = time.time()
        run_manystats(realmin=0,realmax=10,\
                      #maskfile='',maskname='', \
                      maskfile = 'data/masks/COM_Mask_CMB-IQU-UT76_0064_R2.01.fits', maskname = 'UT76',
                      datadir = 'output/lcdm-map-testspace/',\
                      statdir = 'output/stat-testspace/',\
                      lvmapdir = 'output/lvmap-testspace/',\
                      mapbase = 'map',redoCl = 0, redoCt = 0, \
                      statlist =['S12','Cl2','s16','R10','R27','R50','Ct180','align23S','ALV'],\
                      Ncore = 2)
                      #statlist =['align23S'],Ncore = 0)
        t1 = time.time()
        print 'TOOK TIME',t1-t0

    if 0: #production run for map generation
        gen_manymaps_from_cldat(cldatfile = "data/COM_PowerSpect_CMB-base-plikHM-TT-lowTEB-minimum-theory_R2.02.txt",\
                        Nmaps = 100000, outdir ='/Users/Shared/lcdm_NSIDE64_maps_for_cmbparity/lcdm/', outbase = 'map')
        
    if 0: #production run with full sky maps
        t0 = time.time()
        run_manystats(realmin=0,realmax=99999,maskfile='',maskname='', \
                      datadir = '/Users/Shared/lcdm_NSIDE64_maps_for_cmbparity/lcdm/',\
                      statdir  = '/Users/Shared/lcdm_NSIDE64_maps_for_cmbparity/lcdm_fullsky_stats/',\
                      lvmapdir  = '/Users/Shared/lcdm_NSIDE64_maps_for_cmbparity/lcdm_fullsky_LVmaps/',\
                      mapbase = 'map',redoCl = 0, redoCt = 1, \
                      #statlist =['S12'],\
                      #statlist =['S12','Cl2','Cl3','R27','Ct180','Rall'],\
                      #statlist =['R10','R27','R50','Rall'],\
                      statlist =['S12','Ct180'],\
                      #statlist =['Cl3'],\
                      #statlist =['Rall'],\
                      #statlist =['ALV'],\
                      Ncore = 0, redoLVmaps = False)
        t1 = time.time()
        print 'TOOK TIME',t1-t0

    if 0: #production run with UT78 mask applied to realizations
        t0 = time.time()
        run_manystats(realmin=0,realmax=99999, \
                      datadir = '/Users/Shared/lcdm_NSIDE64_maps_for_cmbparity/lcdm/',\
                      statdir  = '/Users/Shared/lcdm_NSIDE64_maps_for_cmbparity/lcdm_UT78_stats/',\
                      lvmapdir  = '/Users/Shared/lcdm_NSIDE64_maps_for_cmbparity/lcdm_UT78_LVmaps/',\
                      mapbase = 'map',redoCl = 0, redoCt = 1, \
                      maskfile = 'data/masks/COM_Mask_CMB-IQU-UT78_0064_R2.01.fits', maskname = 'UT78',
                      #statlist =['S12'],\
                      #statlist =['S12','Cl2','s16','R10','R27','R50','Ct180','ALV'],\
                      #statlist =['ALV'],\
                      #statlist =['S12','Cl2','Cl3','R27','Ct180','Rall'],\
                      #statlist =['Rall'],\
                      statlist =['S12','Ct180'],\
                      Ncore = 0, redoLVmaps = False)

        t1 = time.time()
        print 'TOOK TIME',t1-t0
        
    #FFP maps
    if 0: #FFP run with full sky maps
        t0 = time.time()
        run_manystats(realmin=0,realmax=999,maskfile='',maskname='', \
                      datadir = '/Users/Shared/planck_ffp/planck_ffp8.1_compsep_0064_measanomalies-naming/',\
                      statdir  = '/Users/Shared/planck_ffp/smicaffp_fullsky_stats/',\
                      lvmapdir  = '/Users/Shared/planck_ffp/smicaffp_fullsky_LVmaps/',\
                      mapbase = 'smica-ffp8.1',redoCl = 0, redoCt = 1, \
                      #statlist =['S12'],\
                      #statlist =['ALV'],\
                      #statlist =['S12','Cl2','R10','R27','R50','Ct180','Rall'],\
                      #statlist =['S12','Cl2','Cl3','R27','Ct180','Rall'],\
                      statlist =['S12','Ct180'],\
                      #statlist =['align23S'],\
                      Ncore = 0, redoLVmaps = False)
        t1 = time.time()
        print 'TOOK TIME',t1-t0

    if 0: #FFP run with UT78 applied
        t0 = time.time()
        run_manystats(realmin=0,realmax=999, \
                      datadir = '/Users/Shared/planck_ffp/planck_ffp8.1_compsep_0064_measanomalies-naming/',\
                      statdir  = '/Users/Shared/planck_ffp/smicaffp_UT78_stats/',\
                      lvmapdir  = '/Users/Shared/planck_ffp/smicaffp_UT78_LVmaps/',\
                      mapbase = 'smica-ffp8.1',redoCl = 0, redoCt = 1, \
                      maskfile = 'data/masks/COM_Mask_CMB-IQU-UT78_0064_R2.01.fits', maskname = 'UT78',
                      #statlist =['R27'],\
                      statlist =['S12','Ct180'],\
                      #statlist =['S12','Cl2','s16','R10','R27','R50','Ct180','ALV'],\
                      #statlist =['ALV'],\
                      #statlist =['S12','Cl2','Cl3','R27','Ct180','Rall'],\
                      #statlist =['Rall'],\
                      #statlist =['R10','R27','R50'],\
                      #statlist =['s16'],\
                      #statlist =['align23S'],\
                      Ncore = 0, redoLVmaps = False)
        t1 = time.time()
        print 'TOOK TIME',t1-t0
        
    if 0: #FFP run with UT78 applied, Cl's only from Molinari
        t0 = time.time()
        run_manystats(realmin=0,realmax=999, \
                      datadir = '/Users/Shared/planck_ffp/planck_ffp8.1_molinariAPS/',\
                      statdir  = '/Users/Shared/planck_ffp/smicaffp-molinari_UT78_stats/',\
                      mapbase = 'smica-ffp8.1-molinari32',redoCl = 0, redoCt = 0, \
                      maskfile = '', maskname = 'UT78',
                      #statlist =['S12','Cl2','R10','R27','R50','Ct180','Rall'],\
                      statlist =['R27'],\
                      #statlist =['Cl3'],\
                      Ncore = 0, redoLVmaps = False, mapnside = None)
        #setting mapnside = None to avoid doing correction to cls
        t1 = time.time()
        print 'TOOK TIME',t1-t0

    #-----------------------
    # dipole quadropole correction version
    NDQcorr = 100000
    if 0: #make dipole corrected version of lcdm sims
         make_manyDQcorr_maps(indir ='/Users/Shared/lcdm_NSIDE64_maps_for_cmbparity/lcdm/',\
                              outdir ='/Users/Shared/lcdm_NSIDE64_maps_for_cmbparity/lcdm+DQ/',\
                              inbase='map',addtag='+DQ',Nmaps=NDQcorr)
    if 0: #full sky stat meas of DQ corr maps
        t0 = time.time()
        run_manystats(realmin=0,realmax=NDQcorr-1,maskfile='',maskname='', \
                      datadir = '/Users/Shared/lcdm_NSIDE64_maps_for_cmbparity/lcdm+DQ/',\
                      statdir  = '/Users/Shared/lcdm_NSIDE64_maps_for_cmbparity/lcdm+DQ_fullsky_stats/',\
                      lvmapdir  = '/Users/Shared/lcdm_NSIDE64_maps_for_cmbparity/lcdm+DQ_fullsky_LVmaps/',\
                      mapbase = 'map+DQ',redoCl = 0, redoCt = 0, \
                      #statlist =['S12','Cl2','Cl3','s16','R27','Ct180','align23S','Rall','ALV'],\
                      statlist =['R27','align23S'],\
                      #statlist =['S12','Ct180'],\
                      #statlist =['align23S','ALV'],\
                      Ncore = 0, redoLVmaps =False)
        t1 = time.time()
        print 'TOOK TIME',t1-t0
    if 0: #UT78 cut sky meas of DQ corr maps
        t0 = time.time()
        run_manystats(realmin=0,realmax=NDQcorr-1, \
                      datadir = '/Users/Shared/lcdm_NSIDE64_maps_for_cmbparity/lcdm+DQ/',\
                      statdir  = '/Users/Shared/lcdm_NSIDE64_maps_for_cmbparity/lcdm+DQ_UT78_stats/',\
                      lvmapdir  = '/Users/Shared/lcdm_NSIDE64_maps_for_cmbparity/lcdm+DQ_UT78_LVmaps/',\
                      mapbase = 'map+DQ',redoCl = 1, redoCt = 1, \
                      maskfile = 'data/masks/COM_Mask_CMB-IQU-UT78_0064_R2.01.fits', maskname = 'UT78',
                      #statlist =['S12','Cl2','Cl3','s16','R27','Ct180','Rall','ALV'],\
                      statlist =['S12','s16','Ct180','ALV'],\
                      #statlist =['S12','Ct180'],\
                      Ncore = 0, redoLVmaps = True)

        t1 = time.time()
        print 'TOOK TIME',t1-t0
##################################################################
if __name__ == "__main__":
    main()
