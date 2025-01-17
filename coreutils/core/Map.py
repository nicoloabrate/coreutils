import logging
import numpy as np
import itertools as it
from numpy import pi, cos, sin
from collections import OrderedDict
from coreutils.core.UnfoldCore import UnfoldCore
logger = logging.getLogger(__name__)

class Map:
    """
    Build the nuclear reactor core geometry map defined in a file.

    Parameters
    ----------
    geinp: str or np.array
        If ``str`` it is the <Path/filename> of the input file containing the
        geometry arrangement of the core, if ``np.array`` the geometry
        arrangement is already defined by the user.
    rotangle: int
        rotation angle over which the arrangement is symmetrically rotated.
        The rotation angle should be passed in degree (only 0,60,45,90,180
        values are allowed). With rotangle=0 no rotation occurs.
    Geom : ``Geometry``
        Geometry object.
    regionsdict: dict
    inp: np.array
        2D array representing the reactor core sector defined in input.
        The entries represent the assembly types.
    inpdict: dict
        Input dictionary, by default ``False``


    Attributes
    ----------
    inp: np.array
        2D array representing the reactor core sector defined in input.
        The entries represent the assembly types.
    type: np.array
        The geometry arrangement of the core.
    rotation_angle: int
        Rotation angle employed to unfold the input geometry
    Nx: int
        Number of assemblies along x
    Ny: int
        Number of assemblies along y
    fren2serp: dict
        Dictionary mapping the assemblies according to the FRENETIC numeration 
        to the one employed by Serpent 2.
    serp2fren: dict
        Dictionary mapping the assemblies according to the Serpent 2 numeration 
        to the one employed by FRENETIC.
    serpcentermap: dict
        Dictionary mapping assembly number to its center coordinates.

    """

    def __init__(self, geinp=None, rotangle=None, Geom=None,
                 regionsdict=None, inp=None, inpdict=None):
        if inpdict is None:
            self._init(geinp, rotangle, Geom, regionsdict, inp)
        else:
            self._from_dict(inpdict)

    def _init(self, geinp, rotangle, Geom, regionsdict, inp):
        if isinstance(geinp, str):
            rotangle = int(rotangle)
            core = UnfoldCore(geinp, rotangle, regionsdict)
            self.inp = core.inp
            self.type = core.coremap
        else:
            self.inp = inp
            self.type = geinp

        # -- compute assembly geometrical features
        self.rotation_angle = rotangle

        if rotangle == 0:
            # in case of full-core, extract first sextant and then use the usual algorithm
            N, M = self.inp.shape
            nnr = 0
            nnr_zero_top = 0
            for n in range(N):
                if sum(self.inp[n, :]) > 0:
                    nnr += 1
                else:
                    if nnr == 0:
                        nnr_zero_top += 1

            yc = nnr_zero_top + int(np.ceil(nnr/2)) - 1 # -1 for python idx
            
            nnc = 0
            nnc_zero_left = 0
            while nnc == 0:
                nnc_zero_left += 1
                if self.inp[yc, nnc_zero_left] != 0:
                    nnc = 1

            xc = nnc_zero_left + int(np.ceil((np.count_nonzero(self.inp[yc, :]))/2)) - 1

            # sanity check
            if N != M:
                raise MapError("Input map should be squared also for hexagonal cores if rotation angle != 60!")
            self.sector = np.zeros((N, M), dtype=int)
            # count non-zero assemblies along each row
            nL = []
            nnz = True
            row, col = yc, xc
            while nnz:
                nnz = np.count_nonzero(self.inp[row, col:])
                if nnz:
                    nL.append(nnz)
                    self.sector[row, col:col+nnz] = np.ones((nnz, ))
                row -= 1
                col += 1 if col != xc else 2
        else:
            self.sector = self.inp

        self.Nx, self.Ny = (self.type).shape
        # define assembly map
        serpmap = Map.__drawserpmap(self, Geom)  # Serpent numeration
        # define assembly centers coordinate
        coord = Map.__findcenters(self, Geom)

        if Geom.type == "H":
            # define assembly numeration according to FRENETIC
            frenmap = Map.__drawfrenmap(self)
            # Serpent to FRENETIC dict
            self.serp2fren = OrderedDict(zip(serpmap[:], frenmap[:]))
            # sort FRENETIC map in ascending way
            sortind = np.argsort(frenmap)
            frenmap = frenmap[sortind]
            # # sort Serpent map accordingly
            # serpmap = serpmap[sortind]
            # FRENETIC to Serpent dict
            self.fren2serp = dict(zip(frenmap, serpmap[sortind]))

        # Serpent centers map
        self.serpcentermap = dict(zip(serpmap[sortind], coord))

    def getSAsextant(self, sext):
        """Get SAs belonging to a certain core sextant (for hexagonal SAs).

        Parameters
        ----------
        sext: int
            Sextant number. The first sextant is obtained drawing an angle of 60 degrees in the quadrant x>0, y>0. Then the
            others are obtained moving counter-clockwise.
        """
        Nass = len((self.serpcentermap))
        nSAs = int(Nass/6)
        whichSA = [w for w in np.arange(nSAs*(sext-1)+2, nSAs*sext+2)]
        whichSA.append(1)
        return whichSA

    def _from_dict(self, inpdict):
        """Parse object from dictionary.

        Parameters
        ----------
        inpdict : dict
            Input dictionary containing the class object (maybe read from 
            external file).
        """
        for k, v in inpdict.items():
            setattr(self, k, v)

    def __findcenters(self, Geom):
        """Compute x and y coordinates of the centers of each assembly.
        
        Parameters
        ----------
        Geom: ``AssemblyGeometry``
            Geometry object.

        Returns
        -------
        coord: tuple
            Tuple of x and y coordinates of the centers of each assembly.

        """

        # define assemblies characteristics
        Nx, Ny = np.shape(self.type)
        L = Geom.edge  # assembly edge

        if Geom.type == "S":
            # evaluate coordinates points along x- and y- axes
            xspan = (np.arange(-(Nx-1)*L/2, (Nx)*L/2, L))
            yspan = np.flip(np.arange(-(Ny-1)*L/2, (Ny)*L/2, L))

            # replace "numerical zero" with 0
            xspan[abs(xspan) < 1e-10] = 0
            yspan[abs(yspan) < 1e-10] = 0

            # take cartesian product of coordinates set
            coord = tuple(it.product(xspan, yspan))

        elif Geom.type == "H":

            # define core geometrical features
            nsect = 6  # only six sextants for 60 degree rotation are allowed
            nass = nsect*(np.count_nonzero(self.sector)-1)+1  # tot ass.bly number
            P = Geom.pitch  # assembly pitch [cm]
            theta = pi/3  # hexagon inner angle
            L = Geom.edge  # hexagon edge [cm]
            x0 = P  # x-centre coordinate
            y0 = 0  # y-centre coordinate

            # unpack non-zero coordinates of the matrix
            y, x = np.where(self.sector != 0)  # rows, columns
            # define core central assembly
            yc, xc = [np.max(y), np.min(x)]  # central row, central column
            # compute number of assemblies
            Nx, Ny = [np.max(x)-xc, yc-np.min(y)+1]  # tot. columns, tot. rows
            # compute assemblies per row
            FA = [np.count_nonzero(row)
                  for row in self.sector[yc-Ny+1:yc+1, xc+1:]]
            # convert to np array
            FA = np.flip(np.asarray(FA))
            NFA = sum(FA)  # sum number of assemblies per sextant

            # arrays preallocation
            x, y = np.nan*np.empty((nass, )), np.nan*np.empty((nass, ))
            # assign central assembly coordinates
            x[0], y[0] = 0, 0
            count = 1
            # compute (x, y) of assembly centers
            for iSex in range(0, nsect):  # sextant loop
                # compute rotation matrix for each sextant
                rotmat = np.array([[cos(theta*iSex), -sin(theta*iSex)],
                                   [sin(theta*iSex), cos(theta*iSex)]])
                count = NFA*iSex+1  # keep score of number of assemblies done
                for irow in range(0, len(FA)):  # row loop
                    # lists preallocation
                    xc, yc = [], []
                    xcapp, ycapp = xc.append, yc.append
                    # compute centers for each assembly in irow
                    for icol in range(0, FA[irow]):  # column loop
                        xcapp(x0+P*icol+P/2*irow)
                        ycapp(y0+(L*(1+sin(pi/6)))*irow)

                    count = count+FA[irow-1]*(irow > 0)
                    xcyc = np.asarray([xc, yc])
                    FArot = np.dot(rotmat, xcyc)
                    iS, iE = count, count+FA[irow]
                    x[iS:iE, ] = FArot[0, :]
                    y[iS:iE, ] = FArot[1, :]

            coord = tuple(zip(x, y))

        return coord

    def __drawserpmap(self, Geom):
        """Define the core map  according to Serpent 2 code ordering.

        Parameters
        ----------
        Geom: ``AssemblyGeometry``
            Assembly radial geometry object.

        Returns
        -------
        serpmap : np.array
            Array with assembly numbers

        """
        # define assemblies characteristics
        Nx, Ny = np.shape(self.type)
        # define assembly numeration (same for squared and hex.lattice)
        assnum = np.arange(1, Nx*Ny+1)  # array with assembly numbers
        if Geom.type == "H":
            assnum = assnum.reshape(Nx, Ny).T  # reshape as a matrix
            assnum = assnum.flatten('F')  # flattening the matrix by columns
            # flatten the matrix by rows
            coretype = self.type.flatten('C')
            # squeeze out void assembly numbers
            assnum[coretype == 0] = 0
        elif Geom.type == "S":
            assnum = assnum.reshape(Nx, Ny)  # reshape as a matrix
            assnum = assnum.flatten('F')  # flattening the matrix by columns

        # select non-zero elements
        sermap = assnum[assnum != 0]

        return sermap

    def __drawfrenmap(self):
        """Define the core map according to FRENETIC code ordering.

        Parameters
        ----------
        Geom: ``AssemblyGeometry``
            Assembly radial geometry object.

        Returns
        -------
        frenmap : np.array
            Array with assembly numbers

        """
        # check on geometry
        if self.rotation_angle != 60 and self.rotation_angle != 0:
            logger.critical("FrenMap method works only for hexagonal core geometry!")
            raise OSError("rotation angle should be 60 or 0 degrees")

        nsect = 6  # only six sextants for 60 degree rotation are allowed

        nL = np.count_nonzero(self.sector, axis=1)
        self.nonZeroCols = nL[nL != 0]
        frenmap = self.sector + 0  # copy input matrix

        # number of assemblies
        nass = nsect*(np.count_nonzero(frenmap)-1)+1
        # unpack non-zero coordinates of the matrix
        y, x = np.where(frenmap != 0)  # rows, columns
        # define core central assembly
        yc, xc = [np.max(y), np.min(x)]
        # compute number of assemblies
        Nx, Ny = [np.max(x)-xc, yc-np.min(y)+1]
        # frenmap[yc, xc] = 1  # first assembly is the central one
        iS = 1  # the 1st assembly is the central
        for irow in range(0, Ny):
            # take extrema of non-zero arrays
            NZx = np.flatnonzero(frenmap[yc-irow, xc:])
            # compute array length to numerate assemblies
            iE = NZx[-1]-NZx[0]+1  # length of array
            # write assembly numbers
            frenmap[yc-irow, NZx[0]+xc:NZx[-1]+1+xc] = np.arange(iS, iE+iS)
            # keep record of assembly number
            iS = iS+iE

        # unpack non-zero coordinates of the matrix
        y, x = np.where(frenmap != 0)  # rows, columns
        # define core central assembly
        yc, xc = [np.max(y), np.min(x)]
        # compute number of assemblies
        Nx, Ny = [np.max(x)-xc, yc-np.min(y)+1]
        Nxc = sum(frenmap[xc, yc+1:] != 0)
        if Nxc < Nx:
            # non-regular hexagon: add dummy elements to have same Nx and Ny
            frenmap[xc, Nxc+xc+1:Nx+xc+1] = -100000000000
            # find min non-zero index
            indNZy = np.argwhere(frenmap[:, np.max(x)]).min()
            # add elements along y to have Nx=Ny
            frenmap[yc-Nx+1:indNZy, np.max(x)] = -100000000000
            # set Ny equal to Nx (by definition)
            Ny = Nx
            # unpack non-zero coordinates of the matrix (they changed)
            y, x = np.where(frenmap != 0)  # rows, columns

        # -- loop over rows
        for nb in range(1, Nx):

            xsectI = np.arange(xc+Nx, xc+nb, -1)  # decreasing order indeces
            ysectI = yc-nb  # going up along the matrix means to decrease yc
            # select non-zero elements position
            NZx = np.flatnonzero(frenmap[ysectI, xsectI])
            # -- sector II
            # count non-zero element for summation (continue numeration)
            nz = (nass-1)/nsect*(frenmap[ysectI, xsectI] > 0)
            # select rotation coordinates
            dcoord = [np.arange(yc-nb-1, np.min(y)-2, -1),
                      np.arange(xc+1, xc+NZx[-1]+2)]
            # select indeces  matching dcoord
            ind = np.ravel_multi_index(dcoord, dims=np.shape(frenmap),
                                       order='C')
            # write ass numbers
            frenmap.ravel()[ind] = np.flip(frenmap[ysectI, xsectI]+nz)

            # -- sector III
            frenmap[yc-Ny+nb:yc, xc-nb] = frenmap[ysectI, xsectI]+2*nz

            # -- sector IV
            frenmap[yc+nb, xc-Nx:xc-nb] = frenmap[ysectI, xsectI]+3*nz

            # flip since sectors V and VI are in III and IV cartesian quadrants
            xsectI = np.flip(xsectI, -1)

            # -- sector V
            # select rotation coordinates
            dcoord = [np.arange(xc+nb+1, xc+NZx[-1]+2+nb),
                      np.arange(yc-1, np.min(y)-1+(nb-1), -1)]
            # TODO: this is only a temporary patch for ebr-II case
            if dcoord[0].shape != dcoord[1].shape:
                dcoord[0] = np.arange(xc+nb+1,
                                      xc+NZx[-1]+1*(NZx[-1] == 0)+2+nb)
            # select indeces  matching dcoord
            ind = np.ravel_multi_index(dcoord, dims=np.shape(frenmap),
                                       order='C')
            # write ass numbers
            frenmap.ravel()[ind] = frenmap[ysectI, xsectI]+np.flip(4*nz)

            # -- sector VI
            frenmap[yc+1:yc+Ny-nb+1, xc+nb] = frenmap[ysectI,
                                                      xsectI]+np.flip(5*nz)

        # -- fill "diagonals" (only sectors II and V actually are diagonals)
        corediag = frenmap[yc, xc+1:xc+Nx+1]

        # sector II
        # select rotation coordinates
        dcoord = (np.arange(xc+1, xc+Nx+1), np.arange(yc-1, yc-Ny-1, -1))
        # select indeces  matching dcoord
        ind = np.ravel_multi_index(dcoord, dims=np.shape(frenmap), order='F')
        nz = (nass-1)/nsect
        frenmap.ravel()[ind] = corediag+nz

        # sector III
        nz = 2*(nass-1)/nsect
        frenmap[np.arange(yc-1, yc-Ny-1, -1), xc] = corediag+nz

        # sector IV
        nz = 3*(nass-1)/nsect
        frenmap[yc, np.arange(xc-1, xc-Nx-1, -1)] = corediag+nz

        # sector V
        # select rotation coordinates
        dcoord = (np.arange(xc-1, xc-Nx-1, -1), np.arange(yc+1, yc+Ny+1))
        # select indeces  matching dcoord
        ind = np.ravel_multi_index(dcoord, dims=np.shape(frenmap), order='F')
        nz = 4*(nass-1)/nsect
        frenmap.ravel()[ind] = corediag+nz

        # sector VI
        nz = 5*(nass-1)/nsect
        frenmap[np.arange(yc+1, yc+Ny+1), xc] = corediag+nz

        # replace negative dummy elements with 0
        frenmap[frenmap < 0] = 0

        # flatten 2D array along 1 axis
        frenmap = frenmap.flatten('C')
        # squeeze out 0 assemblies
        frenmap = frenmap[frenmap != 0]

        return frenmap


class MapError(Exception):
    pass