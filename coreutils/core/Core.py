"""
Author: N. Abrate.

File: core.py

Description: Class to define the nuclear reactor core geometry defined in an
external text file.
"""
import os
import re
import sys
import json
# from tkinter import NE
import numpy as np
import coreutils.tools.h5 as myh5

from copy import deepcopy as cp
# from collections import OrderedDict
from pathlib import Path
from coreutils.tools.utils import parse, MyDict
from coreutils.core.Map import Map
from coreutils.core.UnfoldCore import UnfoldCore
from coreutils.core.MaterialData import * # NEMaterial, readSerpentRes, CZMaterialData
from coreutils.core.Assembly import AssemblyGeometry, AxialConfig, AxialCuts


class Core:
    """
    Define NE, TH and CZ core configurations.

    Attributes
    ----------
    AssemblyGeom : obj
        Object with assembly geometrical features.
    NElabels : dict
        Dictionary with regions names and strings for plot labelling.
    NEassemblytypes : dict
        Ordered dictionary with names of assembly NE types.
    THassemblytypes : dict
        Ordered dictionary with names of assembly TH types.
    CZassemblytypes : dict
        Ordered dictionary with names of assembly CZ types.
    Map : obj
        Object mapping the core assemblies with the different numerations.
    NEAxialConfig : obj
        Axial regions defined for NE purposes.
    NEMaterialData : obj
        NE data (multi-group constants) for each region defined in input.
    CZMaterialData : obj
        NE data (multi-group constants) for each region defined in input.
    NEconfig : dict
        Neutronics configurations according to time.
    NEtime : list
        Neutronics time instants when configuration changes.
    CZconfig : dict
        Cooling zones configurations according to time.
    CZtime : list
        Cooling zones time instants when configuration changes.

    Methods
    -------
    getassemblytype :
        Get type of a certain assembly.
    replace :
        Replace assemblies with user-defined new or existing type.
    perturb :
        Replace assemblies with user-defined new or existing type.
    translate :
        Replace assemblies with user-defined new or existing type.
    perturbBC :
        Spatially perturb cooling zone boundary conditions.
    writecentermap :
        Write assembly number and x and y coordinates of centers to text file.
    getassemblylist :
        Return assemblies belonging to a certain type.
    writecorelattice :
        Write core lattice to txt file.
    """

    def __init__(self, inpjson):

        if isinstance(inpjson, dict):
            print('TODO')
        elif '.h5' in inpjson:
            self.from_h5(inpjson)
        elif ".json" in inpjson:
            self.from_json(inpjson)
        else:
            raise OSError("Input file must be in .json format!")

    def from_json(self, inpjson):
        # -- parse input file
        if ".json" in inpjson:

            CIargs, NEargs, THargs = parse(inpjson)

            tEnd = CIargs['tEnd']
            nProf = CIargs['nSnap'] 
            pitch = CIargs['pitch'] 
            shape = CIargs['shape'] 
            power = CIargs['power'] 
            trans = CIargs['trans']
            dim = CIargs['dim']

            # check if NEargs and THargs are not empty
            isNE = True if NEargs is not None else False
            isTH = True if THargs is not None else False
            if isNE:
                isPH = True if 'isPH' in NEargs.keys() else False
            if isPH:
                PHargs = NEargs['PH']
        else:
            raise OSError("Input file must be in .json format!")

        # --- sanity check
        if dim != 1 and shape == 'H':
            flag = False
            if THargs is not None:
                if 'rotation' in THargs.keys():
                    flag = THargs['rotation'] != 60
            if NEargs is not None:
                if 'rotation' in NEargs.keys():
                    if not flag:
                        flag = NEargs['rotation'] != 60
            if flag:
                raise OSError('Hexagonal core geometry requires one sextant, so' +
                            '"rotation" must be 60 degree!')

        if not isNE:
            if dim == 1:
                raise OSError('Cannot generate core object without NE namelist!')
            else:
                print('NE input not available, writing TH input only!')
        if not isTH and dim != 1:
            print('TH input not available, writing NE input only!')

        # --- handle temperatures, power, time and core dimensions (1, 2 or 3 D)
        TfTc = []
        # ensure ascending order
        CIargs['Tf'].sort()
        CIargs['Tc'].sort()
        fuel_temp = CIargs['Tf']
        cool_temp = CIargs['Tc']

        for Tf in fuel_temp:
            for Tc in cool_temp:
                if Tf >= Tc:
                    TfTc.append((Tf, Tc))
        self.TfTc = TfTc
        self.Tf = Tf
        self.Tc = Tc
        self.TimeEnd = tEnd
        self.trans = trans
        self.dim = dim
        self.power = power

        if isinstance(nProf, (float, int)):
            dt = tEnd/nProf
            self.TimeSnap = np.arange(0, tEnd+dt, dt) if dt > 0 else [0]
        elif isinstance(nProf, list) and len(nProf) > 1:
            self.TimeSnap = nProf
        else:
            raise OSError('nSnap in .json file must be list with len >1, float or int!')

        # --- initialise assembly radial geometry object
        if dim != 1:
            self.AssemblyGeom = AssemblyGeometry(pitch, shape)  # module indep.

        if isNE:
            # parse inp args
            NEassemblynames = NEargs['assemblynames']
            cuts = {'xscuts': NEargs['xscuts'],
                    'zcuts': NEargs['zcuts']}
            config = NEargs['config']
            NEfren = NEargs['fren']
            NEassemblylabel = NEargs['assemblylabel']
            NEdata = NEargs['NEdata']

            self.NEtime = [0]
            # --- Axial geometry, if any
            if cuts is not None and dim != 2:
                # initial axial configuration
                self.NEAxialConfig = AxialConfig(cuts, NEargs['splitz'],
                                                 labels=NEargs['labels'], NE_dim=self.dim)

            # --- count and store initial regions
            if self.dim == 2:
                # SAs types are the regions
                nReg = len(NEassemblynames)
                nAssTypes = nReg
                self.NEregions = MyDict(dict(zip(np.arange(1, nReg+1), NEassemblynames)))
                self.NElabels = dict(zip(self.NEregions.values(), self.NEregions.values()))
                # define dict mapping strings into ints for assembly type
                self.NEassemblytypes = MyDict(dict(zip(np.arange(1, nReg+1),
                                                            NEargs['assemblynames'])))
            else:
                self.NEregions = self.NEAxialConfig.regions
                self.NElabels = self.NEAxialConfig.labels
                self.NEassemblytypes = MyDict()
                NZ = len(self.NEAxialConfig.zcuts)-1
                NEtypes = ['slab'] if self.dim == 1 else NEassemblynames
                nAssTypes = len(NEtypes)
                # loop over SAs types (one cycle for 1D)
                for iType, NEty in enumerate(NEtypes):
                    self.NEassemblytypes[iType+1] = NEty

            # --- get assembly types, names and parse all regions names
            self.NEconfig = {}
            NEassemblynames = MyDict(dict(zip(NEassemblynames, np.arange(1, nAssTypes+1))))
            # --- define core Map, assembly names and types
            if dim != 1:
                tmp = UnfoldCore(NEargs['filename'], NEargs['rotation'], NEassemblynames)
                NEcore = tmp.coremap
                self.NEmap = Map(NEcore, NEargs['rotation'], self.AssemblyGeom, inp=tmp.inp)
                if 'Map' not in self.__dict__.keys():
                    self.Map = Map(NEcore, NEargs['rotation'], self.AssemblyGeom, inp=tmp.inp)
                if 'Nass' not in self.__dict__.keys():
                    self.NAss = len((self.Map.serpcentermap))
            else:
                NEcore = [1]
                self.NAss = 1
            # --- define core time-dep. configurations
            self.NEconfig[0] = NEcore

            # --- define core time-dep. configurations
            # save coordinates for each layer (it can be updated later!)
            self.NEzcoord = MyDict()
            zc = self.NEAxialConfig.zcuts
            for iR, z1z2 in enumerate(zip(zc[:-1], zc[1:])):
                self.NEzcoord[iR] = z1z2

            if NEassemblylabel is not None:
                self.NEassemblylabel = MyDict(dict(zip(np.arange(1, nAssTypes+1),
                                                    NEassemblylabel)))
            else:
                self.NEassemblylabel = cp(self.NEassemblytypes)

            # --- parse names of NE universes (==region mix)
            if dim != 2:
                univ = []  # consider axial regions in "cuts"
                for k in self.NEAxialConfig.cuts.keys():
                    univ.extend(self.NEAxialConfig.cuts[k].reg)
            else:
                univ = cp(NEassemblynames)  # regions are assembly names
            # squeeze repetitions
            univ = list(set(univ))

            if NEdata is not None:
                # --- NE ENERGY GRID
                if 'egridname' in NEargs.keys():
                    ename = NEargs['egridname']
                else:
                    ename = None

                if 'energygrid' in NEargs.keys():
                    energygrid = NEargs['energygrid']
                else:
                    energygrid = ename

                if isinstance(energygrid, (list, np.ndarray, tuple)):
                    self.nGro = len(energygrid)-1
                    self.NEegridname = f'{self.nGro}G' if ename is None else ename
                    self.NEenergygrid = energygrid
                elif isinstance(energygrid, (str, float, int)):
                    pwd = Path(__file__).parent.parent.parent
                    if 'coreutils' not in str(pwd):
                        raise OSError(f'Check coreutils tree for NEdata: {pwd}')
                    else:
                        pwd = pwd.joinpath('NEdata')
                        if isinstance(energygrid, str):
                            fgname = f'{energygrid}.txt'
                            self.NEegridname = str(energygrid)
                        else:
                            fgname = f'{energygrid}G.txt'
                            self.NEegridname = str(energygrid)

                        egridpath = pwd.joinpath('group_structures', fgname)
                        self.NEenergygrid = np.loadtxt(egridpath)
                        self.nGro = len(self.NEenergygrid)-1
                else:
                    raise OSError(f'Unknown energygrid \
                                  {type(energygrid)}')

                if self.NEenergygrid[0] < self.NEenergygrid[0]:
                    self.NEenergygrid[np.argsort(-self.NEenergygrid)]

                # --- NE MATERIAL DATA
                try:
                    path = NEdata['path']
                except KeyError:
                    pwd = Path(__file__).parent.parent.parent
                    if 'coreutils' not in str(pwd):
                        raise OSError(f'Check coreutils tree for NEdata: {pwd}')

                    # look into default NEdata dir
                    path = str(pwd.joinpath('NEdata', energygrid))

                if "checktempdep" not in NEdata.keys():
                    NEdata["checktempdep"] = 0

                try:
                    files = NEdata['beginwith']
                except KeyError:
                    # look for Serpent files in path/serpent
                    serpath = str(pwd.joinpath('NEdata', f'{self.NEegridname}',
                                              'serpent'))
                    files = [f for f in os.listdir(serpath)]

                self.NEMaterialData = {}
                for temp in TfTc:
                    self.NEMaterialData[temp] = {}
                    tmp = self.NEMaterialData[temp]
                    # get temperature for OS operation on filenames or do nothing
                    T = temp if NEdata["checktempdep"] else None
                    # look for all data in Serpent format
                    serpres = {}
                    serpuniv = []
                    for f in files:
                        sdata = readSerpentRes(path, self.NEenergygrid, T, 
                                                beginswith=f, egridname=self.NEegridname)
                        if sdata is not None:
                            serpres[f] = sdata
                            for univtup in sdata.universes.values():
                                # access to HomogUniv attribute name
                                serpuniv.append(univtup.name)

                    for u in univ:
                        if u in serpuniv:
                            tmp[u] = NEMaterial(u, self.NEenergygrid, egridname=self.NEegridname, 
                                                serpres=serpres, temp=T)
                        else: # look for data in json and txt format
                            tmp[u] = NEMaterial(u, self.NEenergygrid, 
                                                egridname=self.NEegridname,
                                                datapath=path, temp=T, basename=u)
                    # --- HOMOGENISATION (if any)
                    if self.NEAxialConfig.homogenised and self.dim != 2: 
                        for u0 in self.NEregions.values():
                            if "+" in u0: # homogenisation is needed
                                # identify SA type and subregions
                                strsplt = re.split(r"\d_", u0, maxsplit=1)
                                NEty = strsplt[0]
                                names = re.split(r"\+", strsplt[1])
                                # parse weights
                                w = np.zeros((len(names), ))
                                for iM, mixname in enumerate(names):
                                    idx = self.NEAxialConfig.cutsregions[NEty][f"M{iM+1}"].index(mixname)
                                    w[iM] = self.NEAxialConfig.cutsweights[NEty][f"M{iM+1}"][idx]
                                # perform homogenisation
                                mat4hom = {}
                                for name in names:
                                    mat4hom[name] = self.NEMaterialData[temp][name]
                                weight4hom = dict(zip(names, w))
                                tmp[u0] = Homogenise(mat4hom, weight4hom, u0)

                # --- check precursors family consistency
                NP = -1
                NPp = -1
                for temp in TfTc:
                    for k, mat in self.NEMaterialData[temp].items():
                        if NP == -1:
                            NP = mat.NPF
                        else:
                            if NP != mat.NPF:
                                raise OSError(f'Number of neutron precursor families in {k} '
                                            'not consistent with the other regions!')
                    if isPH: # add photon data
                        for k, mat in self.PHMaterialData[temp].items():
                            if NPp == -1:
                                NPp = mat.NPF
                            else:
                                if NPp != mat.NPF:
                                    raise OSError(f'Number of photon precursor families in {k} '
                                                'not consistent with the other regions!')

                self.nPre = NP
                if isPH:
                    self.nPrp = NP
                    self.nGrp = len(self.PHenergygrid)-1
                    self.nDhp = 1 # FIXME TODO!
                    print("WARNING: DHP set to 1!")
                else:
                    self.nPrp = 0
                    self.nGrp = 0
                    self.nDhp = 0
            # --- do replacements if needed at time=0 s
            if NEargs["replaceSA"] is not None:
                self.replaceSA(NEargs["replaceSA"], 0, isfren=NEfren)
            if NEargs["replace"] is not None:
                self.replace(NEargs["replace"], 0, isfren=NEfren)

            # --- build time-dependent core configuration
            if config is not None:
                for time in config.keys():
                    if time != '0':
                        t = float(time)
                        # increment time list
                        self.NEtime.append(t)
                    else:
                        # set initial condition
                        t = 0
                    # check operation
                    if "translate" in config[time]:
                        self.translate(config[time]["translate"], time,
                                       isfren=NEfren)

                    if "perturb" in config[time]:
                        self.perturb(config[time]["perturb"], time,
                                     isfren=NEfren)

                    if "replace" in config[time]:
                        self.replace(config[time]["replace"], time,
                                     isfren=NEfren)

                    if "replaceSA" in config[time]:
                        self.replaceSA(config[time]["replaceSA"], time,
                                     isfren=NEfren)
            # --- clean dataset from unused regions
            if NEdata is not None:
                # remove Material objects if not needed anymore
                for temp in TfTc:
                    tmp = self.NEMaterialData[temp]
                    universes = list(tmp.keys())
                    for u in universes:
                        if u not in self.NEregions.values():
                            tmp.pop(u)
        # --- define TH core geometry and data
        if isTH and dim != 1:
            CZassemblynames = THargs['coolingzonenames']
            THdata = THargs['THargs']
            # sort list
            assnum = np.arange(1, len(CZassemblynames)+1)

            CZassemblynames = MyDict(dict(zip(CZassemblynames, assnum)))
            # define dict between strings and ints for assembly type
            self.CZassemblytypes = MyDict(dict(zip(assnum,
                                                        CZassemblynames)))

            # define TH core with assembly types
            CZcore = UnfoldCore(THargs['coolingzonesfile'], THargs['rotation'], CZassemblynames).coremap

            if THdata is not None:
                THassemblynames = THdata['assemblynames']
                assnum = np.arange(1, len(THassemblynames)+1)
                THassemblynames = MyDict(dict(zip(THassemblynames,
                                                       assnum)))
                # define dict between strings and ints for assembly type
                self.THassemblytypes = MyDict(dict(zip(assnum,
                                                        THassemblynames)))
                THinp = THdata['filename']
                tmp = UnfoldCore(THinp, THargs['rotation'], THassemblynames)
                THcore = tmp.coremap
                THinp = tmp.inp

                # define input matrix for core mapping
                # if already assigned, not an issue if shape is ok
                rotation = THargs['rotation']
                self.THmap = Map(THcore, rotation, self.AssemblyGeom, inp=THinp)
                if 'Map' not in self.__dict__.keys():
                    self.Map = Map(THcore, rotation, self.AssemblyGeom, inp=THinp)
                if 'Nass' not in self.__dict__.keys():
                    self.NAss = len((self.Map.serpcentermap))

            if THcore.shape != CZcore.shape:
                raise OSError("CZ and TH core dimensions mismatch!")
            
            if THdata is not None and "replace" in THdata.keys():
                # loop over assembly types
                for k, v in THdata["replace"].items():
                    try:
                        THcore = self.replace(THassemblynames[k], v, THargs['fren'],
                                              THcore)
                    except KeyError:
                        raise OSError("%s not present in TH assembly types!"
                                      % k)
            # TH configuration
            self.THtime = [0]
            self.THconfig = {}
            self.THconfig[0] = THcore

            # CZ replace
            if isinstance(THargs['replace'], dict):
                # loop over assembly types
                for k, v in THargs['replace'].items():
                    try:
                        CZcore = self.replace(CZassemblynames[k], v, THargs['fren'],
                                              CZcore)
                    except KeyError:
                        raise OSError("%s not present in CZ assembly types!"
                                      % k)
            else:
                if THargs['replace'] is not None:
                    raise OSError("'replace' in TH must be of type dict!")

            # assign material properties
            cz = CZMaterialData(THargs['massflowrates'], THargs['pressures'], 
                                THargs['temperatures'], self.CZassemblytypes.values())
            self.CZMaterialData = cz

            # keep each core configuration in time
            self.CZconfig = {}
            self.CZconfig[0] = CZcore
            self.CZtime = [0]
            # check if boundary conditions change in time
            THbcs = THargs['boundaryconditions']
            if THbcs is not None:
                for time in THbcs.keys():
                    t = float(time)
                    # increment time list
                    self.CZtime.append(t)
                    self.perturbBC(THbcs[time], time, isfren=THargs['fren'])

        # --- NE and TH final consistency check
        if isNE and isTH:
            # dimensions consistency check
            if CZcore.shape != NEcore.shape:
                raise OSError("NE and TH core dimensions mismatch:" +
                              f"{CZcore.shape} vs. {NEcore.shape}")

            # non-zero elements location consistency check
            tmp1 = cp(CZcore)
            tmp1[tmp1 != 0] = 1

            tmp2 = cp(NEcore)
            tmp2[tmp2 != 0] = 1

            if THdata is not None:
                tmp3 = cp(THcore)
                tmp3[tmp3 != 0] = 1

            if (tmp1 != tmp2).all():
                raise OSError("Assembly positions in CZ and NE mismatch. " +
                              "Check core input file!")

            if (tmp1 != tmp3).all():
                raise OSError("Assembly positions in CZ and TH mismatch. " +
                              "Check core input file!")

    def from_h5(self, h5name):
        """Instantiate object from h5 file.

        Parameters
        ----------
        h5name : str
            Path to h5 file.
        """
        h5f = myh5.read(h5name, metadata=False)
        mydicts = ["NEassemblytypes", "NEregions", "NEzcoord", "NEassemblylabel",
                   "CZassemblynames", "THassemblynames"]
        for k, v in h5f.core.items():
            if k == "NEAxialConfig":
                setattr(self, k, AxialConfig(inpdict=v))
            elif k == 'AssemblyGeom':
                setattr(self, k, AssemblyGeometry(inpdict=v))
            elif k == 'Map':
                setattr(self, k, Map(inpdict=v))
            else:
                if k in mydicts:
                    setattr(self, k, MyDict(v))
                else:
                    setattr(self, k, v)

    def getassemblytype(self, assemblynumber, time=0, isfren=False,
                        whichconf="NEconfig"):
        """
        Get type of a certain assembly.

        Parameters
        ----------
        assemblynumber : int
            Number of the assembly of interest
        isfren : bool, optional
            Flag for FRENETIC numeration. The default is False.

        Returns
        -------
        which : int
            Type of assembly.

        """
        if self.dim == 1:
            return self.NEconfig[time][0]
        if isfren:
            # translate FRENETIC numeration to Serpent
            index = self.Map.fren2serp[assemblynumber]-1  # -1 for py indexing
        else:
            index = assemblynumber-1  # -1 for py indexing
        # get coordinates associated to these assemblies
        rows, cols = np.unravel_index(index, self.Map.type.shape)

        if whichconf == "NEconfig":
            which = self.NEconfig[time][rows, cols]
        elif whichconf == "THconfig":
            which = self.THconfig[time][rows, cols]
        elif whichconf == "CZconfig":
            which = self.CZconfig[time][rows, cols]
        else:
            raise OSError("Unknown core config!")

        return which

    def replaceSA(self, repl, time, isfren=False):
        """
        Replace full assemblies or axial regions.

        Parameters
        ----------
        repl : dict
            Dictionary with SA name as key and list of SAs to be replaced as value
        isfren : bool, optional
            Flag for FRENETIC numeration. The default is ``False``.

        Returns
        -------
        ``None``

        """
        if float(time) in self.NEconfig.keys():
            now = float(time)
        else:
            nt = self.NEtime.index(float(time))
            now = self.NEtime[nt-1]
        
        asstypes = self.NEassemblytypes.reverse()
        for NEtype in repl.keys():
            if NEtype not in self.NEassemblytypes.values():
                raise OSError(f"SA {NEtype} not defined! Replacement cannot be performed!")
            lst = repl[NEtype]
            if not isinstance(lst, list):
                raise OSError("replaceSA must be a dict with SA name as key and"
                                "a list with assembly numbers (int) to be replaced"
                                "as value!")
            if self.dim == 1:
                newcore = [asstypes[NEtype]]
            else:
                # --- check map convention
                if isfren:
                    # translate FRENETIC numeration to Serpent
                    index = [self.Map.fren2serp[i]-1 for i in lst]  # -1 for index
                else:
                    index = [i-1 for i in lst]  # -1 to match python indexing
                # --- get coordinates associated to these assemblies
                index = (list(set(index)))
                rows, cols = np.unravel_index(index, self.Map.type.shape)
                newcore = self.NEconfig[now]+0
                # --- load new assembly type
                newcore[rows, cols] = asstypes[NEtype]
            
            self.NEconfig[float(time)] = newcore

    def replace(self, rpl, time, isfren=False, action='repl'):
        """
        Replace full assemblies or axial regions.
        
        This method is useful to replace axial regions in 1D or 3D models. Replacements can
        affect disjoint regions, but each replacement object should involve either the
        region subdivision (self.NEAxialConfig.zcuts) or the xscuts subdivision (the one in 
        self.NEAxialConfig.cuts[`AssType`]). If the replacement affect this last axial grid,
        homogenised data are computed from scratch and added to the material regions.
        The methods ``perturb`` and ``translate`` rely on this method to arrange the new
        regions.

        Parameters
        ----------
        isfren : bool, optional
            Flag for FRENETIC numeration. The default is ``False``.

        Returns
        -------
        ``None``

        """
        if self.dim == 2:
            return None
        
        if not isinstance(rpl, dict):
            raise OSError("Replacement object must be a dict with"
                          " `which`, `where` and `with` keys!")
        # map region names into region numbers
        regtypes = self.NEregions.reverse()
        if len(rpl['which']) != len(rpl['where']) or len(rpl['where']) != len(rpl['with']):
            raise OSError('Replacement keys must have the same number of elements '
                          'for which, where and with keys!')

        pconf = zip(rpl['which'], rpl['with'], rpl['where'])

        if float(time) in self.NEconfig.keys():
            now = float(time)
        else:
            nt = self.NEtime.index(float(time))
            now = self.NEtime[nt-1]

        iR = 0  # replacement counter
        for r in list(self.NEregions.values()):
            if action in r:
                iR += 1

        for which, withreg, where in pconf:
            iR += 1
            # arrange which into list of lists according to SA type
            whichlst = {}
            for w in which:
                itype = self.getassemblytype(w, now, isfren=isfren,
                                             whichconf="NEconfig")
                if itype not in whichlst.keys():
                    whichlst[itype] = []
                whichlst[itype].append(w)

            for itype, assbly in whichlst.items(): # loop over each assembly
                atype = self.NEassemblytypes[itype]
                # --- parse replacement axial locations
                axpos = []
                axposapp = axpos.append

                cutaxpos = [] 
                cutaxposapp = cutaxpos.append
                if not isinstance(where, list):
                    where = [where]
                else:
                    if len(where) == 2:
                        if not isinstance(where[0], list):
                            where = [where]
                for rplZ in where:
                    rplZ.sort()
                    notfound = True  # to check consistency of "where" arg
                    incuts = False  # replacement may be in SA cuts
                    for ipos, coord in self.NEzcoord.items(): # check in zcoords
                        if tuple(rplZ) == coord:
                            notfound = False
                            axposapp(ipos)
                            break
                    if notfound: # check in cuts
                        zcuts = zip(self.NEAxialConfig.cuts[atype].loz, self.NEAxialConfig.cuts[atype].upz)
                        for ipos, coord in enumerate(zcuts):
                            if tuple(rplZ) == coord:
                                notfound = False
                                incuts = True
                                cutaxposapp(ipos)
                                break
                        if notfound:
                            raise OSError(f"Cannot find axial region in {rplZ} for replacement!")
                # --- avoid replacement in xscuts and zcuts at the same time
                if len(axpos) > 0 and len(cutaxpos) > 0:
                    raise OSError('Cannot replace in xscuts and zcuts at the same time!'
                                  ' Add separate replacements!')
                # --- identify new region number (int)
                if isinstance(withreg, list):  # look for region given its location
                    withwhich, z = withreg[0], withreg[1]
                    if not isinstance(z, list):
                        raise OSError("Replacement: with should contain a list with integers"
                                        " and another list with z1 and z2 coordinates!")
                    z.sort()
                    notfound = True  # to check consistency of "withreg" arg
                    for ipos, coord in self.NEzcoord.items():
                        if tuple(z) == coord:
                            notfound = False
                            break
                    if notfound:
                        for ipos, coord in enumerate(zcuts):
                            if tuple(rplZ) == coord:
                                notfound = False
                                break
                        if notfound:
                            raise OSError(f"Cannot find axial region to be replaced in {z}!")

                    iasstype = self.getassemblytype(withwhich, now, isfren=isfren, 
                                                        whichconf="NEconfig")
                    if not incuts:
                        asstype = self.NEassemblytypes[iasstype]
                        newreg_str = self.NEAxialConfig.config_str[asstype][ipos]
                        newlab_str = self.NElabels[newreg_str]
                        newreg_int = self.NEAxialConfig.config[iasstype][ipos]
                    else:
                        newreg_str = self.NEAxialConfig.cuts[atype].reg[ipos]
                        newlab_str = self.NEAxialConfig.cuts[atype].label[ipos]
                        newreg_int = False
                elif isinstance(withreg, str):
                    newreg_str = withreg
                    if not incuts:
                        newreg_int = regtypes[withreg]
                        newlab_str = self.NElabels[newreg_str]
                    else:
                        newreg_int = False
                        idx = self.NEAxialConfig.cuts[atype].reg.index(newreg_str)
                        newlab_str = self.NEAxialConfig.cuts[atype].labels[idx]
                else:
                    raise OSError("'with' key in replacemente must be list or string!")

                # --- update object with new SA type
                if action in atype:
                    regex = rf"\-[0-9]{action}" # TODO test with basename like `IF-1-XXX-1repl`
                    basetype = re.split(regex, atype, maxsplit=1)[0]
                    newtype = f"{basetype}-{iR}{action}"
                    oldtype = atype
                else:
                    newtype = f"{atype}-{iR}{action}"
                    oldtype = newtype

                nTypes = len(self.NEassemblytypes.keys())
                newaxregions = cp(self.NEAxialConfig.config[itype])
                newaxregions_str = cp(self.NEAxialConfig.config_str[atype])
                if not incuts:
                    for ax in axpos:
                        newaxregions[ax] = newreg_int
                        newaxregions_str[ax] = newreg_str
                    # add new type in xscuts (mainly for plot)
                    for rplZ in where:
                        cuts = cp(self.NEAxialConfig.cuts[atype])
                        upz, loz, reg, lab = cuts.upz, cuts.loz, cuts.reg, cuts.labels
                        if rplZ[0] not in loz:
                            loz.append(rplZ[0])
                            loz.sort()
                            upz.insert(loz.index(rplZ[0])-1, rplZ[0])
                        if rplZ[1] not in upz:
                            upz.append(rplZ[1])
                            upz.sort()
                            loz.insert(upz.index(rplZ[1])+1, rplZ[1])
                        for iz, zc in enumerate(list(zip(loz, upz))):
                            if tuple(rplZ) == zc:
                                break
                        reg.insert(iz, newreg_str)
                        lab.insert(iz, newlab_str)
                        self.NEAxialConfig.cuts[newtype] = AxialCuts(upz, loz, reg, lab)
                else: 
                    # --- define cutsregions of new SA type
                    cuts = cp(self.NEAxialConfig.cuts[atype])
                    # replace region in cuts
                    for ax in cutaxpos:
                        cuts.reg[ax] = newreg_str
                        cuts.labels[ax] = newlab_str
                    # --- update cuts object
                    self.NEAxialConfig.cuts[newtype] = cuts
                    cuts = list(zip(cuts.reg, cuts.labels, cuts.loz, cuts.upz))
                    zr, zl, zw = self.NEAxialConfig.mapFine2Coarse(cuts, self.NEAxialConfig.zcuts)
                    # --- update info for homogenisation
                    self.NEAxialConfig.cutsregions[newtype] = zr
                    self.NEAxialConfig.cutslabels[newtype] = zl
                    self.NEAxialConfig.cutsweights[newtype] = zw

                    regs = []
                    lbls = []
                    regsapp = regs.append
                    lblsapp = lbls.append
                    for k, val in zr.items():
                        # loop over each axial region
                        for iz in range(self.NEAxialConfig.nZ):
                            if k == 'M1':
                                regsapp(val[iz])
                                lblsapp(zl[k][iz])
                            else:
                                mystr = val[iz]
                                mylab = zl[k][iz]
                                if mystr != 0: # mix name
                                    regs[iz] = f'{regs[iz]}+{mystr}'
                                    lbls[iz] = f'{lbls[iz]}+{mylab}'
                    # --- UPDATE REGIONS
                    # make mixture name unique wrt iType and axial coordinates
                    iMix = 1
                    newmix = []  # new mix of different materials
                    newonlymat = []  # material used in mix but not present alone
                    # axial regions are equal except for replacements and new mixes
                    for jReg, r in enumerate(regs): # axial loop
                        if '+' in r:
                            # update counter if mix already exists
                            if r in regs[:jReg]:
                                iMix += 1 
                            # add SAs type
                            newmixname = f'{atype}{iMix}_{r}'
                            if newmixname not in self.NEregions.values():
                                newmix.append(f'{newtype}{iMix}_{r}')
                                self.NEregions[self.nReg+1] = f'{newtype}{iMix}_{r}'
                                self.NElabels[f'{newtype}{iMix}_{r}'] = f'{lbls[jReg]}'
                                newaxregions_str[jReg] = f'{newtype}{iMix}_{r}'
                                newaxregions[jReg] = self.nReg
                        else:
                            if r not in self.NEregions.values():
                                self.NEregions[self.nReg+1] = r
                                self.NElabels[r] = f'{lbls[jReg]}'
                    # --- homogenise
                    if self.NEAxialConfig.homogenised:
                        for temp in self.TfTc:
                            tmp = self.NEMaterialData[temp]  
                            for u0 in newmix:
                                # identify SA type and subregions
                                strsplt = re.split(r"\d_", u0, maxsplit=1)
                                NEty = strsplt[0]
                                names = re.split(r"\+", strsplt[1])
                                # parse weights
                                w = np.zeros((len(names), ))
                                for iM, mixname in enumerate(names):
                                    idz = newaxregions_str.index(u0)
                                    w[iM] = self.NEAxialConfig.cutsweights[NEty][f"M{iM+1}"][idz]
                                # perform homogenisation
                                mat4hom = {}
                                for name in names:
                                    mat4hom[name] = self.NEMaterialData[temp][name]
                                weight4hom = dict(zip(names, w))
                                tmp[u0] = Homogenise(mat4hom, weight4hom, u0)

                # --- update info in object
                if newtype not in self.NEassemblytypes.keys():
                    self.NEassemblytypes.update({nTypes+1: newtype})
                    self.NEassemblylabel.update({nTypes+1: newtype})
                    self.NEAxialConfig.config.update({nTypes+1: newaxregions})
                    self.NEAxialConfig.config_str.update({newtype: newaxregions_str})        
                # --- replace assembly
                if not isinstance(assbly, list):
                    assbly = [assbly]
                self.replaceSA({newtype: assbly}, time, isfren=isfren)

    def perturb(self, prt, time=0, sanitycheck=True, isfren=True,
                action='pert'):
        """

        Perturb material composition.

        Parameters
        ----------
        what : TYPE
            DESCRIPTION.
        howmuch : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if not isinstance(prt, list):
            prt = [prt]

        iP = 0  # perturbation counter
        for p in list(self.NEregions.values()):
            if action in p:
                iP += 1
        if float(time) in self.NEconfig.keys():
            now = float(time)
        else:
            nt = self.NEtime.index(float(time))
            now = self.NEtime[nt-1]
        
        zcoord = self.NEzcoord
        # loop over perturbations
        for prtdict in prt:
            # check perturbation
            if not isinstance(prtdict, dict):
                raise OSError("Replacement object must be a dict with"
                              " `which`, `where` and `howmuch` keys!")
            iP += 1
            # --- dict sanity check
            mandatorykeys = ['where', 'howmuch', 'which', 'what']
            for mk in mandatorykeys:
                if mk not in prtdict.keys():
                    if mk == 'which' and self.dim == 1:
                        prtdict['which'] = [1]
                    elif mk == 'where' and self.dim == 2:
                        prtdict['where'] = None
                    else:
                        if mk == 'where' and 'region' in prtdict.keys():
                            prtdict['where'] = None
                            continue
                        else:
                            raise OSError(f'Mandatoy key {mk} missing in perturbation for t={time} s')
            if 'depgro' not in prtdict.keys():
                prtdict['depgro'] = None

            # arrange which into list of lists according to SA type
            whichlst = {}
            for w in prtdict['which']:
                itype = self.getassemblytype(w, now,
                            isfren=isfren,
                            whichconf="NEconfig")
                if itype not in whichlst.keys():
                    whichlst[itype] = []
                whichlst[itype].append(w)

            z1z2 = prtdict['where']
            howmuch = prtdict['howmuch']
            depgro = prtdict['depgro']
            perturbation = prtdict['what']
            if perturbation != 'density':
                if len(howmuch) != self.nGro:
                    raise OSError(f'The perturbation intensities required \
                                  should be {self.nGro}')

            notfound = True  # to check consistency of "where" arg
            for itype, assbly in whichlst.items(): # loop over each assembly
                atype = self.NEassemblytypes[itype]
                # --- localise region to be perturbed
                if z1z2 is not None:
                    z1z2.sort()
                    incuts = False  # replacement may be in SA cuts
                    for ipos, coord in self.NEzcoord.items(): # check in zcoords
                        if tuple(z1z2) == coord:
                            notfound = False
                            break
                    if notfound: # check in cuts
                        zcuts = list(zip(self.NEAxialConfig.cuts[atype].loz, self.NEAxialConfig.cuts[atype].upz))
                        for ipos, coord in enumerate(zcuts):
                            if tuple(z1z2) == coord:
                                notfound = False
                                incuts = True
                                break
                        if notfound:
                            raise OSError(f"Cannot find axial region in {z1z2} for replacement!")
                    if incuts:
                        oldreg = self.NEAxialConfig.cuts[atype].regs[ipos]
                        zpert = [list(zcuts[ipos])]
                    else:
                        oldreg = self.NEAxialConfig.config_str[atype][ipos]
                        zpert = [list(self.NEzcoord[ipos])]
                else:
                    if self.dim == 2:  # perturb full SA (only in 2D case)
                        oldreg = self.NEassemblytypes[itype]
                    else:
                        # --- parse all axial regions with oldreg
                        oldreg = prtdict['region']
                        izpos = []
                        for i, r in enumerate(self.NEAxialConfig.config_str[atype]):
                            if r == oldreg:
                                izpos.append(i)
                        # ensure region is not also in mix
                        for r in self.NEregions.values():
                            if "+" in r:
                                if oldreg in r:
                                    raise OSError('Cannot perturb region which is both alone and'
                                                  'in mix! Use separate perturbation cards!')                     
                        if izpos == []:  # look in xscuts
                            for i, r in enumerate(self.NEAxialConfig.cuts[atype].regs):
                                if r == oldreg:
                                    izpos.append(i)
                        # get coordinates
                        zpert = []
                        for i in izpos:
                            zpert.append(list(self.NEzcoord[i]))
                # define perturbed region name
                prtreg = f"{oldreg}-{iP}{action}"
                # --- perturb data and assign it
                for temp in self.TfTc:
                    self.NEMaterialData[temp][prtreg] = cp(self.NEMaterialData[temp][oldreg])
                    self.NEMaterialData[temp][prtreg].perturb(perturbation, howmuch, depgro, sanitycheck=sanitycheck)
                # --- add new assemblies
                self.NEregions[self.nReg+1] = prtreg
                self.NElabels[prtreg] = f"{self.NElabels[oldreg]}-{iP}{action}"
                # --- define replacement dict to introduce perturbation
                if self.dim == 2:
                    repl = {atype: assbly}
                    self.replaceSA(repl, time, isfren=isfren)
                else:
                    repl = {"which": [assbly], "with": [prtreg], "where": [zpert]}
                    self.replace(repl, time, isfren=isfren, action=action)

    def translate(self, transconfig, time, isfren=False, action='trans'):
        """
        Replace assemblies with user-defined new or existing type.

        Parameters
        ----------
        transconfig : dict
            Dictionary with details on translation transformation
        isfren : bool, optional
            Flag for FRENETIC numeration. The default is ``False``.

        Returns
        -------
        ``None``

        """
        # check input type
        # check consistency between dz and which
        if len(transconfig['which']) != len(transconfig['dz']):
            raise OSError('Groups of assemblies and number of ' +
                            'translations do not match!')

        if 'dz' not in transconfig.keys():
            raise OSError('dz missing in translate input!')
        if 'which' not in transconfig.keys():
            raise OSError('which missing in translate input!')

        if isinstance(transconfig['which'], str):
            str2int = self.NEassemblytypes.reverse()
            which = self.getassemblylist(str2int[transconfig['which']])
            transconfig['which'] = which

        if float(time) in self.NEconfig.keys():
            now = float(time)
        else:
            nt = self.NEtime.index(float(time))
            now = self.NEtime[nt-1]
        
        iT = 0  # replacement counter
        for v in list(self.NEregions.values()):
            if action in v:
                iT += 1

        for dz, which in zip(transconfig['dz'], transconfig['which']):
            # repeat configuration if dz = 0
            if dz != 0:
                iT += 1
                # arrange which into list of lists according to SA type
                whichlst = {}
                for w in which:
                    itype = self.getassemblytype(w, now,
                                isfren=isfren,
                                whichconf="NEconfig")
                    if itype not in whichlst.keys():
                        whichlst[itype] = []
                    whichlst[itype].append(w)
                for itype, assbly in whichlst.items():
                    atype = self.NEassemblytypes[itype]
                    nTypes = len(self.NEassemblytypes.keys())
                    # --- assign new assembly name
                    if action in atype:
                        regex = rf"\-[0-9]{action}" # TODO test with basename like `IF-1-XXX-1trans`
                        basetype = re.split(regex, atype, maxsplit=1)[0]
                        newtype = f"{basetype}-{iT}{action}"
                        oldtype = atype
                    else:
                        newtype = f"{atype}-{iT}{action}"
                        oldtype = newtype

                    # --- define new cuts
                    if newtype not in self.NEAxialConfig.cuts.keys():
                        # --- operate translation
                        cuts = cp(self.NEAxialConfig.cuts[atype])
                        cuts.upz[0:-1] = [z+dz for z in cuts.upz[0:-1]]
                        cuts.loz[1:] = [z+dz for z in cuts.loz[1:]]
                        self.NEAxialConfig.cuts[newtype] = AxialCuts(cuts.upz, cuts.loz, cuts.reg, cuts.labels)
                        cuts = list(zip(cuts.reg, cuts.labels, cuts.loz, cuts.upz))
                        zr, zl, zw = self.NEAxialConfig.mapFine2Coarse(cuts, self.NEAxialConfig.zcuts)
                        # --- update info for homogenisation
                        self.NEAxialConfig.cutsregions[newtype] = zr
                        self.NEAxialConfig.cutslabels[newtype] = zl
                        self.NEAxialConfig.cutsweights[newtype] = zw

                        regs = []
                        lbls = []
                        regsapp = regs.append
                        lblsapp = lbls.append
                        for k, val in zr.items():
                            # loop over each axial region
                            for iz in range(self.NEAxialConfig.nZ):
                                if k == 'M1':
                                    regsapp(val[iz])
                                    lblsapp(zl[k][iz])
                                else:
                                    mystr = val[iz]
                                    mylab = zl[k][iz]
                                    if mystr != 0: # mix name
                                        regs[iz] = f'{regs[iz]}+{mystr}'
                                        lbls[iz] = f'{lbls[iz]}+{mylab}'
                        # --- update region dict
                        newaxregions = [None]*len(regs)
                        newaxregions_str = [None]*len(regs)
                        # make mixture name unique wrt itype and axial coordinates
                        iMix = 1
                        newmix = []  # new mix of different materials
                        newonlymat = []  # material used in mix but not present alone
                        for jReg, r in enumerate(regs): # axial loop
                            # nMIX = self.nReg
                            if '+' in r:
                                # update counter if mix already exists
                                if r in regs[:jReg]:
                                    iMix += 1
                                # add SAs type
                                newmixname = f'{newtype}{iMix}_{r}'
                                if newmixname not in self.NEregions.values():
                                    newmix.append(f'{newtype}{iMix}_{r}')
                                    self.NEregions[self.nReg+1] = f'{newtype}{iMix}_{r}'
                                    self.NElabels[f'{newtype}{iMix}_{r}'] = f'{lbls[jReg]}'
                                    newaxregions_str[jReg] = f'{newtype}{iMix}_{r}'
                                    newaxregions[jReg] = self.nReg
                                else:
                                    str2int = self.NEregions.reverse()
                                    newaxregions[jReg] = str2int[f'{newtype}_{r}']
                                    newaxregions_str[jReg] = f"{newtype}_{r}"  # or oldtype?
                            else:
                                if r not in self.NEregions.values():
                                    self.NEregions[self.nReg+1] = r
                                    self.NElabels[r] = f'{lbls[jReg]}'
                                    nMIX = self.nReg
                                else:
                                    str2int = self.NEregions.reverse()
                                    nMIX = str2int[r]
                                newaxregions[jReg] = nMIX
                                newaxregions_str[jReg] = r
                        # --- homogenise
                        for temp in self.TfTc:
                            tmp = self.NEMaterialData[temp]  
                            for u0 in newmix:
                                # identify SA type and subregions
                                strsplt = re.split(r"\d_", u0, maxsplit=1)
                                NEty = strsplt[0]
                                names = re.split(r"\+", strsplt[1])
                                # parse weights
                                w = np.zeros((len(names), ))
                                for iM, mixname in enumerate(names):
                                    idz = newaxregions_str.index(u0)
                                    w[iM] = self.NEAxialConfig.cutsweights[NEty][f"M{iM+1}"][idz]
                                # perform homogenisation
                                mat4hom = {}
                                for name in names:
                                    mat4hom[name] = self.NEMaterialData[temp][name]
                                weight4hom = dict(zip(names, w))
                                tmp[u0] = Homogenise(mat4hom, weight4hom, u0)

                    # --- update info in object
                    if newtype not in self.NEassemblytypes.keys():
                        self.NEassemblytypes.update({nTypes+1: newtype})
                        self.NEassemblylabel.update({nTypes+1: newtype})
                        self.NEAxialConfig.config.update({nTypes+1: newaxregions})
                        self.NEAxialConfig.config_str.update({newtype: newaxregions_str})        
                    # --- replace assembly
                    if not isinstance(assbly, list):
                        assbly = [assbly]
                    self.replaceSA({newtype: assbly}, now, isfren=isfren)

    def perturbBC(self, pertconfig, time, isfren=False):
        """
        Spatially perturb cooling zone boundary conditions.

        Parameters
        ----------
        newtype : list
            List of new/existing types of assemblies.
        asslst : list
            List of assemblies to be replaced.
        isfren : bool, optional
            Flag for FRENETIC numeration. The default is ``False``.

        Returns
        -------
        ``None``

        """
        # check input type
        try:
            # check consistency between dz and which
            if len(pertconfig['which']) != len(pertconfig['what']):
                raise OSError('Groups of assemblies and perturbations do' +
                              'not match in TH "boundaryconditions"!')

            if len(pertconfig['with']) != len(pertconfig['what']):
                raise OSError('Each new value in TH "boundaryconditions"' +
                              ' must come with its identifying parameter!')

            pconf = zip(pertconfig['which'], pertconfig['with'],
                        pertconfig['what'])

            newcore = None
            if float(time) in self.CZconfig.keys():
                now = float(time)
            else:
                nt = self.CZtime.index(float(time))
                now = self.CZtime[nt-1]

            p = 0
            for which, withpar, whatpar in pconf:
                p = p + 1
                for assbly in which:
                    nt = self.CZtime.index(float(time))
                    atype = self.getassemblytype(assbly, now,
                                                 isfren=isfren,
                                                 whichconf="CZconfig")
                    what = self.CZassemblytypes[atype]
                    basename = re.split(r"_t\d+.\d+_p\d+", what)[0]
                    newname = "%s_t%s_p%d" % (basename, time, p)

                    # take region name
                    if newname not in self.CZassemblytypes.values():
                        nass = len(self.CZassemblytypes.keys())
                        self.CZassemblytypes[nass + 1] = newname
                        # update values inside parameters
                        self.CZMaterialData.__dict__[whatpar][newname] = withpar

                    # replace assembly
                    if newcore is None:
                        # take previous time-step configuration
                        newcore = self.replace(nass+1, assbly, isfren,
                                               self.CZconfig[now])
                    else:
                        # take "newcore"
                        newcore = self.replace(nass+1, assbly, isfren,
                                               newcore)
            # update cooling zones
            self.CZconfig[float(time)] = newcore

        except KeyError:
            raise OSError('"which" and/or "with" and/or "what" keys missing' +
                          ' in "boundaryconditions" in TH!')

    def writecentermap(self, numbers=True, fname="centermap.txt"):
        """
        Write centermap to text file.

        Parameters
        ----------
        numbers : bool, optional
            Write assembly numbers in the first columns. The default is
            ``True``. If ``False``, the assembly type are written instead.
        fname : str, optional
            Centermap file name. The default is "centermap.txt".

        Returns
        -------
        ``None``

        """
        # define regions
        typelabel = np.reshape(self.Map.type, (self.Map.Nx*self.Map.Ny, 1))
        regions = []
        for key, coord in (self.Map.serpcentermap).items():
            x, y = coord
            if numbers is False:
                key = typelabel[key-1, 0]
            regions.append((key, x, y))

        # write region to external file
        with open(fname, 'w') as f:  # open new file
            f.write("\n".join("{:03d} {:5f} {:5f}".format(elem[0], elem[1],
                                                          elem[2])
                              for elem in regions))
            f.write("\n")

    def getassemblylist(self, atype, time=0, match=True, isfren=False,
                        whichconf="NEconfig"):
        """
        Return assemblies belonging to a certain type.

        Parameters
        ----------
        atype : int
            Desired assembly type.
        match : bool, optional
            If ``True``, it takes the assemblies matching with atype. If
            ``False``, it takes all the others. The default is ``True``.

        Returns
        -------
        matchedass : list
            List of matching/non-matching assemblies.

        """
        if whichconf == "NEconfig":
            asstypes = self.NEconfig[time].flatten(order='C')
        elif whichconf == "THconfig":
            asstypes = self.THconfig[time].flatten(order='C')
        elif whichconf == "CZconfig":
            asstypes = self.CZconfig[time].flatten(order='C')
        else:
            raise OSError("Unknown core config!")

        if match:
            matchedass = np.where(asstypes == atype)[0]+1  # +1 for py indexing
        else:
            matchedass = np.where(asstypes != atype)[0]+1

        if isfren:
            matchedass = [self.Map.serp2fren[m] for m in matchedass]

        matchedass.sort()

        return matchedass
    # TODO: add writeregionmap method to plot region id, x and y for each assembly

    def writecorelattice(self, flatten=False, fname="corelattice.txt",
                         serpheader=False, string=True, whichconf="NEconfig",
                         numbers=False, fren=True, time=0):
        """
        Write core lattice to txt file.

        Parameters
        ----------
        flatten : bool, optional
            Flag to print matrix or flattened array. The default is ``False``.
        fname : str, optional
            File name. The default is "corelattice.txt".
        serpheader : bool, optional
            Serpent 2 code instructions for core lattice geometry
            header. The default is ``False``.
        numbers : bool, optional
            Print assembly numbers instead of assembly names in the core 
            lattice.

        Returns
        -------
        ``None``

        """
        Nx, Ny = self.Map.Nx+0, self.Map.Ny+0
        if whichconf == "NEconfig":
            asstypes = cp(self.NEconfig[time])
            assemblynames = self.NEassemblytypes
        elif whichconf == "THconfig":
            asstypes = cp(self.THconfig[time])
            assemblynames = self.THassemblytypes
        elif whichconf == "CZconfig":
            asstypes = cp(self.CZconfig[time])
            assemblynames = self.CZassemblytypes
        else:
            raise OSError("Unknown core config!")

        if numbers:
            # flatten typelabel matrix
            typemap = np.reshape(asstypes, (Nx*Ny, 1))
            if fren:
                for s, f in self.Map.serp2fren.items():
                    typemap[s-1] = f
            else:
                for s in self.Map.serp2fren.keys():
                    typemap[s-1] = s
            asstypes = np.reshape(typemap, (Nx, Ny), order='F')

        # define regions
        if flatten is False:
            typelabel = asstypes
        else:
            typelabel = np.reshape(asstypes, (Nx*Ny, 1))

        # determine file format
        if string is False:
            # determine number of digits
            nd = str(len(str(self.Map.type.max())))
            fmt = "%0"+nd+"d"
        else:
            fmt = "%s"
            typelabel = typelabel.astype(str)
            for key, val in assemblynames.items():
                typelabel[typelabel == str(key)] = val

            typelabel[typelabel == '0'] = 'VV'

        # determine header
        if serpheader is False:
            header = ""
            comm = '#'
        else:  # Serpent-2 style header (define core lattice)
            # define assembly type according to Serpent
            if self.AssemblyGeom.type == 'S':
                asstype = '1'  # square
            elif self.AssemblyGeom.type == 'H':
                asstype = '3'  # hexagon
            # define central assembly coordinates
            x0, y0 = self.Map.serpcentermap[self.Map.fren2serp[1]][:]
            x0, y0 = str(x0), str(y0)
            # define number of assemblies along x and y directions
            Nx, Ny = str(Nx), str(Ny)
            # define assembly pitch
            P = str(2*self.AssemblyGeom.apothema)
            header = " ".join(("lat core ", asstype, x0, y0, Nx, Ny, P))
            comm = ''
        # save array to file
        if fname is not None:
            np.savetxt(fname, typelabel, delimiter=" ", fmt=fmt, header=header,
                    comments=comm)
        else:
            return typelabel

    @property
    def nReg(self):
        return len(self.NEregions.keys())