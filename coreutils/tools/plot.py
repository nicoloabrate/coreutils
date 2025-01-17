"""
Author: N. Abrate.

File: postprocess.py

Description: Class to plot data from Serpent calculations.
"""
import numpy as np
import math
import logging
import shutil as sh
from numbers import Real
from collections import OrderedDict
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import RegularPolygon, Rectangle
from matplotlib.collections import PatchCollection
from serpentTools.utils import formatPlot, normalizerFactory, addColorbar
from matplotlib import rc, rcParams, colors, cm

rcParams['text.usetex']= True if sh.which('latex') else False

mycols1 = ["#19647e", "#28afb0", "#ee964b", # generated with Coloor
           "#ba324f", "#1f3e9e", "#efd28d",
        #    "#004777", 
           "#ffffff", # "#a30000", "#ff7700", # "#175676", "#00afb5",
        #    "#4ba3c3", "#cce6f4", "#d62839", 
        #    "#086788", "#07a0c3", 
           "#f0c808", "#fff1d0", "#dd1c1a",
           "#003049", "#d62828", "#f77f00", "#fcbf49", "#eae2b7",
           "#6C5548", "#B45A38", "#603827", "#13466E", "#F5A964", 
           "#1980A7", "#491F15", "#006094", "#E8743F", "#2AA3C4", "#1f271b", 
           "#f66024", "#104b6d", "#3bbeaa", "#18441b", "#79ac3d", "#e27db1", "white",   # generated with Colorgorical (http://vrl.cs.brown.edu/color)
           "#e99863", "#5b0891", "#3986da", "#8e80fb", "#e3488e", "#02b72e", 
           "#e26df8", "#52351d", "#852405", "#9ca08c", "#2524f9", "#e194be", "#fa1bfc",
           "#851657"]
xkcd = list(colors.XKCD_COLORS.keys())  
mycols1.extend(xkcd)

logger = logging.getLogger(__name__)



def AxialGeomPlot(core, which, time=0, label=False, assembly_name=False,
                  figname=None, fren=False, asstype=False,
                  style='axgeom.mplstyle', splitz=False,
                  color_dict=None,
                  zcuts=False, title=None, scale=1, floating=False,
                  legend=False, seed=None, showhomog=False, **kwargs):
    """
    Plot core geometry on the x-z or y-z plane for NEutronics configuration.

    Parameters
    ----------
    core : :class:`coreutils.core.Core`
        Reactor core object created with "Core" class.
    which : list
        List of int with assemblies to be plotted in the desired numeration
        convention (Serpent, FRENETIC, ...).
    time : float, optional
        Time instant when core configuration is plotted, by default 0.
    whichconf : str, optional
        Configuration to be plotted (NEutronics, THermal-hydraulics or
        Cooling Zones). Defaulit is "NE.config".
    label : bool, optional
        Assembly labels, by default ``False``.
    dictname : dict, optional
        Dictionary with user defined labels, by default ``None``.
    figname : str, optional
        Name to save the figure in .png format, by default ``None``.
    fren : bool, optional
        Boolean for FRENETIC numeration, by default ``False``.
    splitz : bool, optional
        Boolean for plotting user-defined nodal cuts stored in ``Core`` object.
        Default is ``False``.
    zcuts : bool, optional
        Boolean for plotting user-defined cuts stored in ``Core`` object.
        Default is ``None``.
    title : str, optional
        Plot title, by default ``None``.
    scale : float, optional
        Geometry scaling factor, by default 1.
    floating : bool, optional
        Floating object flag. If ``True``, the assemblies are plotted without
        taking into account their position inside the core.
        The default is ``False``.
    legend : bool, optional
        Legend flag. If ``True``, the legend is plotted, by default
        ``False``.
    **kwargs :
        KeyWord optional arguments for plotting.

    Raises
    ------
    ``None``

    Returns
    -------
    ``None``

    """
    if showhomog:
        if not core.NE.AxialConfig.homogenised:
            showhomog = False

    if style == 'axgeom.mplstyle':
        pwd = Path(__file__).parent
        axgesty = str(Path.joinpath(pwd, style))
    else:
        if not Path(style).exists():
            logger.info(f'{style} style sheet not found! \
                            Switching to default...')
        else:
            axgesty = style

    L = core.Geometry.AssemblyGeometry.pitch # edge*2
    # array of assembly type
    NxNy = core.Map.type.size
    config = core.NE.config[time]
    typelabel = np.reshape(config, (NxNy, 1))
    # explore types of assembly to be plotted
    types = []
    for i in which:
        t = core.getassemblytype(i, config, isfren=fren)
        if t not in types:
            types.append(t)

    if showhomog:
        # FIXME FIXME TODO
        reg = list(set(core.NE.AxialConfig.regions.values()))
    else:
        reg = []
        regapp = reg.append
        for NEty, assbly in core.NE.AxialConfig.cuts.items():
            for l in assbly.labels:
                if l not in reg:
                    regapp(l)

    nReg = len(reg)

    # TODO FIXME these lines should be in NE
    if core.NE.plot["AXcolors"] is None:
        core.NE.plot["AXcolors"] = dict(zip(reg, mycols1))
        # assign colors to each assembly axial configuration
        for NEty, ty_dict in core.NE.AxialConfig.cutsregions.items():
            core.NE.AxialConfig.cutscolors[NEty] = {}
            for n, reg_lst in ty_dict.items():
                core.NE.AxialConfig.cutscolors[NEty][n] = [0]*len(reg_lst)
                for i, regcol in enumerate(reg_lst):
                    if regcol != 0:
                        col = core.NE.plot["AXcolors"][regcol]
                        core.NE.AxialConfig.cutscolors[NEty][n][i] = col

    # open figure
    idx = 0
    labels = []
    xlo, xup = np.inf, 0
    ylo, yup = np.inf, 0
    # --- plot with default or user style
    with plt.style.context(axgesty):
        fig, ax = plt.subplots()
        # ax = fig.add_subplot(111)

        kxy = (core.Map.serpcentermap).items()
        if fren:
            which = [core.Map.fren2serp[k] for k in which]
        else:
            tmp = list((core.serpcentermap).keys())
            which = [tmp[k] for k in which]
        xx = []
        for key, coord in kxy:
            # check key is in which list
            if key not in which:
                continue

            # parse axial coordinates
            asstype = core.NE.assemblytypes[core.getassemblytype(key, config)]

            if showhomog:
                loz = np.asarray(core.NE.AxialConfig.zcuts[0:-1])
                upz = np.asarray(core.NE.AxialConfig.zcuts[1:])
                reg = core.NE.AxialConfig.regions
                lab = core.NE.AxialConfig.labels
                zmin = min(loz)
                zmax = max(upz)
                deltaz = upz-loz
            else:
                loz = np.asarray(core.NE.AxialConfig.cuts[asstype].loz)
                upz = np.asarray(core.NE.AxialConfig.cuts[asstype].upz)
                reg = np.asarray(core.NE.AxialConfig.cuts[asstype].reg)
                lab = np.asarray(core.NE.AxialConfig.cuts[asstype].labels)
                zmin = min(loz)
                zmax = max(upz)
                deltaz = upz-loz

            # parse radial coordinates
            L = (upz[-1] - loz[0])/5
            if floating:
                idx = idx + L*2*3/4
                x, y = idx, None
            else:
                x, y = coord

            xx.append(x)

            back_cols = []
            for iz, dz in enumerate(deltaz):
                # scale coordinate
                coord = ((x-L/2)*scale, loz[iz]*scale)
                # update ax limits
                xlo = coord[0] if coord[0] < xlo else xlo
                ylo = coord[1] if coord[1] < ylo else ylo
                xup = coord[0] if coord[0] > xup else xup
                yup = coord[1] if coord[1] > yup else yup
                # define region color and label
                if showhomog:
                    # determine region and resulting color
                    axhomreg = core.NE.AxialConfig.config_str[asstype][iz]
                    col_lst = core.NE.AxialConfig.cutscolors[asstype]
                    # mix colors
                    r, g, b = 0, 0, 0
                    for k in col_lst.keys():
                        col_name = col_lst[k][iz]
                        if col_name != 0:
                            r0, g0, b0 = colors.to_rgb(col_name)
                            w = core.NE.AxialConfig.cutsweights[asstype][k][iz]
                            r += r0*w
                            g += g0*w
                            b += b0*w

                    col = [r, g, b]
                    if ":" in axhomreg:
                        axlabel = axhomreg.split(': ')[1]
                    else:
                        axlabel = axhomreg
                else:
                    if isinstance(core.NE.plot["AXcolors"], dict):
                        axlabel = core.NE.regionslabel[lab[iz]]
                    else:
                        axlabel = lab[iz]

                    col = core.NE.plot['AXcolors'][lab[iz]]
                # background colors used for b/w decorations (dots and lines)
                back_cols.append(col)
                # define assembly patch
                asspatch = Rectangle(coord, L, dz, color=col,
                                    label=axlabel, ec='k', lw=0.25, **kwargs)
                ax.add_patch(asspatch)

            # add my cuts on top of patch
            upz = core.NE.AxialConfig.cuts[asstype].upz
            loz = core.NE.AxialConfig.cuts[asstype].loz
            nZ = len(upz)
            iz = 0
            if zcuts and 'zcuts' in core.NE.AxialConfig.__dict__.keys():
                for myz in core.NE.AxialConfig.zcuts:
                    # parse which color is in background
                    zcol = 'w' if isDark(col) else 'k'
                    plt.hlines(myz*scale, (x-L/2)*scale, (x+L/2)*scale,
                               linestyles='-.', linewidth=0.25, edgecolor=zcol)
                if myz > loz[iz]:
                    iz += 1

            # add node splitting on top of patch
            if splitz and 'AxNodes' in core.NE.AxialConfig.__dict__.keys():
                iz1, iz2 = 0, 0
                izn = 0
                nS = 0
                iNode = 0
                for iCut, z1z2 in core.NE.zcoord.items(): # span het. cuts
                    z1, z2 = z1z2
                    nE = nS+core.NE.AxialConfig.splitz[iCut]
                    for node in core.NE.AxialConfig.AxNodes[nS:nE]:
                        if node < z1 or node > z2:
                            raise OSError("Something is wrong with the nodes!")
                        zn1 = z1 if iNode == 0 else zn2
                        zn2 = zn1+core.NE.AxialConfig.dz[iNode]
                        # --- plot node boundaries
                        # parse which color is in background
                        if zn1 > upz[iz1]:
                            iz1 += 1

                        # define legend label
                        if not showhomog:
                            zcol = 'w' if isDark(core.NE.plot['AXcolors'][lab[iz1]]) else 'k'
                            plt.hlines(zn1*scale, (x-L/2)*scale, (x+L/2)*scale,
                                    linestyles='-', linewidth=0.1, edgecolor=zcol)
                            # parse which color is in background
                            zcol = 'w' if isDark(core.NE.plot['AXcolors'][lab[iz2]]) else 'k'
                            plt.hlines(zn2*scale, (x-L/2)*scale, (x+L/2)*scale,
                                    linestyles='-', linewidth=0.1, edgecolor=zcol)
                            if zn2 > upz[iz2]:
                                iz2 += 1
                        # --- plot node center
                        for izcol in range(izn, len(back_cols)):
                            if node >= loz[izcol] and node <= upz[izcol]:
                                izn = izcol
                                break

                        zcol = 'w' if isDark(back_cols[izcol]) else 'k'
                        plt.scatter(x*scale, node*scale, s=1, c=zcol)
                        iNode += 1

                    nS = nE

            # add assembly name
            if assembly_name:
                x, z = (x*scale, upz[-1]*1.1*scale)
                plt.text(x, z, asstype,
                        ha='center', va='center',
                        color='k', fontsize=8)

        plt.axis('off')
        ax.set_xlim([min(xx)-L/2, max(xx)+L/2])
        ax.set_ylim([zmin-L/2, zmax+L/2])
        plt.tight_layout()
        plt.title(title)
        # add legend, if any
        if legend:
            ncol = 3 if showhomog else 4
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            zoffset = (0.2*np.sign(zmin))
            plt.legend(by_label.values(), by_label.keys(), ncol=ncol,
                       loc="upper center", bbox_transform=ax.transData,
                       bbox_to_anchor=(np.mean(xx), zmin + zoffset),
                       fontsize=8)
        # save figure
        if figname is not None:
            fig.savefig(figname, bbox_inches='tight')


def RadialMap(core, tallies=None, z=0, time=0, pre=0, gro=0, grp=0,
              label=False, figname=None, which=None, fren=False,
              whichconf='NE', asstype=False, dictname=None, colors_dict=None,
              legend=False, fill=True, style='radgeom.mplstyle',
              axes=None, cmap='Spectral_r', thresh=None, fontsize=6,
              cbarfontsize=15, cbarLabel=None, xlabel=None, ylabel=None,
              loglog=None, logx=None, logy=None, title=None,
              scale=1, fmt=None, fmt_cbar=None, txtcol='k', # numbers=False,
              cbar=True, **kwargs):
    """
    Plot something (geometry, input/output data) on the x-y plane.

    Parameters
    ----------
    label : TYPE, optional
        DESCRIPTION, by default False.
    figname : TYPE, optional
        DESCRIPTION, by default None.
    fren : TYPE, optional
        DESCRIPTION, by default False.
    which : TYPE, optional
        DESCRIPTION, by default None.
    what : TYPE, optional
        DESCRIPTION, by default None.
    fill : TYPE, optional
        DESCRIPTION, by default True.
    axes : TYPE, optional
        DESCRIPTION, by default None.
    cmap : TYPE, optional
        DESCRIPTION, by default 'Spectral_r'.
    thresh : TYPE, optional
        DESCRIPTION, by default None.
    cbarLabel : TYPE, optional
        DESCRIPTION, by default None.
    xlabel : TYPE, optional
        DESCRIPTION, by default None.
    ylabel : TYPE, optional
        DESCRIPTION, by default None.
    loglog : TYPE, optional
        DESCRIPTION, by default None.
    logx : TYPE, optional
        DESCRIPTION, by default None.
    logy : TYPE, optional
        DESCRIPTION, by default None.
    title : TYPE, optional
        DESCRIPTION, by default None.
    scale : TYPE, optional
        DESCRIPTION, by default 1.
    fmt : TYPE, optional
        DESCRIPTION, by default "%.2f".
    **kwargs : TYPE
        DESCRIPTION.

    Raises
    ------
    IndexError
        DESCRIPTION.
    TypeError
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if hasattr(core, "FreneticNamelist"):
        isSym = core.FreneticNamelist["PRELIMINARY"]["isSym"]
    else:
        isSym = 0
    # FIXME consider also symmetry for cartesian geometry
    nAss = int((core.nAss-1)/6)*isSym+1 if isSym else core.nAss

    if cbarLabel is None:
        cbarLabel = "data"

    if style == 'radgeom.mplstyle':
        pwd = Path(__file__).parent
        radgesty = str(Path.joinpath(pwd, style))
    else:
        if not Path(style).exists():
            logger.info(f'{style} style sheet not found! \
                        Switching to default...')
        else:
           radgesty = style

    if core.Geometry.AssemblyGeometry.type == "S":
        orientation = np.pi/4
        L = core.Geometry.AssemblyGeometry.edge
        L = L/2*np.sqrt(2)
    elif core.Geometry.AssemblyGeometry.type == "H":
        orientation = 0
        L = core.Geometry.AssemblyGeometry.edge

    if tallies is None:

        NxNy = core.Map.type.size
        if whichconf == "Geometry":
            # FIXME FIXME
            config = core.Geometry.config[time]
        elif whichconf != "NE":
            config = core.TH.__dict__[f"{whichconf}config"][time]
        else:
            config = core.__dict__[whichconf].config[time]
        # array of assembly type
        typelabel = np.reshape(config, (NxNy, 1))
        if whichconf == "Geometry":
            # FIXME FIXME
            coretype = core.Geometry.assemblytypes
        elif whichconf != "NE":
            coretype = core.TH.__dict__[f"{whichconf}assemblytypes"].keys()
        else:
            coretype = core.__dict__[whichconf].assemblytypes.keys() # np.unique(typelabel)  # define
        # color dict
        if colors_dict is not None:
            asscol = colors_dict
            if len(coretype) > len(colors_dict.keys()):
                i_sa = len(coretype) - len(colors_dict.keys())
                for sa_type in coretype[i_sa-1]:
                    colors_dict[sa_type] = mycols1[sa_type]
        else:
            mycols = mycols1
            asscol = dict(zip(coretype, mycols))

    else:
        if len(tallies.shape) > 1:
            raise OSError("tallies must have only one dimension!")

    # check which variable
    amap = core.Map
    if which is None:  # consider all assemblies
        which = list(amap.serpcentermap.keys())

    else:
        if fren:  # FRENETIC numeration to Serpent
            which = [amap.fren2serp[k] for k in which]

    if thresh is None:
        thresh = -np.inf
    elif not isinstance(thresh, (Real, int, float)):
        raise TypeError(
            "thresh should be real, not {}".format(type(thresh)))
    # open figure
    with plt.style.context(radgesty):
        if tallies is not None:
            rcParams.update({'font.size': cbarfontsize})
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        usetex = True if rcParams['text.usetex'] else False
        # if usetex:
        #     rcParams.update({'text.usetex' : False})
        patches, coord, values = [], [], []
        patchesapp = patches.append
        coordapp = coord.append
        valuesapp = values.append
        # loop over SA centers according to Serpent map
        for k, xy in amap.serpcentermap.items():
            if k not in which:
                continue
            if tallies is not None:
                idx = amap.serp2fren[k]-1 if fren else k-1
                if idx >= len(tallies) and isSym:
                    continue
                elif tallies[idx] <= thresh:
                    continue
                else:
                    valuesapp(tallies[idx])
                # plot geometry filled with colour
                asspatch = RegularPolygon(xy, core.Geometry.AssemblyGeometry.numedges,
                                            radius=L*scale, orientation=orientation,
                                            **kwargs)
                coordapp(xy)
                x, y = xy
                # scale coordinate
                xy = (x*scale, y*scale)
                # define assembly patch
                patchesapp(asspatch)
            else:
                # select color
                # idx = amap.serp2fren[k]-1 if fren else 
                col = asscol[typelabel[k-1, 0]]
                # get type
                if whichconf == 'NE':
                    SAslabels = core.NE.assemblylabel
                elif whichconf == 'BC':
                    SAslabels = core.TH.BClabels
                elif whichconf == 'TH':
                    SAslabels = core.TH.THassemblylabels
                elif whichconf == 'Geometry':
                    SAslabels = core.Geometry.assemblytypes

                atype = core.getassemblytype(k, config)
                asspatch = RegularPolygon(xy, core.Geometry.AssemblyGeometry.numedges, radius=L*scale,
                                        orientation=orientation, color=col, ec='k', lw=0.5,
                                        fill=fill, label=SAslabels[atype], **kwargs)
                ax.add_patch(asspatch)

        if tallies is not None:  # plot physics
            coord = np.asarray(coord)
            values = np.asarray(values)
            patches = np.asarray(patches, dtype=object)
            normalizer = normalizerFactory(values, None, False,
                                           coord[:, 0]*scale,
                                           coord[:, 1]*scale)
            pc = PatchCollection(patches, cmap=cmap, ec='k', lw=0.5, **kwargs)

            if title is None:
                if whichconf == 'NE':
                    times = np.array(core.NE.time)
                elif whichconf == 'BC':
                    times = np.array(core.TH.BCtime)
                elif whichconf == 'TH':
                    times = np.array(core.TH.THtime)
                
                idt = np.argmin(abs(time-times))
                if core.dim != 2:
                    nodes = core.NE.AxialConfig.AxNodes if whichconf == 'NE' else core.TH.zcoord
                    idz = np.argmin(abs(z-nodes))
                    title = f'z={nodes[idz]:.2f} [cm], t={time:.2f} [s]'
                else:
                    if len(times) > 1:
                        title = f't={time:.2f} [s]'
                    else:
                        title = None

            formatPlot(ax, loglog=loglog, logx=logx, logy=logy,
                       xlabel=xlabel or "X [cm]",
                       ylabel=ylabel or "Y [cm]", title=title)

            pc.set_array(values)
            pc.set_norm(normalizer)
            ax.add_collection(pc)

            if cbar:
                colorbar = addColorbar(ax, pc, cbarLabel=cbarLabel)
                if fmt_cbar is not None:
                    colorbar.ax.set_yticklabels([fmt % val for val in colorbar.get_ticks()])

            # add labels on top of the polygons
            if label:
                if fmt is None:
                    fmt = "%.1e" if abs(np.max(np.max(tallies))) > 999 else "%.1f"
                else:
                    fmt = fmt

                mapValToCol = cm.ScalarMappable(norm=pc.norm, cmap=cmap)
                for k, coord in (core.Map.serpcentermap).items():
                    # check key is in "which" list
                    idx = amap.serp2fren[k]-1 if fren else k-1
                    if k not in which:
                        continue
                    elif idx >= len(tallies) and isSym:
                        continue
                    elif tallies[idx] <= thresh:
                        continue
                    else:
                        x, y = coord
                        # see https://stackoverflow.com/questions/28752727/map-values-to-colors-in-matplotlib
                        # see https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py
                        # get patch colour
                        col = mapValToCol.to_rgba(tallies[idx])
                        txtcol = 'w' if isDark(col) else 'k'
                        txt = fmt % tallies[idx]
                        plt.text(x*scale, y*scale, txt, ha='center',
                                va='center', color=txtcol, fontsize=fontsize)
        else:
            # add labels on top of the polygons
            for key, coord in (core.Map.serpcentermap).items():
                # check key is in "which" list
                if key not in which:
                    continue

                x, y = coord
                col = asscol[typelabel[key-1, 0]]
                # plot text inside assemblies
                if dictname is None:
                    if label:  # plot assembly number
                        if fren:  # FRENETIC numeration
                            txt = str(core.Map.serp2fren[key])  # translate keys
                        else:
                            txt = str(key)
                        txtcol = 'w' if isDark(col) else 'k'
                        plt.text(x*scale, y*scale, txt, ha='center',
                                va='center', color=txtcol, fontsize=fontsize) # 
                else:
                    if asstype:  # plot assembly type
                        txt = dictname[typelabel[key-1, 0]]
                        txt = txt.split("-")[0]
                        if len(txt) > 3:
                            txt = txt[0:3]
                        txtcol = 'w' if isDark(col) else 'k'
                        plt.text(x*scale, y*scale, txt, ha='center', va='center',
                                 color=txtcol, fontsize=fontsize)

                    # FIXME: must be a better way to do this avoiding asstype, maybe.
                    # change "dictname" because maybe we want tuples to have
                    # overlappings (phytra fig) write other labels
                    else:
                        atype = core.getassemblytype(key, config)
                        x, y = core.Map.serpcentermap[key]
                        try:
                            assk = str(core.Map.serp2fren[key]) if fren else key
                            txt = dictname[int(assk)]
                            txtcol = 'w' if isDark(col) else 'k'
                            plt.text(x*scale, y*scale, txt, ha='center',
                                    va='center', color=txtcol, fontsize=fontsize)
                        except KeyError:
                            continue

        ax.axis('equal')
        if xlabel is None and ylabel is None:
            plt.axis('off')

            if tallies is None:
                # add legend, if any
                if legend:
                    if usetex:
                        rcParams.update({'text.usetex' : True})
                    handles, labels = plt.gca().get_legend_handles_labels()
                    by_label = OrderedDict(zip(labels, handles))
                    plt.legend(by_label.values(), by_label.keys(), ncol=4,
                            loc='lower center', bbox_to_anchor=(0.5, -0.2))

        plt.tight_layout()
        # save figure
        if figname is not None:
            fig.savefig(figname)


def SlabPlot(core, time=0, ax=None, xlabel=None, figname=None, ncols=None, style='axgeom.mplstyle'):
    """Plot regions in config (NOT IN CUTS OBJECT)."""
    if style == 'axgeom.mplstyle':
        pwd = Path(__file__).parent
        axgesty = str(Path.joinpath(pwd, style))
    else:
        if not Path(style).exists():
            logger.info(f'{style} style sheet not found! \
                  Switching to default...')
        else:
            axgesty = style
    # --- plot with default or user style
    with plt.style.context(axgesty):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = ax or plt.gca()

        reg = list(set(core.NE.labels.values()))
        nReg = len(reg)
        # color dict
        asscol = dict(zip(reg, mycols1))

        labels = []
        handles = []
        nConf = core.NE.config[time][0]
        which = core.NE.assemblytypes[nConf]
        cuts = core.NE.AxialConfig.config_str[which]
        for i, which in enumerate(cuts):
            col = asscol[core.NE.labels[which]]
            h1 = ax.axvspan(*core.NE.zcoord[i],
                            alpha=0.5, color=col, zorder=1)
            if i == 0: # add left boundary
                ax.axvline(core.NE.zcoord[i][0], color='k', lw=1, zorder=2)

            ax.axvline(core.NE.zcoord[i][1], color='k', lw=1, zorder=2)
            if core.NE.labels[which] not in labels:
                labels.append(core.NE.labels[which])
                handles.append(h1)

        if ncols is None:
            ncols = 4 if len(cuts) > 2 else 2
        # xlabel = xlabel if xlabel is not None else 'z coordinate [cm]'
        # ax.set_xlabel(xlabel)
        # ax.set_xticks(self.layers)
        plt.axis('off')

        ax.set_xlim((core.NE.AxialConfig.zcuts[0], core.NE.AxialConfig.zcuts[-1]))
        plt.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.3),
                ncol=ncols, framealpha=1, shadow=1) # mode="expand", 
        plt.tight_layout()
        # save figure
        if figname is not None:
            plt.savefig(figname)


def isDark(color):
    # taken from https://stackoverflow.com/questions/22603510/is-this-possible-to-detect-a-colour-is-a-light-or-dark-colour
    [r,g,b] = colors.to_rgb(color)
    r, g, b = r*255, g*255, b*255
    hsp = math.sqrt(0.299 * (r * r) + 0.587 * (g * g) + 0.114 * (b * b))
    if (hsp > 127.5): # 
        return False
    else:
        return True
