import os
import json
import git
import socket
import logging
from pathlib import Path
from datetime import datetime
from collections import OrderedDict, UserDict
from numpy import string_, ndarray, array, asarray
from numpy import int8, int16, int32, int64, float16, float32, float64, \
                  complex128, zeros, asarray, bytes_


_float_types = (float, float16, float32, float64, complex, complex128)
_int_types = (int, bool, int8, int16, int32, int64)
_iter_types = (ndarray, list)

# git repo info
repopath = Path(__file__).resolve().parents[2]
repo = git.Repo(repopath)
sha = repo.head.object.hexsha  # commit id

logger = logging.getLogger(__name__)



def write_log_header():
    """Write .log file header with info for reproducibility.

    Parameters
    ----------
    ``None``

    Returns
    -------
    ``None``
    """
    # datetime object containing current date and time
    sep = "".join(['-']*90)
    now = datetime.now()
    mmddyyhh = now.strftime("%B %d, %Y %H:%M:%S")
    with open("coreutils.log", "a") as f:
        f.write(f"Log file generated with python package `COREutils`: \n")
        f.write(f"{sep}")
        f.write(f"HOSTNAME: {socket.gethostname()} \n")
        try:
            f.write(f"USERNAME: {os.getlogin()} \n")
        except OSError:
            f.write(f"USERNAME: unknown \n")
        f.write(f"GIT_REPO_URL: {repo.remotes.origin.url} \n")
        f.write(f"GIT_COMMIT_ID: {sha} \n")
        f.write(f"GIT_BRANCH: {repo.active_branch} \n")
        f.write(f"DDYYMMHH: {mmddyyhh} \n")
        f.write(f"{sep}")


def write_coreutils_msg(msg):
    """Write a message to the log file.

    Parameters
    ----------
    msg: str
        Message written in the .log file.

    Returns
    -------
    ``None``
    """
    sep = "".join(['-']*90)
    with open("coreutils.log", "a") as f:
        # f.write(f"# {sep} \n")
        f.write(f"- {msg}\n")
        # f.write(f"# {sep} \n")


def uncformat(data, std, fmtn=".5e", fmts=None):

    if len(data) != len(std):
        raise ValueError("Data and std dimension mismatch!")

    if std is None:
        try:
            # data are ufloat
            inptup = [(d.n, d.s) for d in data]
        except OSError:
            raise TypeError("If std is not provided, data must be ufloat type")
    else:
        inptup = list(zip(data, std))

    percent = False

    if '%' in fmtn:
        fmtn.replace('%', '')

    if fmts is None:
        fmt = "{:%s}%s{:%s}" % (fmtn, u"\u00B1", fmtn)
    else:
        if '%' in fmts:
            percent = True
            fmts = fmts.replace('%', '')
            fmt = "{:%s}%s{:%s}%%" % (fmtn, u"\u00B1", fmts)
        else:
            fmt = "{:%s}%s{:%s}" % (fmtn, u"\u00B1", fmts)

    out = []
    outapp = out.append
    for tup in inptup:
        d, s = tup
        if percent is True:
            s = s*100
        outapp(fmt.format(d, s))

    return out


def fortranformatter(value, multiplier=None):
    """
    Convert python format to Fortran-wise format.

    Parameters
    ----------
    value : type
        Value to be converted. It could be ``list``, ``int``,
        ``float``, ``str``.

    Returns
    -------
    str
        Output string.

    """
    if isinstance(value, _int_types):
        return f"{value:d}"
    elif isinstance(value, _float_types):
        s = f"{value:.15e}"
        mantissa, exponent = s.split("e")
        mantissa = f"{float(mantissa):.15g}"
        if "." not in mantissa:
            mantissa = f"{mantissa}.0"
        return f"{mantissa}d{exponent}"
    elif isinstance(value, _iter_types):
        val = [fortranformatter(v) for v in value]
        if isinstance(multiplier, _iter_types):
            mul = [fortranformatter(m) for m in multiplier]
            val = [f"{m}*{v}" for m, v in zip(mul, val)]
        string = ",".join(val)
        return f"{string}"
    elif isinstance(value, str):
        return f"'{value}'"
    elif isinstance(value, np.bytes):
        return f"'{value.decode()}'"
    else:
        raise OSError(f"Cannot convert data of type {type(value)}!")


class MyDict(OrderedDict):

    def reverse(self):
        if len(set(self.values())) != len(self.values()):
            raise OSError("Cannot reverse dict, values not unique!")
        return dict(zip(self.values(), self.keys()))


def lowcasedict(inpdict):
    """Convert a dict into an equivalent dict with lower-case keys.

    Parameters
    ----------
    inpdict : dict
        Input dictionary

    Raises
    ------
    OSError:
        Cannot use lowcasedict, keys cannot become case insensitive!
    """
    lowcasekeys = [k.lower() for k in inpdict.keys()]
    if len(set(lowcasekeys)) != len(inpdict.keys()):
        raise OSError("Cannot use lowcasedict, keys cannot become case insensitive!")
    else:
        return dict(zip(lowcasekeys, inpdict.values())) 


def uppcasedict(inpdict):
    """Convert a dict into an equivalent dict with upper-case keys.

    Parameters
    ----------
    inpdict : dict
        Input dictionary

    Raises
    ------
    OSError:
        Cannot use UppCaseDict, keys cannot become case insensitive!
    """
    uppcasekeys = [k.upper() for k in inpdict.keys()]
    if len(set(uppcasekeys)) != len(inpdict.keys()):
        raise OSError("Cannot use uppcasedict, keys cannot become case insensitive!")
    else:
        return dict(zip(uppcasekeys, inpdict.values())) 
