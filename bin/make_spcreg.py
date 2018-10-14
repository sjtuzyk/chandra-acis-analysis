#!/usr/bin/env python3
#
# This script generate a series of regions for the extraction of
# radial surface brightness profile (SBP).
#
# Regions geneartion algorithm:
# (TODO)
#
# Author: Zhenghao ZHU
# Created: ???
#
# Change logs:
# 2018-10-10，Yongkai Zhu
#    * Rewrite the code with python3.
# 2017-02-26, Weitian LI
#   * Further simplify arguments handling
#   * Remove test support (-t) for simplification
#   * Rename 'stn' to 'SNR'
#   * Add ds9 view
# v2.0, 2015/06/03, Weitian LI
#   * Copy needed pfiles to current working directory, and
#     set environment variable $PFILES to use these first.
#   * Added missing punlearn
#   * Removed the section of dmlist.par & dmextract.par deletion


import argparse
import subprocess
import logging
import tempfile
import re
import numpy as np

from _context import acispy
from acispy.spectrum import Spectrum
from acispy.manifest import get_manifest
from acispy.ciao import setup_pfiles
from acispy.ciao import run_command


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cal_counts(infile, sky_reg, elow=700, ehigh=7000):
    tool = "dmlist"
    fenergy = "energy=%s:%s" % (elow, ehigh)
    infile = "infile=%s[%s][sky=%s]" % (infile, fenergy, sky_reg)
    fsky = "sky=%s" % sky_reg
    option = "opt=counts"
    args = [infile, option]
    counts = run_command(tool, args)
    counts = int(counts)
    return counts


def extract_spec(infile, outfile, sky_reg):
    tool = "dmextract"
    binpi = 'bin pi'
    infile = "infile=%s[sky=%s][%s]" % (infile, sky_reg, binpi)
    outfile = "outfile=%s" % outfile
    wmap = "wmap=[energy=300:12000][bin tdet=8]"
    clobber = "clobber=yes"
    args = [infile, outfile, wmap, clobber]
    run_command(tool, args)


def get_centroid(reg_centroid):
    f = open(reg_centroid)
    fread = f.read()
    f.close()
    ret_centroid = re.search("\(.*\)", fread)
    x, y = list(map(float,ret_centroid[0][1:-1].split(',')))
    return x, y
def calc_snr(src_spec, bkg_spec, elow=700, ehigh=7000,
             elow_pb=9500, ehigh_pb=12000):
    """
    Calculate the signal-to-noise ratio (SNR) for the source spectrum
    with respect to the background spectrum, with particle background
    renormalization considered.

    Definition
    ----------
    SNR = (flux_src / flux_bkg) * (pbflux_bkg / pbflux_src)

    Parameters
    ----------
    region : str
        Region string within which to calculate the SNR
    evt : str
        Input (source) event file
    bkg : str, optional
        Filename of the background event file or corrected background
        spectrum.
    elow, ehigh : float, optional
        Lower and upper energy limit to calculate the photon counts.
    elow_pb, ehigh_pb : float, optional
        Lower and upper energy limit of the particle background.
    """
    # Calculate SNR
    spec = Spectrum(src_spec)
    spec_bkg = Spectrum(bkg_spec)
    flux = spec.calc_flux(elow=elow, ehigh=ehigh)
    flux_bkg = spec_bkg.calc_flux(elow=elow, ehigh=ehigh)
    pbflux = spec.calc_pb_flux(elow=elow_pb, ehigh=ehigh_pb)
    pbflux_bkg = spec_bkg.calc_pb_flux(elow=elow_pb, ehigh=ehigh_pb)
    snr = (flux / flux_bkg) * (pbflux_bkg / pbflux)
    return snr


def genspecreg(evtfile, bkg_spec, outfile, reg_centroid,
        step=5, cnts_min=2500):
    rout_max = 1500
    snr = 10
    r_cnts_sum = []
    snr_data = []
    spc_reg = []
    centroid_x, centroid_y = get_centroid(reg_centroid)
    i, r_in, r_out, cnts, total_cnts = np.zeros(5, dtype=int)
    while snr > 2:
        r_in = r_out
        i += 1
        r_out = 5 if r_out == 0 else r_out + 1
        tmp_reg = "pie(%s,%s,%s,%s,0,360)"\
                % (centroid_x, centroid_y, r_in, r_out)
        if r_out > rout_max:
            break
        cnts = cal_counts(evtfile, tmp_reg)
        cnts_tmp = total_cnts + cnts
        r_cnts_sum.append([i, r_in, r_out, cnts_tmp])
        while cnts <= cnts_min:
            r_out = r_out + step
            if r_out > rout_max:
                break
            tmp_reg =  "pie(%s,%s,%s,%s,0,360)"\
                    % (centroid_x, centroid_y, r_in, r_out)
            cnts = cal_counts(evtfile, tmp_reg)
            cnts_tmp = total_cnts + cnts
            r_cnts_sum.append([i, r_in, r_out, cnts_tmp])
        tmp_spc = "tmpspc.pi"
        extract_spec(evtfile, tmp_spc, tmp_reg)
        snr = calc_snr(tmp_spc, bkg_spec)
        snr_data.append(str(snr))
        total_cnts = cnts_tmp
        spc_reg.append(tmp_reg)
        print("SNR:%s" % snr)
        print("i:%s, reg:%s" % (i,tmp_reg))
    total_cnts = cnts_tmp - cnts
    snr_dara = snr_data[:-1]
    spc_reg = spc_reg[:-1]
    snp_file = "spc_snr.dat"
    f = open(snp_file, "w")
    f.writelines('\n'.join(snr_data))
    f.close()
    r_cnts_sum = np.array(r_cnts_sum[:-1])
    num_annuluse = r_cnts_sum[-1, 0]
    if num_annuluse < 3:
        print("***WARNING: NOT ENOUGH PHOTONS ***\n")
        print("***TOTAL %s regions ***\n\n" % num_annuluse)
    elif num_annuluse > 6:
        f = open(outfile+'_%sbak' % cnts_min, "w")
        f.writelines('\n'.join(spc_reg))
        f.close()
        cnts_use = int(total_cnts / 6)
        r_in = 0
        spc_reg = []
        cnts = cnts_use
        for i in np.arange(1,7):
            index = r_cnts_sum[:,3] >= cnts_use
            if sum(index) == 0:
                r_out = r_cnts_sum[-1,2]
            else:
                r_out = r_cnts_sum[index,2][0]
                cnts_use = cnts + r_cnts_sum[index,3][0]
            tmp_reg =  "pie(%s, %s, %s, %s, 0, 360)"\
                    % (centroid_x, centroid_y, r_in, r_out)
            print(tmp_reg)
            spc_reg.append(tmp_reg)
            r_in = r_out
        f= open(outfile, "w")
        f.writelines("\n".join(spc_reg))
        f.close()


def main():
    parser = argparse.ArgumentParser(
        description="Calculate the spectral regions")
    parser.add_argument("-i", "--infile", dest="infile",
            help="input evt2_clean")
    parser.add_argument("-b", "--bkg_spec", dest="bkg_spec",
            help="input bkg_spec")
    parser.add_argument("-C", "--clobber", dest="clobber", action="store_true",
            help="overwrite existing file")
    parser.add_argument("-L", "--elow", dest="elow", type=int, default=700,
            help="lower energy limit [eV] (default: 700 [eV])")
    parser.add_argument("-H", "--ehigh", dest="ehigh", type=int, default=7000,
            help="upper energy limit [eV] (default: 7000 [eV])")
    parser.add_argument("-s", "--step", dest="step", type=int, default=5,
            help="calculation step (default:5)")
    parser.add_argument("-o", "--outfile", dest="outfile",
            default="rspec.reg",
            help="output spectral region filename " +
            "(default: rspec.reg)")
    args = parser.parse_args()

    setup_pfiles(["dmstat", "dmlist", "dmextract"])

    manifest = get_manifest()
    logger.info("outfile: %s" % args.outfile)
    if args.infile:
        evtfile = args.infile
    else:
        evtfile = manifest.getpath("evt2_clean", relative=True)
    if args.bkg_spec:
        bkg_spec = args.bkg_spec
    else:
        bkg_spec = manifest.getpath("bkg_spec", relative=True)
    reg_centroid = manifest.getpath("reg_centroid", relative=True)
    genspecreg(evtfile=evtfile, bkg_spec=bkg_spec, outfile=args.outfile,
            reg_centroid=reg_centroid, step=args.step)

    key = "rspec_reg"
    manifest.setpath(key, args.outfile)
    logger.info("Added '%s' to manifest: %s" % (key, manifest.get(key)))


if __name__ == "__main__":
    main()
