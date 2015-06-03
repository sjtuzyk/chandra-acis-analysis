#!/bin/sh
##
## This script generate a series of source and background spectra
## for radial spectra profile analysis, and generate two XSPEC
## xcm scripts for ease of use:
## * 'xspec_deproj.xcm': uses 'projct*wabs*apec' model for deprojection
##   spectra analysis to acquire the 3D temperature/abundance profiles.
## * 'xspec_projected.xcm': uses 'wabs*apec' model for projected
##   spectra analysis.
## The input regions may be a series of 'annulus' or 'pie'.
##
## This script invokes 'specextract' to extract the spectra and
## to generate the RMFs and ARFs.
## Also the 'dmgroup' is used to group the spectra.
##
## Background:
## Now 3 types of background format supported:
## * blanksky.fits: will extract the background spectrum from the
##   blanksky.fits of the same region as the source spectrum;
## * lbkg.reg: will use this region file to extract the background from
##   the source evt2 fits file (local background);
## * bkg.pi: will directoryly use the provided spectrum PI file. Just
##   make a copy of this spectrum for each source spectrum, and adjust
##   its 'BACKSCAL' to match the corresponding source spectrum.
## NOTE: the 'BACKSCAL' of the background spectrum will be adjusted for
##       all above 3 conditions.
##
## Reference:
## [1] Chandra spectrum analysis
##     http://cxc.harvard.edu/ciao/threads/extended/
## [2] Ahelp: specextract
##     http://cxc.harvard.edu/ciao/ahelp/specextract.html
## [3] CIAO v4.4 region bugs
##     http://cxc.harvard.edu/ciao/bugs/regions.html#bug-12187
## [4] CIAO 4.6 Release Notes
##     http://cxc.cfa.harvard.edu/ciao/releasenotes/ciao_4.6_release.html
##
AUTHOR="Weitian LI <liweitianux@gmail.com>"
CREATED="2012/07/24"
UPDATED="2015/06/03"
VERSION="v10.0"
##
## ChangeLogs:
## v10.0, 2015/06/03, Aaron LI
##   * Copy needed pfiles to current working directory, and
##     set environment variable $PFILES to use these first.
##   * Replace 'ls' with '\ls'
## v9.0, 2015/02/12, Weitian LI
##   * updated parameter settings for 'specextract' to match
##     specextract revision 2013-12.
##   * updated xcm script generation to use either '.wrmf & .warf'
##     or '.rmf & .arf' extensions. (specextract only use .rmf & .arf
##     extensions for response files since revision 2014-12)
##   * removed 'trap date INT'
##   * removed 'export GREP_OPTIONS=""' and replaced 'grep' with '\grep'
##   * re-arranged descriptions & change logs
## v8.4, 2014/11/13, Weitian LI
##   * replaced 'grppha' with 'dmgroup' to group spectra
##     (dmgroup will add history to fits file, while grppha NOT)
##   * re-arranged change logs
## v8.3, 2014/11/08, Weitian LI
##   * fix problem with 'P_PBKFILE' about the single colon
## v8.2, 2014/07/29, Weitian LI
##   * fix 'pbkfile' parameters for CIAO-4.6
## v8.1, 2014/07/29, Weitian LI
##   * fix variable 'ABUND=grsa'
## v8, 2012/08/14, Weitian LI
##   * use `cmdline' args instead of `cfg file'
##   * add `logging' function
## v7, 2012/08/10, Weitian LI
##   * account blanksky, local bkg, specified bkg
##   * change name to `ciao_deproj_spectra_v*.sh'
##   * add `error status'
##   * Imporve ${CFG_FILE}
##   * Imporve comments
## v6, 2012/08/08, Junhua GU
##   * Modified to using config file to pass parameters
##   * Use grppha to rebin the spectrum
## v5, 2012/08/05
##   * XFLT0005 not modified as pie end angle
##   * added `background renormalization'
##

unalias -a
export LC_COLLATE=C

## usage, help {{{
case "$1" in
    -[hH]*|--[hH]*)
        printf "Usage:\n"
        printf "    `basename $0` evt=<evt2_clean> reg=<radial_reg> bkgd=<blank_evt | lbkg_reg | bkg_spec> basedir=<base_dir> nh=<nH> z=<redshift> [ grouptype=<NUM_CTS|BIN> grouptypeval=<number> binspec=<binspec> log=<log_file> ]\n"
        printf "\nNotes:\n"
        printf "    If grouptype=NUM_CTS, then grouptypeval required.\n"
        printf "    If grouptype=BIN, then binspec required.\n"
        printf "\nVersion:\n"
        printf "    ${VERSION}, ${UPDATED}\n"
        exit ${ERR_USG}
        ;;
esac
## usage, help }}}

## default parameters {{{
# default `event file' which used to match `blanksky' files
#DFT_EVT="_NOT_EXIST_"
DFT_EVT="`\ls evt2*_clean.fits`"
# default `radial region file'
#DFT_REG_IN="_NOT_EXIST_"
DFT_REG_IN="rspec.reg"
# default dir which contains `asols, asol.lis, ...' files
# DFT_BASEDIR=".."
DFT_BASEDIR="_NOT_EXIST_"
# default parameters for 'dmgroup'
DFT_GROUPTYPE="NUM_CTS"
DFT_GROUPTYPEVAL="20"
#DFT_GROUPTYPE="BIN"
DFT_BINSPEC="1:128:2,129:256:4,257:512:8,513:1024:16"
# default `log file'
DFT_LOGFILE="deproj_spectra_`date '+%Y%m%d'`.log"

# default output xspec scripts
XSPEC_DEPROJ="xspec_deproj.xcm"
XSPEC_PROJTD="xspec_projected.xcm"

## howto find files in `basedir'
# default `asol.lis pattern'
DFT_ASOLIS_PAT="acis*asol?.lis"
# default `bad pixel filename pattern'
DFT_BPIX_PAT="acis*repro*bpix?.fits"
# default `pbk file pattern'
DFT_PBK_PAT="acis*pbk?.fits"
# default `msk file pattern'
DFT_MSK_PAT="acis*msk?.fits"

## abundance standard
ABUND="grsa"
## default parameters }}}

## error code {{{
ERR_USG=1
ERR_DIR=11
ERR_EVT=12
ERR_BKG=13
ERR_REG=14
ERR_ASOL=21
ERR_BPIX=22
ERR_PBK=23
ERR_MSK=24
ERR_BKGTY=31
ERR_SPEC=32
ERR_GRPTYPE=41
## error code }}}

## functions {{{
# process commandline arguments
# cmdline arg format: `KEY=VALUE'
getopt_keyval() {
    until [ -z "$1" ]
    do
        key=${1%%=*}                    # extract key
        val=${1#*=}                     # extract value
        keyval="${key}=\"${val}\""
        echo "## getopt: eval '${keyval}'"
        eval ${keyval}
        shift                           # shift, process next one
    done
}

## background renormalization (BACKSCAL) {{{
# renorm background according to particle background
# energy range: 9.5-12.0 keV (channel: 651-822)
CH_LOW=651
CH_HI=822
pb_flux() {
    punlearn dmstat
    COUNTS=`dmstat "$1[channel=${CH_LOW}:${CH_HI}][cols COUNTS]" | \grep -i 'sum:' | awk '{ print $2 }'`
    punlearn dmkeypar
    EXPTIME=`dmkeypar $1 EXPOSURE echo=yes`
    BACK=`dmkeypar $1 BACKSCAL echo=yes`
    # fix `scientific notation' bug for `bc'
    EXPTIME_B=`echo ${EXPTIME} | sed 's/[eE]/\*10\^/' | sed 's/+//'`
    BACK_B=`echo "( ${BACK} )" | sed 's/[eE]/\*10\^/' | sed 's/+//'`
    PB_FLUX=`echo "scale = 16; ${COUNTS} / ${EXPTIME_B} / ${BACK_B}" | bc -l`
    echo ${PB_FLUX}
}

bkg_renorm() {
    # $1: src spectrum, $2: back spectrum
    PBFLUX_SRC=`pb_flux $1`
    PBFLUX_BKG=`pb_flux $2`
    BACK_OLD=`dmkeypar $2 BACKSCAL echo=yes`
    BACK_OLD_B=`echo "( ${BACK_OLD} )" | sed 's/[eE]/\*10\^/' | sed 's/+//'`
    BACK_NEW=`echo "scale = 16; ${BACK_OLD_B} * ${PBFLUX_BKG} / ${PBFLUX_SRC}" | bc -l`
    printf "\`$2': BACKSCAL:\n"
    printf "    ${BACK_OLD} --> ${BACK_NEW}\n"
    punlearn dmhedit
    dmhedit infile=$2 filelist=none operation=add \
        key=BACKSCAL value=${BACK_NEW} comment="old value: ${BACK_OLD}"
}
## bkg renorm }}}
## functions end }}}

## parameters {{{
# process cmdline args using `getopt_keyval'
getopt_keyval "$@"

## check log parameters {{{
if [ ! -z "${log}" ]; then
    LOGFILE="${log}"
else
    LOGFILE=${DFT_LOGFILE}
fi
printf "## use logfile: \`${LOGFILE}'\n"
[ -e "${LOGFILE}" ] && mv -fv ${LOGFILE} ${LOGFILE}_bak
TOLOG="tee -a ${LOGFILE}"
echo "process script: `basename $0`" >> ${LOGFILE}
echo "process date: `date`" >> ${LOGFILE}
## log }}}

# check given parameters
# check evt file
if [ -r "${evt}" ]; then
    EVT=${evt}
elif [ -r "${DFT_EVT}" ]; then
    EVT=${DFT_EVT}
else
    read -p "clean evt2 file: " EVT
    if [ ! -r "${EVT}" ]; then
        printf "ERROR: cannot access given \`${EVT}' evt file\n"
        exit ${ERR_EVT}
    fi
fi
printf "## use evt file: \`${EVT}'\n" | ${TOLOG}
# check given region file(s)
if [ -r "${reg}" ]; then
    REG_IN="${reg}"
elif [ -r "${DFT_REG_IN}" ]; then
    REG_IN=${DFT_REG_IN}
else
    read -p "> radial spec region file: " REG_IN
    if [ ! -r "${REG_IN}" ]; then
        printf "ERROR: cannot access given \`${REG_IN}' region file\n"
        exit ${ERR_REG}
    fi
fi
printf "## use radial reg: \`${REG_IN}'\n" | ${TOLOG}
# check given bkgd, determine background {{{
if [ -z "${bkgd}" ]; then
    read -p "> background (blanksky_evt | lbkg_reg | bkg_spec): " BKGD
else
    BKGD=${bkgd}
fi
if [ ! -r "${BKGD}" ]; then
    printf "ERROR: cannot access given \`${BKGD}'\n"
    exit ${ERR_BKG}
fi
printf "## use bkgd: \`${BKGD}'\n" | ${TOLOG}
# determine bkg type: blanksky, lbkg_reg, bkg_spec ?
# according to file type first: text / FITS
# if FITS, then get values of `HDUCLAS1' and `OBJECT'
if file -bL ${BKGD} | \grep -qi 'text'; then
    printf "## given \`${BKGD}' is a \`text file'\n"
    printf "##   use it as local bkg region file\n"
    printf "##   use *LOCAL BKG SPEC*\n" | ${TOLOG}
    # just set flags, extract spectrum later
    USE_LBKG_REG=YES
    USE_BLANKSKY=NO
    USE_BKG_SPEC=NO
elif file -bL ${BKGD} | \grep -qi 'FITS'; then
    printf "## given \`${BKGD}' is a \`FITS file'\n"
    # get FITS header keyword
    HDUCLAS1=`dmkeypar ${BKGD} HDUCLAS1 echo=yes`
    if [ "${HDUCLAS1}" = "EVENTS" ]; then
        # event file
        printf "##   given file is \`event'\n"
        # check if `blanksky' or `stowed bkg'
        BKG_OBJ=`dmkeypar ${BKGD} OBJECT echo=yes`
        if [ "${BKG_OBJ}" = "BACKGROUND DATASET" ] || [ "${BKG_OBJ}" = "ACIS STOWED" ]; then
            # valid bkg evt file
            printf "##   given FITS file is a valid bkgrnd file\n"
            printf "##   use *BLANKSKY*\n" | ${TOLOG}
            USE_BLANKSKY=YES
            USE_LBKG_REG=NO
            USE_BKG_SPEC=NO
            # specify `BLANKSKY'
            BLANKSKY=${BKGD}
        else
            # invalid bkg evt file
            printf "ERROR: invalid bkg evt file given\n"
            exit ${ERR_BKGTY}
        fi
    elif [ "${HDUCLAS1}" = "SPECTRUM" ]; then
        # spectrum file
        printf "##   given file is \`spectrum'\n"
        printf "##   use *BKG SPECTRUM*\n" | ${TOLOG}
        USE_BKG_SPEC=YES
        USE_BLANKSKY=NO
        USE_LBKG_REG=NO
        # specify `BKG_SPEC'
        BKG_SPEC=${BKGD}
    else
        # other type
        printf "ERROR: other type FITS given\n"
        exit ${ERR_BKGTY}
    fi
else
    printf "ERROR: given \`${BKGD}' type UNKNOWN\n"
    exit ${ERR_BKGTY}
fi
# bkgd }}}
# check given nH
if [ -z "${nh}" ]; then
    read -p "> value of nH: " N_H
else
    N_H=${nh}
fi
printf "## use nH: ${N_H}\n" | ${TOLOG}
# check given redshift
if [ -z "${z}" ]; then
    read -p "> value of redshift: " REDSHIFT
else
    REDSHIFT=${z}
fi
printf "## use redshift: ${REDSHIFT}\n" | ${TOLOG}
# check given dir
if [ -d "${basedir}" ]; then
    BASEDIR=${basedir}
elif [ -d "${DFT_BASEDIR}" ]; then
    BASEDIR=${DFT_BASEDIR}
else
    read -p "> basedir (contains asol files): " BASEDIR
    if [ ! -d "${BASEDIR}" ]; then
        printf "ERROR: given \`${BASEDIR}' NOT a directory\n"
        exit ${ERR_DIR}
    fi
fi
# remove the trailing '/'
BASEDIR=`echo ${BASEDIR} | sed 's/\/*$//'`
printf "## use basedir: \`${BASEDIR}'\n" | ${TOLOG}
# check given dmgroup parameters: grouptype, grouptypeval, binspec
if [ -z "${grouptype}" ]; then
    GROUPTYPE="${DFT_GROUPTYPE}"
elif [ "x${grouptype}" = "xNUM_CTS" ] || [ "x${grouptype}" = "xBIN" ]; then
    GROUPTYPE="${grouptype}"
else
    printf "ERROR: given grouptype \`${grouptype}' invalid.\n"
    exit ${ERR_GRPTYPE}
fi
printf "## use grouptype: \`${GROUPTYPE}'\n" | ${TOLOG}
if [ ! -z "${grouptypeval}" ]; then
    GROUPTYPEVAL="${grouptypeval}"
else
    GROUPTYPEVAL="${DFT_GROUPTYPEVAL}"
fi
printf "## use grouptypeval: \`${GROUPTYPEVAL}'\n" | ${TOLOG}
if [ ! -z "${binspec}" ]; then
    BINSPEC="${binspec}"
else
    BINSPEC="${DFT_BINSPEC}"
fi
printf "## use binspec: \`${BINSPEC}'\n" | ${TOLOG}
# rootname for output files
[ "x${ROOTNAME}" = "x" ] && ROOTNAME="${REG_IN%.reg}"
printf "## use rootname: \`${ROOTNAME}'\n" | ${TOLOG}
## parameters }}}

## check needed files {{{
# check the validity of *pie* regions
printf "check pie reg validity ...\n"
INVALID=`cat ${REG_IN} | \grep -i 'pie' | awk -F, '{ print $6 }' | tr -d ')' | awk '$1 > 360'`
if [ "x${INVALID}" != "x" ]; then
    printf "WARNING: some pie region's END_ANGLE > 360\n" | ${TOLOG}
    printf "    CIAO v4.4 tools may run into trouble\n"
fi

# check files in `basedir'
printf "check needed files in basedir \`${BASEDIR}' ...\n"
# check asolis files
ASOLIS=`\ls -1 ${BASEDIR}/${DFT_ASOLIS_PAT} | head -n 1`
if [ -z "${ASOLIS}" ]; then
    printf "ERROR: cannot find \"${DFT_ASOLIS_PAT}\" in dir \`${BASEDIR}'\n"
    exit ${ERR_ASOL}
fi
printf "## use asolis: \`${ASOLIS}'\n" | ${TOLOG}
# check badpixel file
BPIX=`\ls -1 ${BASEDIR}/${DFT_BPIX_PAT} | head -n 1`
if [ -z "${BPIX}" ]; then
    printf "ERROR: cannot find \"${DFT_BPIX_PAT}\" in dir \`${BASEDIR}'\n"
    exit ${ERR_BPIX}
fi
printf "## use badpixel: \`${BPIX}'\n" | ${TOLOG}
# check pbk file
PBK=`\ls -1 ${BASEDIR}/${DFT_PBK_PAT} | head -n 1`
if [ -z "${PBK}" ]; then
    printf "ERROR: cannot find \"${DFT_PBK_PAT}\" in dir \`${BASEDIR}'\n"
    exit ${ERR_PBK}
fi
printf "## use pbk: \`${PBK}'\n" | ${TOLOG}
# check msk file
MSK=`\ls -1 ${BASEDIR}/${DFT_MSK_PAT} | head -n 1`
if [ -z "${MSK}" ]; then
    printf "ERROR: cannot find \"${DFT_MSK_PAT}\" in dir \`${BASEDIR}'\n"
    exit ${ERR_MSK}
fi
printf "## use msk: \`${MSK}'\n" | ${TOLOG}
## check files }}}

## prepare parameter files (pfiles) {{{
CIAO_TOOLS="dmstat dmkeypar dmhedit specextract dmextract dmgroup"

# Copy necessary pfiles for localized usage
for tool in ${CIAO_TOOLS}; do
    pfile=`paccess ${tool}`
    [ -n "${pfile}" ] && punlearn ${tool} && cp -Lvf ${pfile} .
done

# Modify environment variable 'PFILES' to use local pfiles first
export PFILES="./:${PFILES}"
## pfiles }}}

## process local background {{{
if [ "${USE_LBKG_REG}" = "YES" ]; then
    BKG_EVT=${EVT}
    LBKG_REG=${BKGD}
    LBKG_REG_CIAO="_tmp_${LBKG_REG}"
    cp -fv ${LBKG_REG} ${LBKG_REG_CIAO}
    ## check background region (CIAO v4.4 bug) {{{
    printf "check background region ...\n"
    INVALID=`\grep -i 'pie' ${LBKG_REG_CIAO} | awk -F, '{ print $6 }' | tr -d ')' | awk '$1 > 360'`
    if [ "x${INVALID}" != "x" ]; then
        printf "WARNING: fix for pie region:\n"
        cat ${LBKG_REG_CIAO}
        for angle in ${INVALID}; do
            A_OLD=`echo ${angle} | sed 's/\./\\\./'`
            A_NEW=`echo ${angle}-360 | bc -l | sed 's/\./\\\./'`
            sed -i'' "s/${A_OLD}\ *)/${A_NEW})/" ${LBKG_REG_CIAO}
        done
        printf "    -->"
        cat ${LBKG_REG_CIAO}
        printf "======================\n"
    fi
    ## background region }}}
    ## extract local background spectrum
    printf "extract local background spectrum ...\n"
    BKG_SPEC="${LBKG_REG%.reg}.pi"
    punlearn dmextract
    dmextract infile="${BKG_EVT}[sky=region(${LBKG_REG_CIAO})][bin pi]" \
        outfile=${BKG_SPEC} wmap="[bin det=8]" clobber=yes
    rm -fv ${LBKG_REG_CIAO}
    printf "renormalizing the spectrum later ...\n"
fi
## local bkg }}}

## modify the region file, remove the commented and blank lines {{{
REG_NEW="_new.reg"
REG_TMP="_tmp.reg"
[ -f ${REG_NEW} ] && rm -fv ${REG_NEW}
[ -f ${REG_TMP} ] && rm -fv ${REG_TMP}
cat ${REG_IN} | sed 's/#.*$//' | \grep -Ev '^\s*$' > ${REG_NEW}
## REG_IN }}}

## `specextract' to extract spectrum {{{
LINES="`wc -l ${REG_NEW} | cut -d' ' -f1`"
printf "\n======================================\n"
printf "TOTAL *${LINES}* regions to process ......\n"
for i in `seq ${LINES}`; do
    printf "\n==============================\n"
    printf ">>> PROCESS REGION ${i} ...\n"

    ## generate corresponding `region' file
    rm -f ${REG_TMP} ${REG_CIAO} 2> /dev/null
    head -n ${i} ${REG_NEW} | tail -n 1 > ${REG_TMP}
    REG_CIAO="${REG_TMP%.reg}_ciao.reg"
    cp -fv ${REG_TMP} ${REG_CIAO}
    ## check the validity of *pie* regions {{{
    INVALID=`\grep -i 'pie' ${REG_TMP} | awk -F, '{ print $6 }' | tr -d ')' | awk '$1 > 360'`
    if [ "x${INVALID}" != "x" ]; then
        printf "WARNING: fix for pie region:\n"
        cat ${REG_CIAO}
        A_OLD=`echo ${INVALID} | sed 's/\./\\\./'`
        A_NEW=`echo ${INVALID}-360 | bc -l | sed 's/\./\\\./'`
        sed -i'' "s/${A_OLD}\ *)/${A_NEW})/" ${REG_CIAO}
        printf "    -->\n"
        cat ${REG_CIAO}
    fi
    # check pie region }}}

    ## use `specextract' to extract spectra {{{
    # NOTE: set `binarfwmap=2' to save the time for generating `ARF'
    # I have tested that this bin factor has little impact on the results.
    # NO background response files
    # NO background spectrum (generate by self)
    # NO spectrum grouping (group by self using `dmgroup')
    # Determine parameters for different versions of specextract {{{
    # 'pbkfile' parameter deprecated in CIAO-4.6
    if `pget specextract pbkfile >/dev/null 2>&1`; then
        P_PBKFILE="pbkfile=${PBK}"
    else
        P_PBKFILE=""
    fi
    # specextract: revision 2013-06:
    # 'correct' parameter renamed to 'correctpsf'
    if `pget specextract correct >/dev/null 2>&1`; then
        P_CORRECT="correct=no"
    else
        P_CORRECT="correctpsf=no"
    fi
    # specextract: revision 2013-12:
    # 'weight' parameter controls whether ONLY ARFs are weighted.
    # 'weight_rmf' added to control whether RMFs are weighted.
    # NOTE: 
    #   (1) only when 'weight=yes' will the 'weight_rmf' parameter be used.
    #   (2) no longer distingush between unweighted and weighted reponses
    #       in file extension; only .arf & .rmf are now used.
    if `pget specextract weight_rmf >/dev/null 2>&1`; then
        P_WEIGHTRMF="weight_rmf=yes"
    else
        P_WEIGHTRMF=""
    fi
    # }}}
    punlearn specextract
    specextract infile="${EVT}[sky=region(${REG_CIAO})]" \
        outroot="r${i}_${ROOTNAME}" bkgfile="" asp="@${ASOLIS}" \
        mskfile="${MSK}" badpixfile="${BPIX}" \
        ${P_PBKFILE} ${P_CORRECT} \
        weight=yes ${P_WEIGHTRMF} \
        energy="0.3:11.0:0.01" channel="1:1024:1" \
        bkgresp=no combine=no binarfwmap=2 \
        grouptype=NONE binspec=NONE \
        clobber=yes verbose=2
    ## specextract }}}

    RSPEC_PI="r${i}_${ROOTNAME}.pi"
    RSPEC_BKG_PI="${RSPEC_PI%.pi}_bkg.pi"
    ## background spectrum {{{
    ## generate the blanksky bkg spectrum by self
    if [ "${USE_BLANKSKY}" = "YES" ]; then
        # use blanksky as background file
        printf "extract blanksky bkg spectrum ...\n"
        punlearn dmextract
        dmextract infile="${BLANKSKY}[sky=region(${REG_CIAO})][bin pi]" \
            outfile=${RSPEC_BKG_PI} wmap="[bin det=8]" clobber=yes
    elif [ "${USE_LBKG_REG}" = "YES" ] || [ "${USE_BKG_SPEC}" = "YES" ]; then
        # use *local background* or specified background spectrum
        cp -fv ${BKG_SPEC} ${RSPEC_BKG_PI}
    fi
    ## background }}}

    ## bkg renormalization {{{
    printf "Renormalize background ...\n"
    bkg_renorm ${RSPEC_PI} ${RSPEC_BKG_PI}
    ## bkg renorm }}}

    ## group spectrum {{{
    # use 'dmgroup' instead of 'grppha', because 'dmgroup' will add
    # command history to FITS header (maybe useful for later reference).
    printf "group spectrum \`${RSPEC_PI}' using \`dmgroup'\n"
    RSPEC_GRP_PI="${RSPEC_PI%.pi}_grp.pi"
    punlearn dmgroup
    dmgroup infile="${RSPEC_PI}" outfile="${RSPEC_GRP_PI}" \
        grouptype="${GROUPTYPE}" grouptypeval=${GROUPTYPEVAL} \
        binspec="${BINSPEC}" xcolumn="CHANNEL" ycolumn="COUNTS" \
        clobber=yes
    ## group }}}

    ## `XFLT####' keywords for XSPEC model `projct' {{{
    printf "update file headers ...\n"
    punlearn dmhedit
    if \grep -qi 'pie' ${REG_TMP}; then
        R_OUT="`awk -F',' '{ print $4 }' ${REG_TMP} | tr -d ')'`"
        A_BEGIN="`awk -F',' '{ print $5 }' ${REG_TMP} | tr -d ')'`"
        A_END="`awk -F',' '{ print $6 }' ${REG_TMP} | tr -d ')'`"
        # RSPEC_PI
        dmhedit infile=${RSPEC_PI} filelist=NONE operation=add \
            key="XFLT0001" value=${R_OUT}
        dmhedit infile=${RSPEC_PI} filelist=NONE operation=add \
            key="XFLT0002" value=${R_OUT}
        dmhedit infile=${RSPEC_PI} filelist=NONE operation=add \
            key="XFLT0003" value=0
        dmhedit infile=${RSPEC_PI} filelist=NONE operation=add \
            key="XFLT0004" value=${A_BEGIN}
        dmhedit infile=${RSPEC_PI} filelist=NONE operation=add \
            key="XFLT0005" value=${A_END}
        # RSPEC_GRP_PI
        dmhedit infile=${RSPEC_GRP_PI} filelist=NONE operation=add \
            key="XFLT0001" value=${R_OUT}
        dmhedit infile=${RSPEC_GRP_PI} filelist=NONE operation=add \
            key="XFLT0002" value=${R_OUT}
        dmhedit infile=${RSPEC_GRP_PI} filelist=NONE operation=add \
            key="XFLT0003" value=0
        dmhedit infile=${RSPEC_GRP_PI} filelist=NONE operation=add \
            key="XFLT0004" value=${A_BEGIN}
        dmhedit infile=${RSPEC_GRP_PI} filelist=NONE operation=add \
            key="XFLT0005" value=${A_END}
    elif \grep -qi 'annulus' ${REG_TMP}; then
        R_OUT="`awk -F',' '{ print $4 }' ${REG_TMP} | tr -d ')'`"
        # RSPEC_PI
        dmhedit infile=${RSPEC_PI} filelist=NONE operation=add \
            key="XFLT0001" value=${R_OUT}
        dmhedit infile=${RSPEC_PI} filelist=NONE operation=add \
            key="XFLT0002" value=${R_OUT}
        dmhedit infile=${RSPEC_PI} filelist=NONE operation=add \
            key="XFLT0003" value=0
        # RSPEC_GRP_PI
        dmhedit infile=${RSPEC_GRP_PI} filelist=NONE operation=add \
            key="XFLT0001" value=${R_OUT}
        dmhedit infile=${RSPEC_GRP_PI} filelist=NONE operation=add \
            key="XFLT0002" value=${R_OUT}
        dmhedit infile=${RSPEC_GRP_PI} filelist=NONE operation=add \
            key="XFLT0003" value=0
    else
        printf "*** WARNING: region file NOT MATCH!!\n"
    fi
    ## `XFLT####' }}}

done  # end *for*, `specextract'
## `specextract' }}}

## clean
printf "clean ...\n"
rm -f ${REG_TMP} ${REG_CIAO} 2> /dev/null


###########################################################
## generate a script file for XSPEC                      ##
###########################################################
printf "generate scripts for XSPEC ...\n"
## xspec script (deproj) {{{
printf "XSPEC script for deprojection analysis\n"
[ -e ${XSPEC_DEPROJ} ] && rm -fv ${XSPEC_DEPROJ}

cat >> ${XSPEC_DEPROJ} << _EOF_
#
# XSPEC
# radial spectra (deprojection analysis)
# model projct*wabs*apec
#
# generated by script: \``basename $0`'
# `date`

statistic chi

# load data
_EOF_

for i in `seq ${LINES}`; do
    RSPEC="r${i}_${ROOTNAME}"
    [ -r "${RSPEC}.wrmf" ] && RMF="${RSPEC}.wrmf" || RMF="${RSPEC}.rmf"
    [ -r "${RSPEC}.warf" ] && ARF="${RSPEC}.warf" || ARF="${RSPEC}.arf"
    cat >> ${XSPEC_DEPROJ} << _EOF_
data ${i}:${i} ${RSPEC}_grp.pi
response 1:${i} ${RMF}
arf 1:${i} ${ARF}
backgrnd ${i} ${RSPEC}_bkg.pi

_EOF_
done

cat >> ${XSPEC_DEPROJ} << _EOF_

# filter needed energy range
ignore bad
ignore **:0.0-0.7,7.0-**

method leven 1000 0.01
# change abundance standard
abund ${ABUND}
xsect bcmc
cosmo 70 0 0.73
xset delta 0.01
systematic 0
# auto answer
query yes

# plot related
setplot energy

# model to use
model  projct*wabs*apec
              0
              0
              0
         ${N_H}     -0.001          0          0     100000      1e+06
              1       0.01      0.008      0.008         64         64
            0.5      0.001          0          0          5          5
         ${REDSHIFT}      -0.01     -0.999     -0.999         10         10
              1       0.01          0          0      1e+24      1e+24
_EOF_

INPUT_TIMES=`expr ${LINES} - 1`
for i in `seq ${INPUT_TIMES}`; do
    cat >> ${XSPEC_DEPROJ} << _EOF_
= 1
= 2
= 3
= 4
              1       0.01      0.008      0.008         64         64
            0.5      0.001          0          0          5          5
= 7
              1       0.01          0          0      1e+24      1e+24
_EOF_
done
## xspec script }}}

###########################################################
## xspec script (projected) {{{
printf "XSPEC script for projected analysis\n"
[ -e ${XSPEC_PROJTD} ] && rm -fv ${XSPEC_PROJTD}

cat >> ${XSPEC_PROJTD} << _EOF_
#
# XSPEC
# radial spectra (projected analysis)
# model wabs*apec
#
# generated by script: \``basename $0`'
# `date`

statistic chi

# load data
_EOF_

for i in `seq ${LINES}`; do
    RSPEC="r${i}_${ROOTNAME}"
    [ -r "${RSPEC}.wrmf" ] && RMF="${RSPEC}.wrmf" || RMF="${RSPEC}.rmf"
    [ -r "${RSPEC}.warf" ] && ARF="${RSPEC}.warf" || ARF="${RSPEC}.arf"
    cat >> ${XSPEC_PROJTD} << _EOF_
data ${i}:${i} ${RSPEC}_grp.pi
response 1:${i} ${RMF}
arf 1:${i} ${ARF}
backgrnd ${i} ${RSPEC}_bkg.pi

_EOF_
done

cat >> ${XSPEC_PROJTD} << _EOF_

# filter needed energy range
ignore bad
ignore **:0.0-0.7,7.0-**

method leven 1000 0.01
# change abundance standard
abund ${ABUND}
xsect bcmc
cosmo 70 0 0.73
xset delta 0.01
systematic 0
# auto answer
query yes

# plot related
setplot energy

# model to use
model  wabs*apec
         ${N_H}     -0.001          0          0     100000      1e+06
              1       0.01      0.008      0.008         64         64
            0.5      0.001          0          0          5          5
         ${REDSHIFT}      -0.01     -0.999     -0.999         10         10
              1       0.01          0          0      1e+24      1e+24
_EOF_

INPUT_TIMES=`expr ${LINES} - 1`
for i in `seq ${INPUT_TIMES}`; do
    cat >> ${XSPEC_PROJTD} << _EOF_
= 1
              1       0.01      0.008      0.008         64         64
            0.5      0.001          0          0          5          5
= 4
              1       0.01          0          0      1e+24      1e+24
_EOF_
done

printf "DONE\n"
## xspec script }}}
###########################################################

printf "ALL FINISHED\n"

# vim: set ts=8 sw=4 tw=0 fenc=utf-8 ft=sh: #