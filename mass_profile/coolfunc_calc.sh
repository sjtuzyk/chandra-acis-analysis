#!/bin/sh
#
# unalias -a
#
###########################################################
## Task:                                                 ##
## Calc `cooling function' data according to             ##
## given `temperature profile'                           ##
##                                                       ##
## NOTE:                                                 ##
## given `tprofile': <radius> <temperature>              ##
## calc `cooling function' by invoking `XSPEC'           ##
## using model `wabs*apec'                               ##
##                                                       ##
## LIweitiaNux <liweitianux@gmail.com>                   ##
## August 17, 2012                                       ##
###########################################################

## cmdline arguments {{{
if [ $# -ne 5 ]; then
    printf "usage:\n"
    printf "    `basename $0` <tprofile> <avg_abund> <nH> <redshift> <coolfunc_outfile>\n"
    exit 1
fi
base_path=`dirname $0`
TPROFILE=$1
ABUND_VAL=$2
N_H=$3
REDSHIFT=$4
NORM=`$base_path/calc_distance $REDSHIFT|grep norm|awk '{print $2}'`

echo $NORM


COOLFUNC_DAT=$5
COOLFUNC_DAT_RATIO=flux_cnt_ratio.txt

if [ ! -r "${TPROFILE}" ]; then
    printf "ERROR: given tprofile '${TPROFILE}' NOT accessiable\n"
    exit 2
fi
[ -e "${COOLFUNC_DAT}" ] && rm -f ${COOLFUNC_DAT}
[ -e "${COOLFUNC_DAT_RATIO}" ] && rm -f ${COOLFUNC_DAT_RATIO}
## arguments }}}

## specify variable name outside while loop
## otherwise the inside vars invisible
XSPEC_CF_XCM="_coolfunc_calc.xcm"
[ -e "${XSPEC_CF_XCM}" ] && rm -f ${XSPEC_CF_XCM}

## generate xspec script {{{
cat >> ${XSPEC_CF_XCM} << _EOF_
## XSPEC Tcl script
## calc cooling function data
##
## generated by: `basename $0`
## date: `date`

set xs_return_results 1
set xs_echo_script 0
# set tcl_precision 12
dummyrsp .01 100 4096
## set basic data {{{
set nh ${N_H}
set redshift ${REDSHIFT}
set abund_val ${ABUND_VAL}
set norm ${NORM}
## basic }}}

## xspec related {{{
# debug settings {{{
chatter 0
# debug }}}
query yes
abund grsa
dummyrsp 0.3 11.0 1024
# load model 'wabs*apec' to calc cooling function
model wabs*apec & \${nh} & 1.0 & \${abund_val} & \${redshift} & \${norm} & /*
## xspec }}}

## set input and output filename
set tpro_fn "${TPROFILE}"
set cf_fn "${COOLFUNC_DAT}"
set cff_fn "${COOLFUNC_DAT_RATIO}"
if { [ file exists \${cf_fn} ] } {
    exec rm -fv \${cf_fn}
}

if { [ file exists \${cff_fn} ] } {
    exec rm -fv \${cff_fn}
}

## open files
set tpro_fd [ open \${tpro_fn} r ]
set cf_fd [ open \${cf_fn} w ]
set cff_fd [ open \${cff_fn} w ]

## read data from tprofile line by line
while { [ gets \${tpro_fd} tpro_line ] != -1 } {
    # gets one line
    scan \${tpro_line} "%f %f" radius temp_val
    #puts "radius: \${radius}, temperature: \${temp_val}"
    # set temperature value
    newpar 2 \${temp_val}
    # calc flux & tclout
    flux 0.7 7.0
    tclout flux 1
    scan \${xspec_tclout} "%f %f %f %f" holder holder holder cf_data
    #puts "cf_data: \${cf_data}"
    puts \${cf_fd} "\${radius}    \${cf_data}"
    flux 0.01 100.0
    tclout flux 1
    scan \${xspec_tclout} "%f %f %f %f" cff_data holder holder holder
    puts \${cff_fd} "\${radius}   [expr \${cff_data}/\${cf_data}]"
}

## close opened files
close \${tpro_fd}
close \${cf_fd}

## exit
tclexit
_EOF_

## extract xcm }}}

## invoke xspec to calc
printf "invoking XSPEC to calculate cooling function data ...\n"
# xspec - ${XSPEC_CF_XCM}
xspec - ${XSPEC_CF_XCM} > /dev/null

## clean
# if [ -e "${XSPEC_CF_XCM}" ]; then
#     rm -f ${XSPEC_CF_XCM}
# fi

exit 0
