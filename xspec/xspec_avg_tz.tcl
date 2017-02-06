# Copyright (c) 2017 Weitian LI <liweitianux@live.com>
# MIT license
#
# XSPEC Tcl script to calculate the average temperature and abundance
# by ting the temperatures and abundances of all regions.
#
# NOTE:
# * Use XSPEC model `projct*wabs*apec`
# * if error < 0.01, then assume error = 0.01
# * if fabs(error) < 1e-7, then set error as `NULL' in output
#   (this may be caused by frozen or tied parameters)
#
# Output:
# * tz_average.txt : average temperature and abundance (and errors)
#
# Change logs:
# 2017-02-06, Weitian LI
#   * Initial version based on `xspec_tprofile.tcl`


set NAME "xspec_avg_tz.tcl"
set DATE [ exec date ]
# Flag to indicate invalid errors
set FLAG_ERR "FALSE"
# Flag to indicate too large reduced chisq
set FLAG_CHI "FALSE"

# Return TCL results for XSPEC commands.
set xs_return_result 1

# Keep going until fit converges.
query yes

# Get the value of the specified parameter
proc get_value {pn} {
    set value [ lindex [ tcloutr param $pn ] 0 ]
    return $value
}

# Calculate the error of the specified parameter
proc get_error {pn} {
    global FLAG_ERR
    global FLAG_CHI
    set chisq [ tcloutr stat ]
    set dof [ lindex [ tcloutr dof ] 0 ]
    if {[ expr {$chisq / $dof} ] > 2.0} {
        # reduced chisq too large; use sigma instead
        set err [ tcloutr sigma $pn ]
        set FLAG_CHI "TRUE"
    } else {
        error 1.0 $pn
        tclout error $pn
        scan $xspec_tclout "%f %f" val_l val_u
        set err [ expr {($val_u - $val_l) / 2.0} ]
    }
    # error treatment
    if {abs($err) < 1.0e-7} {
        puts "*** WARNING: invalid error value ***"
        set err "NULL"
        set FLAG_ERR "TRUE"
    } elseif {$err < 0.01} {
        set err 0.01
    }
    return $err
}

# Output file
set tz_fn "tz_average.txt"
if {[ file exists $tz_fn ]} {
    exec mv -fv $tz_fn ${tz_fn}_bak
}
set tz_fd [ open $tz_fn w ]

# Header
puts $tz_fd "# Average temperature and abundance"
puts $tz_fd "#"
puts $tz_fd "# Generated by: ${NAME}"
puts $tz_fd "# Created: ${DATE}"
puts $tz_fd "#"
puts $tz_fd "# Item  Value  Error"

# Number of data group
set datasets [ tcloutr datasets ]

# Untie and thaw the temperature and abundance of the first region
untie 5 6
thaw  5 6

# Tie all other temperature and abundance to that of the first region
for {set i 2} {$i <= ${datasets}} {incr i} {
    # Parameter number of temperature and abundance
    set temp_pn  [ expr {8 * $i - 3} ]
    set abund_pn [ expr {8 * $i - 2} ]
    newpar $temp_pn  = 5
    newpar $abund_pn = 6
}

# Fit the spectra again
fit

# Get the values and errors of average temperature and abundance
set temp_val  [ get_value 5 ]
set temp_err  [ get_error 5 ]
set abund_val [ get_value 6 ]
set abund_err [ get_error 6 ]

# Output the average temperature and abundance
puts "Average temperature: $temp_val, $temp_err"
puts "Average abundance: $abund_val, $abund_err"
puts $tz_fd "temp    $temp_val    $temp_err"
puts $tz_fd "abund   $abund_val    $abund_err"

close $tz_fd

# Print a WARNING if there are any errors
if {[ string equal $FLAG_ERR "TRUE" ]} {
    puts "*** WARNING: there are invalid error values ***"
}
if {[ string equal $FLAG_CHI "TRUE" ]} {
    puts "*** WARNING: reduced chi^2 too large to calculate errors ***"
}
