Chandra ACIS analysis tools and documents
=========================================

Weitian (Aaron) LI, Junhu GU, and Zhenghao ZHU


Introduction
------------
This repository currently contains the following tools:
+ Chandra ACIS data reduction
+ Point source and flares removal
+ Blanksky reprojection
+ Background spectrum correction
+ Source spectra extraction and deprojection analysis (temperature profile)
+ Surface brightness profile extraction
+ Gravitational mass profile calculation (NFW profile extrapolation)
+ Luminosity and flux calculation

These tools are developed to help and automate our batch analysis of the
big galaxy groups and clusters sample observed by Chandra ACIS.
Therefore, there are many assumptions and hacks in these tools, and many
cleanups are needed.  Last but not least, documents are badly needed.

These tools are tested with:
+ CIAO v4.4 (support will be dropped)
+ CIAO v4.6
+ HEASoft v6.12
+ HEASoft v6.16


WARNING
-------
Our Chandra sample has been finished for a period of time, and we at the moment
have no plan to re-process/update the results.
In consequence, these tools/scripts are currently **untested** and very likely have
some **bugs**.


TODO
----
+ drop ``calc_distance`` in favor of ``cosmo_calc``
+ use JSON as the output format (for easier parse and conversion)
+ use python (instead of shell) to manipulate JSON data files
+ add arguments to control the ``cosmo_calc`` output for easier use in scripts
+ integrate the memos/docs for Chandra data analysis
+ integrate the ``chandra_guide`` doc


Installation
------------
1. ``mass_profile``
```
$ cd mass_profile
$ make clean
$ make
(or use this to enable OpenMP)
$ make OPENMP=yes
```

2. ``cosmo_calc``
Get it from repository [atoolbox](https://github.com/liweitianux/atoolbox),
under the directory ``astro/cosmo_calc``.


Settings
--------
Add the following settings to your shell's initialization file
(e.g., ``~/.bashrc`` or ``~/.zshrc``).
```
# Environment variables:
export MASS_PROFILE_DIR="<path>/chandra-acis-analysis/mass_profile"
export CHANDRA_SCRIPT_DIR="<path>/chandra-acis-analysis/scripts"

# Handy aliases:
alias fitmass="${MASS_PROFILE_DIR}/fit_mass.sh"
alias fitnfw="${MASS_PROFILE_DIR}/fit_nfw_mass mass_int.dat"
alias fitsbp="${MASS_PROFILE_DIR}/fit_sbp.sh"
alias fittp="${MASS_PROFILE_DIR}/fit_wang2012_model"
alias calclxfx="${MASS_PROFILE_DIR}/calc_lxfx_simple.sh"
alias getlxfx="${MASS_PROFILE_DIR}/get_lxfx_data.sh"
```


Usage
-----
See the doc ``HOWTO_chandra_acis_analysis.txt``


Useful Links
------------
* [Chandra Chaser](http://cda.harvard.edu/chaser/)
* [NED search by name](http://ned.ipac.caltech.edu/forms/byname.html)
* [SIMBAD](http://simbad.u-strasbg.fr/simbad/)
* [HEASARC nH tool](https://heasarc.gsfc.nasa.gov/cgi-bin/Tools/w3nh/w3nh.pl)


License
-------
* The tools are distributed under the **MIT license** unless otherwise declared.
* The documents are distributed under the ??? License (TBD).
