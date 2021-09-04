"""
script to generate the following for stress strain analysis

===== Input =====
+ particle positions, cleaned and in right handed rheo coordinates
+ sediment and gel positions with coordinate string for analysis (um, rheo) preferred
  but will work with (um, imageStack)
  - columns should include x,y,z positions
  - dist from gel top and dist sed bottom.
    (in pricnple this should be cleaned up to allow for different behavior depending on coordinate string

---- Figures ----
+ cumulative histogram of index intersection as a function of boolean cut
  parameter for
  - gel fitted planes
  - sed fitted planes
+ heatmap of spatially binned displacements for interface
  - # of sed particles in bin
  - # of gel particle in bin
  - x,y and z displacements of sed averaged over bins (function of time)
  - x,y and z displacements of gel averaged over bins (function of time)
  - difference of x,y,z dipslacements of sed - gel (function of time)
+ sediment displacements averaged over z-bins
+ shear stress vs time for xz yz and zz
+ stress vs strain
+ volume averaged local strain vs time
+ boundary strain vs time
+ two strain measures vs time

---- Data (pickled?, hdf?) ----
+ bin averaged displacements and counts per bin for sed and gel


"""