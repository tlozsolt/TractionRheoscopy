import numpy as np

# starting from a list of deconvolved tiff stacks and some metaData about that DPL run
# check to see that all the chunks were deconvolved
# if they were, submit a job to read them in and stitch based on metaData information
# also take some choice xy, xz, and yz slices for visualization
# if somethning is missing, output a list of hashvalue that need to be resubmitted.
# and generate and submit a script to resubmit the missing hashvalues.
# if you can stitch them, then after stitching, immediately start particle locating. Otherwise, give up the time.
