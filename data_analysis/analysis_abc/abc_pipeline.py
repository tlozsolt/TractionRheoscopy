# This script list the steps, in order, to go from odsy output to stress strain curves on an entire experiment

# stitchTrack()
# >> make ovito xyz files. 3 files?
# >> dataClean.clean()
# >> gelGlobal tracking?
# >> stress.Stress() (call functio in stress module)
# >> stress.driftCorr_sediment()
# >> stress.gelStrain_boundary()
# then compute strain
#  strain.Strain()
# run strain.Strain.avgStrain()

# now run stress/strain?