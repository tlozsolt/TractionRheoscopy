function textArray = makeMetaData(lnoise,image_filename,lobject,diameters,min_separation,masscut_initial,masscut_residuals,false_particle_size,bridgeThreshold, zeroPxThreshold)

textArray = [
    {'###',''},    
    {'#locating paramters',''},
    {'# date:', datestr(now)},    
    {'# lnoise',num2str(lnoise)},
    {'# image_filename', image_filename},
    {'# diameters:',num2str(diameters)},
    {'# min_separation:',num2str(min_separation)}, 
    {'# masscut_initial:',num2str(masscut_initial)},
    {'# masscut_residuals:',num2str(masscut_residuals)},
    {'# false_particle_size:', num2str(false_particle_size)},
    {'# bridgeThreshold:',num2str(bridgeThreshold)},
    {'# zeroPxThreshold:', num2str(zeroPxThreshold)},
    {'#',''}
    ];
end