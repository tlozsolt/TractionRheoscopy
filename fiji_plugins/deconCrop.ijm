function cropDecon(input, output, fileName){
	open(input + fileName);
    makeRectangle(10, 10, 440, 440);
    run("Crop");
    run("Make Substack...", "  slices=17-183");
    saveAs("Tiff", output+fileName);
    close();
}

inputDir = '/Volumes/TFR/tfrGel10212018A_shearRun10292018f/decon/';
outputDir = '/Volumes/TFR/tfrGel10212018A_shearRun10292018f/decon/deconCrop_ilastikInput/';

setBatchMode(true);
list = getFileList(inputDir);
for (i=0; i < list.length; i++){
	if (endsWith(list[i], '.tif')){
		cropDecon(inputDir,outputDir,list[i]);
		}
	// In future versions, consider more complicated file selection using matches(list[i], "*decon*tif") function, or something to that effect
	// see https://forum.image.sc/t/using-regular-expressions-in-endswith-suffix/9465/2
}
setBatchMode(false);
