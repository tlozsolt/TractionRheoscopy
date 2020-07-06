% This is a script that handles the input parameters for
% doing particle locating using the Kilfoil location software, but with the
% addition of using a technique of iterative residual locating in order to
% find particles that would have otherwise been missed.
%
% In choosing the parameters for particle locating, it's best to err on the
% side of missing particles on a given locating pass rather than risk
% double-counting a particle. These missed particles will be found during
% later iterations. All of the parameters should be set here (or be set
% manually in the workspace), and the location software should be in
% Matlab's known path structure.
%
% For a more detailed description of the method of locating using iterative
% residuals, see Katharine Jensen's thesis (May 2013). Or just ask Kate.
%
% Kate Jensen - cleaned up and prepared for distribution January 28, 2013

function locating_input_parameters(timestep)

run_on_Odyssey = 1; %a simple toggle for directory path structure, etc. differences between my local computer and the Odyssey sever
invert_image = 0; %whether the image will be inverted before processing; set to 1 when the dye is in the fluid phase, not in the particles
run_interactively = 0;

%% Ensure that the locating software is on the current Matlab path structure:
if run_on_Odyssey
%addpath '/n/regal/spaepen_lab/redston/Sanne_111templated/Kilfoil_locating_3D_feature_finding_algorithms/' %odyssey
%addpath '/n/regal/spaepen_lab/szvanloenen/masterproject/programs/Kilfoil_locating_3D_feature_finding_algorithms' %odyssey
addpath '/n/home04/jzterdik/SCRIPTS/OdysseyParticleLocating/' % add the home directory in odyssey 
addpath '/n/regal/spaepen_lab/zsolt/DATA/' % add path to images on scratch directory
else
addpath 'D:/Masterproject/Data_analyse/2014_09_11/After_overnight_measurement/particle_locating_search/Kilfoil_locating_3D_feature_finding_algorithms'
end

%% User-defined settings for image loading and data saving:

%the range of z-levels (or slices) that you want to load in:
if run_on_Odyssey
range = [1 625] %for z-slices [0:409] (note that the Leica confocal uses zero-indexing, but Matlab does not)
else
range = [100 200]; %optionally, limit the stack height for faster local processing while optimizing the locating parameters
end

%can manually define the file name, or assemble it automatically:
%image_filename = '20140911_timeseriesafter_position2_s001_CE.tif';
%image_filename = '20140911_timeseriesafter_position2_s001_smoothCE.tif';


if ~exist('image_filename','var') % 'var' is second optional arguement sepcificying the type as variable in the workspace
    
    %the components of the filename of the contrast-enhanced 3D tiff:
    %(this is useful when this script will be called automatically by a
    %higher-level script that sets the time step for each stack)
    %image_filename_start = 'Result_of_CoreShell_Cy3MM_700nmCore_Jan10_2017_MaxLaser_20180110_94036_gaussBkSubtract60px_8bit_decon_snr10';
    image_filename_start = '';
    
	%OPTIONAL: include the timestep in the filename:
	if run_on_Odyssey
		timestep_string = ['t' sprintf('%04d',timestep)]; %generate the string automatically; set to '%02d' if 2-digit timesteps
	else
		timestep_string = '_t001'; %as a string with the correct number of digits; OR:
	end
    
    image_filename_end = '.tif'; %or just '.tif'; whatever is applicable for your data
    
    %assemble the filename; make timestep_string optional
    if exist('timestep_string','var')
        image_filename = [image_filename_start timestep_string image_filename_end];
    else
        image_filename = [image_filename_start image_filename_end];
    end
end

display(['Image file ' image_filename ' will be used.'])


%OPTIONAL: define the folder where the constrast-enhanced images are saved
%if 'folder' is not defined, the software will look for the images in the
%current directory

if run_on_Odyssey
	%folder = './';
	%folder = '/n/regal/spaepen_lab/redston/Sanne_111templated/111_template_pos4/111_templated_pos4_mat'; %odyssey
    folder = '/n/regal/spaepen_lab/zsolt/DATA/tfrGel09102018b_shearRun09232018b/Colloid_z205_829_deconSNR12_SingleStacks/Colloid_z205_829_singleStacks' %data folder on odyssey, NO trailing '/'
else
	%folder = 'D:/Masterproject/Data_analyse/2014_09_11/After_overnight_measurement/measurements/20140911_timeseriesafter_positions2'; %local
    folder = 'D:/Masterproject/Data_analyse/2014_09_11/After_overnight_measurement/measurements/20140911_timeseriesafter_position2_probeersel';
end

if ~exist('folder','var')
    display('No data folder defined. The images will be read from the current directory.')
    folder = '.';
else
    display(['Data will be read from the directory: ' folder])
end

%OPTIONAL: define a filename for the output data
%(by default, will use the image_filename without the .tif extension)

%output_filename = '20140911_after_position2_xyz_coordinates_search_z100200.txt';

%% User-defined settings for image processing and particle locating:

%the approximate size of the image noise, in pixels (x,y,and z directions):
%lnoise = [0.6 0.6 0.8]; %the example values I had in the program
% input for band pass filtering to reduce poisson noise 
lnoise = [0.9 0.9 0.7];

%the approximate particle diameter in pixels (x,y, and z directions):
% input for band pass filtering
%lobject = [10 10 12]; %the example values I had in the program
lobject = [11 11 9];


%numbers somewhat larger than the largest particle diameter:
% I think diameters is used during llmx3dMB, ie local max finding routine
%diameters = [16 16 18];
diameters = [13 13 11];
%numbers somewhat smaller than the smallest particle diameter:
% I think that mask_size is used to calculate properties such as total
% intensity over the mask size area...with some potentially significant
% effects of rounding and pixelation due to extensive use of floor(*)
% functions. -Zsolt Apr 2017
% mask_size also comes into play during fracShift, so perhaps it limits the
% area over which you shift the particles. 
mask_size = [9 9 7];

%a minimum separation cutoff to help prevent doubly-located particles 
%(set conservatively; this is to avoid a single particle being detected twice)
% This is a bit misleading variable name. Any local max that are closer
% than min_separation will be *merged* and if the center of mass shift by
% more than 1/2? pix the position is discarded...this really should be set close but less than a particle diameter.  
min_separation = [9 9 7];

%masscut parameters determined by looking at the histogram of masses;
%"particles" having less integrated intensity than this cutoff will be ignored
%adjusted based on a histogram of the masses; note that during iterations, the histogram 
%has a long tail due to particle near the edges of the sample, but only the highest 
%"mass" particles are real

masscut_initial = 0; %3.5e4;
masscut_residuals = 3e4;

%if the masscuts are not set, they will be set to default values by the locating
%program (useful if the user needs to run a first-pass of locating in order
%to look at the ma1ss histogram)


%for creating residual images, define the extent of the virtual particle
%that will be used to remove already-found particles from the image data:
%(these should be odd integers; if not, the software will automatically
%round up to the nearest odd integer)
false_particle_size = [17 17 15];

% use threshold as described in the Kilfoil's comments on feature3dMB to
% remove pixel biasing due to faint bridges between particles
% value should be less than 1
% This applies a local (elliposoidal region within diameter) threshold on
% the intensity to improve subpixel accuracy. 
bridgeThreshold = 0.5;

% New parameter added to prevent erroneous locations during iterative
% particle locating. 
% This parameter places threshold on the number of identitically zero pixels
% that can be part of the local max region. The threshold is applied before
% shift the locations to the center
% The proper choice should be correlated with the size of the object and
% used in conjunction with masscut
zeroPxThreshold =500;

%%
run('/n/regal/spaepen_lab/zsolt/SCRIPTS/locating_mat.m');
