% This is a script that handles the procedure for
% doing particle locating using the Kilfoil location software, but with the
% addition of using a technique of iterative residual locating in order to
% find particles that would have otherwise been missed.
%
% In choosing the parameters for particle locating, it's best to err on the
% side of missing particles on a given locating pass rather than risk
% double-counting a particle. These missed particles will be found during
% later iterations. Parameters should already be in the workspace before
% this script is called (see also ...input_parameters.m). Also, the
% location software should be in Matlab's known path structure.
%
% For a more detailed description of the method of locating using iterative
% residuals, see Katharine Jensen's Ph.D. thesis. Or just ask Kate.
%
% Important note:
% The method of iterative residuals is, in theory, prone to creating double-hits,
% because the individual iterations don't check for double counts against
% each other. However, KEJ finds that this doesn't actually happen that much.
% However, I still recommend checking for doubly-located particles before 
% doing any further processing!
%
% Kate Jensen - cleaned up and prepared for distribution July 2013


%% Preliminaries

set(0,'DefaultFigureWindowStyle','docked') %this will cause all figures to be docked (KEJ prefers)

display('Checking whether the required parameters are defined, and saving a record of what parameters were used.')

%if needed, generate the output filename
if ~exist('output_filename','var')
    %create an output filename based on the image filename after removing
    %the image file extension
    output_filename = [image_filename(1:find(image_filename == '.',1,'last')-1) '_xyz_coordinates.txt'];
end

%% Load in the raw data image stack:

display(['Image processing and particle locating started at ' datestr(now,'HH:MM:ss') ' on ' datestr(now, 'mm-DD-YYYY') '.'])

display('reading in files...')
% This code is garbage. "range" is vector defined by kate that happens
% to have the same name as a builtin matlab function. Why god why?
%for i=1:1+range(2)-range(1)
%    I = imread([folder '/' image_filename],'Index',range(1)+i-1);
%    raw(:,:,i) = I;
%end
% fix by getting number of z-slices using image info
fName_tiffStack = [folder '/' image_filename];
info_tiffStack = imfinfo(fName_tiffStack);
num_zSlices=numel(info_tiffStack);
for i=1:num_zSlices
    I = imread(fName_tiffStack,i,'Info',info_tiffStack);
    raw(:,:,i) = I;
end

display(['Image loading finished at ' datestr(now,'HH:MM:ss') ' on ' datestr(now, 'mm-DD-YYYY') '.'])

%% Processing step 1: remove high-frequency noise and flatten the image background
%resBPass=bpass3dMB(raw, lnoise, lobject, [0 0]);
res = 1 + 254.0/255.0*raw;

%% Processing step 2: locate the particles

display('feature finding...')
r=feature3dMB(res, diameters, mask_size, size(res), [1 1 0], min_separation, masscut_initial, 0,zeroPxThreshold);
display(['Initial particle locating complete at ' datestr(now,'HH:MM:ss') ' on ' datestr(now, 'mm-DD-YYYY') '. Moving on to find any missed particles by running particle locating iteratively on the image residuals.'])

display('recording first pass coordinates...')

%% Processing step 3: Find any missed particles by iterating the particle locating on the image residuals

% Create new "raw" data by replacing the already-found particles with a
% virtual/false particle of all zeros, thus removing those particles from
% the original image. The remaining image only contains the residuals --
% those particles that were not found by the original particle locating --
% and the [relatively sparse] remaining particles are easy to locate. The
% method iterates until no new particles are found. Typically, for samples
% of about 50,000 particles, the method converges in about 5 iterations.
%
% This method was developed by Nobotomo Nakamura and Katharine Jensen in
% 2011 and early 2012 (somewhat independently, but it was Nobutomo's idea first).


%The false_particle_size parameters must be integers and odd so that the
%virtual particle has a well-defined center.
false_particle_size = round(false_particle_size); %ensure integers
false_particle_size = false_particle_size + ~mod(false_particle_size,2); %ensure odd numbers

%This just makes these variables easier for a person to read below:
a = false_particle_size(1)/2;
b = false_particle_size(2)/2;
c = false_particle_size(3)/2;

%by default, the matrix elements are 1 so that they will leave the image
%unchanged:
false_particle = ones(false_particle_size(1),false_particle_size(2),false_particle_size(3));

%created a virtual particle of zeros:
for i=1:false_particle_size(1)
    for j=1:false_particle_size(2)
        for k=1:false_particle_size(3)
            v = [i j k]-(false_particle_size./2+1/2); %subtract the center coordinates so looking radially out from center
            %Flash back to Algebra II with the equation for an ellipse!
            if ((v(1)/a)^2 + (v(2)/b)^2 + (v(3)/c)^2)  <= 1
                false_particle(i,j,k) = 0; %so will remove data when multiplied in
            end
        end
    end
end

%raw_residuals = double(raw); %make a new "raw" data set while retaining the original images (this may be questionable wrt memory usage)
raw_residuals = double(res); %make a new "raw" data set while retaining the original images (this may be questionable wrt memory usage)

%how many pixels in each direction is the virtual particle matrix?
x_range = -(a-1/2):a-1/2;
y_range = -(b-1/2):b-1/2;
z_range = -(c-1/2):c-1/2;


residual_start_index= 1;
still_searching = 1;
locating_iteration = 2;

while still_searching == 1 %there are still particles to find!
    display(['Beginning particle locating iteration ' num2str(locating_iteration) '...'])
    
    %Use the false_particle matrix to delete the raw date everywhere I found a particle:
    %(this requires careful bookkeeping of matrix indices, especially close to
    %the edges of the image)
    
    for i=residual_start_index:size(r,1)
        this_particle_center = round(r(i,1:3)); %the coordinates are, after all, in pixels!
        
        %figure out the ranges; and remove any array-out-of-bounds problems:
        this_xrange = this_particle_center(1) + x_range;
        invalid_xrange = this_xrange<1 | this_xrange>size(raw,1);
        this_xrange(invalid_xrange) = [];
        
        this_yrange = this_particle_center(2) + y_range;
        invalid_yrange = this_yrange<1 | this_yrange>size(raw,2);
        this_yrange(invalid_yrange) = [];
        
        this_zrange = this_particle_center(3) + z_range;
        invalid_zrange = this_zrange<1 | this_zrange>size(raw,3);
        this_zrange(invalid_zrange) = [];
        
        %remove this already-found particle from the image by multiplying in the virtual particle matrix:
        raw_residuals(this_xrange, this_yrange, this_zrange) = ...
            raw_residuals(this_xrange, this_yrange, this_zrange) .* false_particle(~invalid_xrange,~invalid_yrange,~invalid_zrange); %wipe it out
    end
    
    %if run_interactively
    %    beep; pause(0.2); beep
    %    % look through the new raw to see how they look:
    %    figure; for i=1:1+range(2)-range(1); imagesc(raw_residuals(:,:,i)); axis image; pause(0.1); end
    %    % or go through the images one by one (hit ENTER to go to the next image):
    %    %figure; for i=1:1+range(2)-range(1); imagesc(raw_residuals(:,:,i)); axis image; input(''); end
    %end
    
    
    % Next, locate any particles that remain, using the same settings to filter
    % and locate as before, except for the masscut parameter.
    % Having a different masscut parameter *may* not be strictly necessary, but
    % it seemed to be useful; may be worth exploring further, however. -KEJ 1/28/2013
    
    %display('bandpass filtering the residual raw data...')
    %res_residuals=bpass3dMB(raw_residuals, lnoise, lobject, [0 0]);
    
    %if ~run_interactively
    %    beep; pause(0.2); beep
    %    % look through the new filtered images to see how they look:
    %    %figure; for i=1:1+range(2)-range(1); imagesc(res_residuals(:,:,i)); axis image; pause(0.1); end
    %    
    %    % look through raw residuals with no addition bpass filtering
    %    figure; for i=1:1+range(2)-range(1); imagesc(raw_residuals(:,:,i)); axis image; pause(0.1); end
    %    for i=1:size(raw_residuals,1); imagesc(permute(raw_residuals(i,:,:),[3 2 1])); axis image; pause(0.1); end
    %    % or go through the images one by one (hit ENTER to go to the next image):
    %    %figure; for i=1:1+range(2)-range(1); imagesc(res_residuals(:,:,i)); axis image; input(''); end
    %end
    
    
    display('feature finding additional particles...')
    %r_residuals=feature3dMB(res_residuals, diameters, mask_size, size(res_residuals), [1 1 0], min_separation, masscut_residuals, 0);
    r_residuals=feature3dMB(raw_residuals, diameters, mask_size, size(raw_residuals), [1 1 0], min_separation, masscut_residuals, bridgeThreshold,zeroPxThreshold);
    %r_residuals2=feature3dMB(raw_residuals, diameters, mask_size, size(raw_residuals), [1 1 0], min_separation, 6000, bridgeThreshold);
    
    % can pause here and use show_slice to overlay the particle coordinates on the residuals images to check the locating
    
    if isempty(r_residuals) %we're done; all particles have been found
        still_searching = 0; %this will kick us out of this while loop
    else %there is still more to explore!
        %add the new particles to the master list, increment the locating iteration
        residual_start_index = size(r,1)+1; %where to start blanking out found particles the new "raw" data
        r = [r; r_residuals]; %add the new ones
        locating_iteration = locating_iteration + 1; %so we know how it's going
        %newly-found particles will be removed from the raw_residuals at the start of the next loop
    end
    
end

%% Save the data:
%finally, save out the found particle positions:

display('recording coordinates...')

metaData = makeMetaData(lnoise,image_filename,lobject,diameters,min_separation,masscut_initial,masscut_residuals,false_particle_size,bridgeThreshold,zeroPxThreshold) % str dictionary of key-vale pairs
fileID = fopen(output_filename,'w');
formatSpec = '%s %s\n';
[nrows,ncols] = size(metaData);
for row = 1:nrows
    fprintf(fileID,formatSpec,metaData{row,:})
end

dlmwrite(output_filename,r,'-append','delimiter','\t')
%dlmwrite(output_filename, r, '\t'); % I should add some metadata writing as commented lines

display(['Particle locations saved in file ' output_filename])
display(['Particle locating for ' image_filename ' entirely finished at ' datestr(now,'HH:MM:ss') ' on ' datestr(now, 'mm-DD-YYYY') '.'])
