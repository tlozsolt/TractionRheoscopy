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

run_interactively = 0; %if set to 1, will display figures and stacks for the user to view; if set to 0, nothing will be displayed
if ~exist('run_on_Odyssey','var')
    run_on_Odyssey = 0; %default is that we're running on a local machine, not on the Odyssey server
end
if ~exist('invert_image','var')
    invert_image = 0; %default is NOT to invert the image; this is appropriate for dyed particles; for dye in the fluid phase, invert the image!
end


set(0,'DefaultFigureWindowStyle','docked') %this will cause all figures to be docked (KEJ prefers)

display('Checking whether the required parameters are defined, and saving a record of what parameters were used.')

%if needed, generate the output filename
if ~exist('output_filename','var')
    %create an output filename based on the image filename after removing
    %the image file extension
    output_filename = [image_filename(1:find(image_filename == '.',1,'last')-1) '_xyz_coordinates.txt'];
end

%if needed, set the masscut parameters to default values, and warn:
if ~exist('masscut_initial','var')
    display('Warning: The parameter masscut_initial was not defined. It will be set to a default value, but it is strongly advised that the user define this parameter in the future.')
    masscut_initial = 3e4
end

if ~exist('masscut_residuals','var')
    display('Warning: The parameter masscut_residuals was not defined. It will be set to a default value, but it is strongly advised that the user define this parameter in the future.')
    masscut_residuals = 1e5
end

%save as a .mat file (and check that everything is in place):
%(mark with current date and time so can run multiple locating tests)
if run_on_Odyssey
    save(['locating_parameters.mat'],'range','image_filename','folder','lnoise','lobject','diameters','mask_size','min_separation','masscut_initial','masscut_residuals','false_particle_size','output_filename','invert_image','run_on_Odyssey','bridgeThreshold','zeroPxThreshold')
else
    save(['locating_parameters_' datestr(now,'yymmdd_HHMMSS') '.mat'],'range','image_filename','folder','lnoise','lobject','diameters','mask_size','min_separation','masscut_initial','masscut_residuals','false_particle_size','output_filename','invert_image','run_on_Odyssey')
end
%this save statement will error out if everything isn't properly defined


%% Load in the raw data image stack:

display(['Image processing and particle locating started at ' datestr(now,'HH:MM:ss') ' on ' datestr(now, 'mm-DD-YYYY') '.'])

display('reading in files...')
%load([folder '/' image_filename]); %loading full .mat data-set
%raw = stack; %Rename image (Important)
for i=1:1+range(2)-range(1)
    I = imread([folder '/' image_filename],'Index',range(1)+i-1);
    if ~run_on_Odyssey
        %I = I(1:100,1:100); %for testing a small region only, as when refining the locating parameters on a local machine; automatically switch off on Odyssey
    end
    raw(:,:,i) = I;
end
% for i=1:1+range(2)-range(1)
%     I = imread([folder '/' image_filename],'Index',range(1)+i-1);
%     if ~run_on_Odyssey
%         I = I(150:300,150:300); %for testing a small region only, as when refining the locating parameters on a local machine; automatically switch off on Odyssey
%     end
%     raw(:,:,i) = I;
% end

if invert_image
    raw = 255 - raw; %invert the 8-bit image; use when the fluorescent dye is in the fluid phase, not in the particles
end
% Do a slight compression of the pixel content by just shifting everything
% up 1...zsolt july 25 2017
raw = 1 + 254.0/255.0*raw;

display(['Image loading finished at ' datestr(now,'HH:MM:ss') ' on ' datestr(now, 'mm-DD-YYYY') '.'])

if run_interactively
    beep; pause(0.2); beep
    % at this point, you should look through the images to see how they look; this bit of code will do that for you automatically(comment out later, if you like):
    figure; for i=1:1+range(2)-range(1); imagesc(raw(:,:,i)); axis image; pause(0.1); end
    figure; for i=1:size(raw,1); imagesc(permute(raw(i,:,:),[3 2 1])); axis image; axis xy; pause(0.1); end
    % if you would prefer to go through the images one by one (hitting enter to go to the next image), run this instead:
    %figure; for i=1:1+range(2)-range(1); imagesc(raw(:,:,i)); axis image; input(''); end
end

%remove_uneven_illumination %necessary for Ams data

%% Processing step 1: remove high-frequency noise and flatten the image background

% display('bandpass filtering the raw data...')
% res=bpass3dMB(raw, lnoise, lobject, [0 0]); %the [0,0] means that there are additional options in the function we are not using, so just ignore that
% 
% display(['Image bandpass filtering finished at ' datestr(now,'HH:MM:ss') ' on ' datestr(now, 'mm-DD-YYYY') '.'])
% 
% if run_interactively
%     beep; pause(0.2); beep
%     % at this point, you should look through the filtered images to see how they look compared to the original; this bit of code will do that for you automatically(comment out later, if you like):
%     figure; for i=1:1+range(2)-range(1); subplot(1,2,1); imagesc(raw(:,:,i)); axis image; subplot(1,2,2); imagesc(res(:,:,i)); axis image; pause(0.1); end
%     figure; for i=1:size(raw,1); subplot(1,2,1); imagesc(permute(raw(i,:,:),[3 2 1])); axis image; subplot(1,2,2); imagesc(permute(res(i,:,:),[3 2 1])); axis image; pause(0.1); end   %input(''); end
%     % you are looking for the filtered images to have well-defined bright spots, well-separated from each other.
%     % I recommend iterating these parameters first until the filtered image looks good, then moving on to the particle locating
%     %only the bandpassfiltered Images
%     %figure; for i=1:1+range(2)-range(1); imagesc(res(:,:,i)); axis image; pause(0.1); end
%     %figure; for i=1:size(raw,1); imagesc(permute(res(i,:,:),[3 2 1])); axis image; pause(0.1); end
%     %figure; for i=1:5:1+range(2)-range(1); imagesc(res(:,:,i)); axis image; input(''); end  
% end



%% Processing step 2: locate the particles

display('feature finding...')
r=feature3dMB(raw, diameters, mask_size, size(raw), [1 1 0], min_separation, masscut_initial, 0,zeroPxThreshold); %the [1,0,0] here means we are using only the first optional setting: a minimum separation

% during manual parameter-refining, at the end of this step, use show_slice to overlay the particle coordinates on the original images to check the locating.
save r_firstpass.mat r
display(['Initial particle locating complete at ' datestr(now,'HH:MM:ss') ' on ' datestr(now, 'mm-DD-YYYY') '. Moving on to find any missed particles by running particle locating iteratively on the image residuals.'])

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

res_residuals = double(raw); %make a new "raw" data set while retaining the original images (this may be questionable wrt memory usage)

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
        res_residuals(this_xrange, this_yrange, this_zrange) = ...
            res_residuals(this_xrange, this_yrange, this_zrange) .* false_particle(~invalid_xrange,~invalid_yrange,~invalid_zrange); %wipe it out
    end
    
%     if run_interactively
%         beep; pause(0.2); beep
%         % look through the new raw to see how they look:
%         figure; for i=1:1+range(2)-range(1); imagesc(res_residuals(:,:,i)); axis image; pause(0.1); end
%         % or go through the images one by one (hit ENTER to go to the next image):
%         %figure; for i=1:1+range(2)-range(1); imagesc(raw_residuals(:,:,i)); axis image; input(''); end
%     end
%     
    
    % Next, locate any particles that remain, using the same settings to filter
    % and locate as before, except for the masscut parameter.
    % Having a different masscut parameter *may* not be strictly necessary, but
    % it seemed to be useful; may be worth exploring further, however. -KEJ 1/28/2013
    
    %display('bandpass filtering the residual raw data...')
    %res_residuals=bpass3dMB(raw_residuals, lnoise, lobject, [0 0]);
    
    if run_interactively
        beep; pause(0.2); beep
        % look through the new filtered images to see how they look:
        figure; for i=1:1+range(2)-range(1); imagesc(res_residuals(:,:,i)); axis image; pause(0.1); end
        for i=1:size(res_residuals,1); imagesc(permute(res_residuals(i,:,:),[3 2 1])); axis image; pause(0.1); end
        % or go through the images one by one (hit ENTER to go to the next image):
        %figure; for i=1:1+range(2)-range(1); imagesc(res_residuals(:,:,i)); axis image; input(''); end
    end
    
    
    display('feature finding additional particles...')
    %r_residuals=feature3dMB(res_residuals, diameters, mask_size, size(res_residuals), [1 1 0], min_separation, masscut_residuals, 0);
    r_residuals=feature3dMB(res_residuals, diameters, mask_size, size(res_residuals), [1 1 0], min_separation, masscut_residuals, bridgeThreshold,zeroPxThreshold);
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

dlmwrite(output_filename, r, '\t');

display(['Particle locations saved in file ' output_filename])
display(['Particle locating for ' image_filename ' entirely finished at ' datestr(now,'HH:MM:ss') ' on ' datestr(now, 'mm-DD-YYYY') '.'])
