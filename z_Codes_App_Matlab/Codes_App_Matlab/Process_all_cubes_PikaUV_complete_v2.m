function Process_all_cubes_PikaUV_complete_v2(mode,stage,white_reflectance, crop, rotate, imageband, meta, binning,smoo)

% This code reads the hyperspectral data from spectral cubes captured with
% Resonon PikaUV and Spectronon software. Please be sure to store in the
% same folder the files '.bil' (or '.bip' or '.bsq'), and their corresponding
% files (with the same name) '.hdr' containing metadata. This script
% processes all full cubes in the selected folder. Giving the option for
% manual crop, rotation and/or vertical or horizontal flip.
% The code assumes that, if captured on reflectance mode (mode = ref)
% the white reference used for the flat field has a known spectral reflectance contained in white_reflectance.
% Use 'teflon_white_reflectance_PikaUV.mat' file for teflon bar. Otherwise,
% if captured in raw or radiance mode, no correction is performed.
% If imageband is set different from 0, a '.png' image file is created with
% the selected bands as grayscale image or as false RGB image.
% The final c1ube is stored as a MATLAB Hypercube object.

% Example usage: Process_all_cubes_PikaUV_complete('ref','linear','Multi_90',1,1,[64 48 23],0,1,0);
%
% For a 3 band display of the resulting cubes type: imshow(cube.DataCube(:,:,[64 48 23]),[])
%
% INPUTS:   mode                [char]      -> 'raw', 'rad', 'ref'. Capturing mode: raw, radiance or reflectance.
%           stage               [char]      -> 'linear': if captured in linear stage.
%                                           -> 'angular': if captured in rotating tripod.
%           white_reflectance   [char]      -> 'Multi_90', 'Multi_50', 'Multi_20', 'Multi_5', 'Sphere', 'Teflon'.
%                               [95 x 1]    ->  Otherwise Spectral reflectance of the reference white tile used for flat field correction.
%                                              when capturing on the linear stage. Only used if stage == 'linear'.
%           crop                [bool]      -> if 'true', the cubes will be cropped.
%           rotate              [bool]      -> if 'true', the cubes will be rotated or flipped.
%           imageband           [integer]   -> if set to integer value different from 0, a '.png' image of the selected band is created in grayscale.
%                               [triplet]   -> if set to a triplet of integer values, a '.png' image of a false RGB is crated with the three selected bands.
%           meta                [bool]      -> if true, the script allows to input and store metadata for each cube. Otherwise it does not.
%           binning             [bool]      -> if true, it assumes spectral binning is on during capture (135 bands). Otherwise assumes no binning during capture (270 bands).
%           smoo                [integer]   -> if 0, no smoothing is applied to the spectra. Otherwise, smoothing is applied with the span value indicated in smoo (recommended value 30).
%
% Author: Miguel Angel Martinez Domingo.    martinezm@ugr.es
% New functionalities: Eva Valero Benito.   valerob@ugr.es
% Adaptation: Ana Belén López Baldomero.
% Last edited: 08-March-2023.
% Color Imaging Laboratory, Department of Optics, University of Granada, Spain.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    program = tic;
    
    % Load cube and metadata files:
    folder_name = uigetdir;
    cd(folder_name);
    d = dir('*.bil');
    d = cat(1,d,dir('*.bip'));
    d = cat(1,d,dir('*.bsq'));
    
    for i=1:length(d)
        wl = set_wl(binning);
        wl_new = (330:5:800)';
        clc, disp(['Processing cube ' num2str(i) ' of ' num2str(length(d)) '...'])
        filename = d(i).name;
        filenametext = [filename '.hdr'];
        fileID = fopen(filenametext);
        for j = 1:2
            tline = fgets(fileID);  % go to line 2
        end
        interleave = tline(14:16);  
        tline = fgets(fileID);      % go to line 3
        data_type = str2double(tline(13:14));
        tline = fgets(fileID);      % go to line 4
        lines = str2double(tline(9:end-2));
        tline = fgets(fileID);      % go to line 5
        samples = str2double(tline(11:end-2));
        tline = fgets(fileID);      % go to line 6
        bands = str2double(tline(9:end-2));
        while ~strcmp(tline(1:7),'shutter')
            tline = fgets(fileID);
        end
        texp = str2double(tline(11:end));
        tline = fgets(fileID);
        while ~strcmp(tline(1:10),'byte order')
            tline = fgets(fileID);
        end
        byte_order = str2double(tline(14));
        tline = fgets(fileID);
        header_offset = str2double(tline(17));
        switch data_type
            case 1
                precision = 'int8';
            case 2
                precision = 'int16';
            case 3
                precision = 'int32';
            case 4
                precision = 'float';
            case 12
                precision = 'uint16';
        end
    
        switch byte_order
            case 0
                bord = 'ieee-le';
            case 1
                bord = 'ieee-be';
        end
        siz = [lines, samples, bands];
        cube = single(multibandread_mike(filename,siz,precision,header_offset,interleave,bord));
        cube = interp_cube(wl,cube,wl_new); % Interpolate cube to 5 nm wavlength range.
        if strcmp(mode,'ref')
            white = set_white(white_reflectance);
            cube = cube .* repmat(reshape(white,1,1,length(white)),size(cube,1),size(cube,2));
        end
        close all;
        if crop == 1
            cube = cropint_PikaUV_cube(cube); % After croping press enter.
        end
        if rotate == 1
            im = cat(3,imadjust(cube(:,:,42)),imadjust(cube(:,:,28)),imadjust(cube(:,:,7)));
            figure, imshow(im,[]);
            clc, disp(['Processing cube ' num2str(i) ' of ' num2str(length(d)) '...'])
            prompt = {'\fontsize{15} Enter rotation angle in degrees (negative angles rotate clockwise):','\fontsize{15} Enter Left-Right flip flag (1 = true or 0 = false):','\fontsize{15} Enter Up-Down flip flag (1 = true or 0 = false)'};
            dlgtitle = 'Rotation / flip data';
            dims = [2 100];
            definput = {'0','0','0'};
            opts.Resize = 'on';
            opts.Interpreter = 'tex';
            answer = inputdlg(prompt,dlgtitle,dims,definput,opts);
            cube = rotate_flip_Lint(cube,str2double(answer{1}),str2double(answer{2}),str2double(answer{3}));
            close all
            if imageband == 0
                [~] = create_maximized_figureint(1);
                imshow(cube(:,:,90),[]);
                title('Final cube to be saved');
                pause(0.5)
            end
        end
        if imageband ~= 0
            if (length(imageband)>1)
                im_cube = cat(3,imadjust(cube(:,:,imageband(1))),imadjust(cube(:,:,imageband(2))),imadjust(cube(:,:,imageband(3))));
            else
                im_cube = imadjust(cube(:,:,imageband(1)));
            end
            [~] = create_maximized_figureint(1);
            imshow(im_cube,[]); title('Final cube to be saved')
            pause(0.5)
            imwrite(im_cube,[filename(1:end-3) 'png']);
        end
        wl = wl_new;
        cube = hypercube(cube,wl);
        if smoo ~= 0
            cube = smooth_hypercube(cube,smoo);
        end
        if meta
            prompt = {'\fontsize{15} Cube number:','\fontsize{15} Cube info:','\fontsize{15} Parent cube:',...
                      '\fontsize{15} Illumination','\fontsize{15} Reference white','\fontsize{15} Date',...
                      '\fontsize{15} Aged (No, Nat, Art):','\fontsize{15} Restored (0 or 1)','\fontsize{15} Substrate:',...
                      '\fontsize{15} Varnish (0 or 1):','\fontsize{15} Varnish type:' };
            dlgtitle = 'Metadata';
            dims = [2 100];
            if i == 1
                definput = {'0','This cube contains ...','0','Hallogen','Teflon','XV','No','0','Paper','0','None'};
            else
                definput = {num2str(Metadata.number),Metadata.cubeinfo,Metadata.parent_cube,Metadata.illumination,...
                            Metadata.reference_white,Metadata.date,Metadata.aged,num2str(Metadata.restored),...
                            Metadata.substrate,num2str(Metadata.varnish),Metadata.varnish_type};
            end
            opts.Resize = 'on';
            opts.Interpreter = 'tex';
            answer = inputdlg(prompt,dlgtitle,dims,definput,opts);
            
            Metadata.number = str2double(answer{1});
            Metadata.name = filename(1:end-4);
            Metadata.wl = [false; true; false];
            Metadata.cubeinfo = answer{2};
            Metadata.RGB = im_cube;
            Metadata.GT = [];
            Metadata.GTLabels = [];
            Metadata.pixels_averaged = [];
            Metadata.spectra_mean = [];
            Metadata.spectra_std = [];
            Metadata.parent_cube = answer{3};
            Metadata.position = [];
            Metadata.texp = texp;
            Metadata.device = 'PikaUV';
            Metadata.stage = stage;
            Metadata.illumination = answer{4};
            Metadata.reference_white = answer{5};
            Metadata.height = size(cube.DataCube,1);
            Metadata.width = size(cube.DataCube,2);
            Metadata.bands = size(cube.DataCube,3);
            Metadata.date = answer{6};
            Metadata.aged = answer{7};
            Metadata.restored = str2double(answer{8});
            Metadata.substrate = answer{9};
            Metadata.varnish = str2double(answer{10});
            Metadata.varnish_type = answer{11};
            save([filename(1:end-3) 'mat'],'cube','wl','Metadata','-v7.3')
        else
            save([filename(1:end-3) 'mat'],'cube','wl','texp','stage','mode','-v7.3')
        end
    end

    fclose('all');
    
    %% Finish:
    clc
    disp('Finished!')
    disp(['Total processing time = ' secs2hms(toc(program)) '.']); clear program
    make_a_beep(0.15,400);
    close all
end

%% Now the auxiliary functions are delcared:

function cube = interp_cube(wl,cube,wl_new)
    [r,c,b] = size(cube);
    cube = reshape(cube,r*c,b)';
    
    cube = interp1(wl,cube,wl_new);
    cube = reshape(cube',r,c,length(wl_new));
end
%*****************************************************
function croped_cube = cropint_PikaUV_cube(cube)
    isok = false;
    while ~isok
        a = get(0,'MonitorPositions');
        f = create_maximized_figureint(size(a,1));
        subplot(1,2,1)
        imshow(cat(3,imadjust(cube(:,:,56)),imadjust(cube(:,:,42)),imadjust(cube(:,:,21))),[])
        title('Full cube (press enter after cropping)')
        
        h = imrect(gca);
        pos = round(getPosition(h));

        if (pos(2)>0) && (pos(1)>0) && ((pos(2)+pos(4)-1)<size(cube,1)) && ((pos(1)+pos(3)-1)<size(cube,2))
            croped_cube = cube(pos(2):pos(2)+pos(4)-1,pos(1):pos(1)+pos(3)-1,:);
            subplot(1,2,2)
            imshow(cat(3,imadjust(croped_cube(:,:,56)),imadjust(croped_cube(:,:,42)),imadjust(croped_cube(:,:,21))),[])
            title('Croped cube')
    
            opts.Default = 'Yes. It is ok'; opts.Resize = 'on'; opts.Interpreter = 'tex';
            answer = questdlg('Is this OK?','Confirmation','Yes. It is ok','No. I want to repeat it',opts);
            switch answer
                case 'Yes. It is ok'
                    isok = true;
            end
        end
        close(f)
    end
end
%*****************************************************
function handler = create_maximized_figureint(sc)
    a = get(0,'MonitorPositions');
    handler = figure('units','pixels','outerposition',a(sc,:));
end
%*****************************************************
function cuber = rotate_flip_Lint(cube, angle, LR,UP)
    
    % INPUTS:
    % cube: data cube to be rotated.
    % angle: rotation angle (-90 to 180). Negative angles rotate clockwise.
    % LR: flip left-right if set to true; 
    % UP: flip up-down if set to true;                  
    
    if UP
        cuber = flipud(imrotate(cube,angle));
    elseif LR
        cuber = fliplr(imrotate(cube,angle));
    else
        cuber = imrotate(cube,angle);
    end
end
%*****************************************************
function time_string=secs2hms(time_in_secs)

% Input: time in seconds (like the output of command 'toc').
% Output: String containig the time in hours, minutes and seconds.
    time_string='';
    nhours = 0;
    nmins = 0;
    if time_in_secs >= 3600
        nhours = floor(time_in_secs/3600);
        if nhours > 1
            hour_string = ' hours, ';
        else
            hour_string = ' hour, ';
        end
        time_string = [num2str(nhours) hour_string];
    end
    if time_in_secs >= 60
        nmins = floor((time_in_secs - 3600*nhours)/60);
        if nmins > 1
            minute_string = ' mins, ';
        else
            minute_string = ' min, ';
        end
        time_string = [time_string num2str(nmins) minute_string];
    end
    nsecs = time_in_secs - 3600*nhours - 60*nmins;
    time_string = [time_string sprintf('%2.1f', nsecs) ' secs'];
end
%*****************************************************
function make_a_beep(duration,frequency)

    amp = 0.5;                        % amplitude (default 1).
    fs = 11025;                     % sampling frequency (default 11025).
    time = 0:1/fs:duration;
    a = amp * sin(2 * pi * frequency * time);
    sound(a)
    
end
%*****************************************************
function im = multibandread_mike(filename,dims,precision,offset,interleave,byteOrder,varargin)
    narginchk(6,9);
    nargoutchk(0,4);
    
    % Check input values.
    filename = convertStringsToChars(filename);
    precision = convertStringsToChars(precision);
    interleave = convertStringsToChars(interleave);
    byteOrder = convertStringsToChars(byteOrder);
    
    if nargin > 6
        isCell = cellfun(@(x) iscell(x), varargin);
        if ~isempty(varargin(isCell))
            for i = isCell
            varargin{i} = cellfun(@(x) convertStringsToChars(x), varargin{i},...
                               'UniformOutput', false);
            end
        end
    end
    
    % Get any subsetted dimensions
    info = parseInputs(filename, dims,...
        precision, offset, byteOrder, varargin{:});
    
    % Make sure that the file is large enough for the requested operation
    fileInfo = dir(info.filename);
    fileSize = fileInfo.bytes;
    if fileSize < info.offset + info.dataSize
        error(message('MATLAB:imagesci:multibandread:badFileSize'));
    end
    
    % Take care of the file ordering
    interleave = validatestring(interleave,{'bil','bip','bsq'});
    switch lower(interleave)
        case 'bil'
            readOrder = [2 3 1];
            permOrder = [3 1 2];
        case 'bip'
            readOrder = [3 2 1];
            permOrder = readOrder;
        case 'bsq'
            readOrder = [2 1 3];
            permOrder = readOrder;
    end
    
    % Create a cell array of the dimension indices
    ndx = {info.rowIndex info.colIndex info.bandIndex};
    
    % Decide which reading algorithm to use
    if useMemMappedFile(info.inputClass)
        try
            % Read from a memory mapped file
            im = readMemFile(filename, info, ndx, readOrder);
        catch  %#ok<CTCH>
            % We may not have been able to map the file into memory.
            % In this case, try reading the data directly from the disk.
            info = getDefaultIndices(info, false);
            ndx = {info.rowIndex info.colIndex info.bandIndex};
            im = readDiskFile(filename, info, ndx, readOrder);
        end
        im = permute(im, permOrder);
    elseif strcmpi(interleave, 'bip') && ...
            (isempty(info.subset) || isequal(info.subset, 'b'))
        % Special optimization for BIP cases
        if isempty(info.subset)
            % Read full dataset
            im = readDiskFileBip(filename, info);
        else
            % Read a subset.
            im = readDiskFileBipSubset(filename, info);
        end
    else
        % Use the general-purpose routine.
        im = readDiskFile(filename, info, ndx, readOrder);
        im = permute(im, permOrder);
    end
end
%*****************************************************
function im = readMemFile(filename, info, ndx, readOrder)
    % Memory map the file
    m = memmapfile(filename, 'offset', info.offset, 'repeat', 1, ...
        'format', {info.inputClass info.dims(readOrder) 'x'});
    
    % Permute the indices so that they are in read order
    ndx = ndx(readOrder);
    
    % Do any necessary subsetting.
    im = m.data.x(ndx{1}, ndx{2}, ndx{3});
    
    [~, ~,endian] = computer;
    
    % Change the endianness, if necessary
    if strcmpi(endian, 'l')
        if strcmpi(info.byteOrder, 'ieee-be') || strcmpi(info.byteOrder, 'b')
            im = swapbytes(im);
        end
    else
        if strcmpi(info.byteOrder, 'ieee-le') || strcmpi(info.byteOrder, 'l')
            im = swapbytes(im);
        end
    end
    
    % Change the type of the output, if necessary.
    if ~strcmp(info.inputClass, info.outputClass)
        im = feval(info.outputClass, im);
    end
end
%*****************************************************
function im = readDiskFile(filename, info, srcNdx, readOrder)
    % A general-purpose routine to read from the disk
    % We use fread, which will handle non-integral (bit) types.
    info.fid = fopen(filename, 'r', info.byteOrder);
    lastReadPos = 0;
    skip(info,info.offset,lastReadPos);
    
    % Do permutation of sizes and indices
    srcNdx = srcNdx(readOrder);
    dim = info.dims(readOrder);
    
    % Preallocate image output array
    outputSize = [length(srcNdx{1}), length(srcNdx{2}), length(srcNdx{3})];
    im = zeros(outputSize(1), outputSize(2), outputSize(3), info.outputClass);
    
    % Determine the start and ending read positions
    kStart=srcNdx{1}(1);
    kEnd=srcNdx{1}(end);
    
    % srcNdx is a vector which contains the desired row, column, and band
    % subsets of the input.  destNdx contains the destination in the output
    % matrix.
    destNdx(3) = 1;
    for i=srcNdx{3}
        pos(1) = (i-1)*dim(1)*dim(2);
        destNdx(2) = 1;
        for j=srcNdx{2}
            pos(2) = (j-1)*dim(1);
    
            % Determine what to read
            posStart = pos(1) + pos(2) + kStart;
            posEnd = pos(1) + pos(2) + kEnd;
            readAmt = posEnd - posStart + 1;
    
            % Read the entire dimension
            skipNum = (posStart-1)-lastReadPos;
            if skipNum
                if info.bitPrecision
                    fread(info.fid, skipNum, info.precision);
                else
                    fseek(info.fid, skipNum*info.eltsize, 'cof');
                end
            end
            [data, count] = fread(info.fid, readAmt, info.precision);
            if count ~= readAmt
                msg = ferror(info.fid);
                fclose(info.fid);
                error(message('MATLAB:imagesci:multibandread:readProblem', msg));
            end
            lastReadPos = posEnd;
    
    
            % Assign the specified subset of what was read to the output matrix
            im(:,destNdx(2),destNdx(3)) = data(srcNdx{1}-kStart+1);
            destNdx(2) = destNdx(2) + 1;
        end
        destNdx(3) = destNdx(3) + 1;
    end
    
    fclose(info.fid);
end
%*****************************************************
function im = readDiskFileBip(filename, info)
    % Read a file from disk when no subsetting is requested.  Only a single
    % call to FREAD is necessary.
    info.fid = fopen(filename, 'r', info.byteOrder);
    lastReadPos = 0;
    skip(info,info.offset,lastReadPos);
    
    % extract the dims into meaningful terms
    height = info.dims(1);
    width  = info.dims(2);
    bands  = info.dims(3);
    
    % Read all bands at once.
    im = fread(info.fid,prod(info.dims),['1*' info.precision]);
    im = reshape(im,[bands width height]);
    im = permute(im,[3 2 1]);
    fclose(info.fid);
end
%*****************************************************
function im = readDiskFileBipSubset(filename, info)
    % Read a file from disk, using optimizations applicable to the BIP case
    % when we read bands at a time.
    info.fid = fopen(filename, 'r', info.byteOrder);
    lastReadPos = 0;
    skip(info,info.offset,lastReadPos);
    
    % extract the dims into meaningful terms
    height = info.dims(1);
    width  = info.dims(2);
    bands  = info.dims(3);
    im = zeros(width, height, numel(info.bandIndex), info.outputClass);
    
    % Read the file, one band at a time.
    plane = 1;
    for i=info.bandIndex
        skip(info,info.offset,i-1);
        [data,count] = fread(info.fid, height*width, ...
            ['1*' info.precision], ...
            (bands-1)*info.eltsize);
        if count ~= height*width
            msg = ferror(info.fid);
            fclose(info.fid);
            error(message('MATLAB:imagesci:multibandread:readProblem', msg));
        end
        im(:,:,plane) = reshape(data,size(im(:,:,plane)));
        plane = plane + 1;
    end
    
    im = permute(im,[2 1 3]);
    fclose(info.fid);
end
%*****************************************************
function skip(info,offset,skipSize)
    % Skip to a specified position in the file
    if info.bitPrecision
        fseek(info.fid,offset,'bof');
        fread(info.fid,skipSize,info.precision);
    else
        fseek(info.fid,offset+skipSize*info.eltsize,'bof');
    end
end
%*****************************************************
function info = parseInputs(filename, dims, precision,offset, byteOrder, varargin)
    % Open the file. Determine pixel width and input/output classes.
    fid = fopen(filename,'r',byteOrder);
    if fid == -1
        error(message('MATLAB:imagesci:validate:fileOpen', filename));
    end
    info = getPixelInfo(fid,precision);
    info.filename = fopen(fid);
    fclose(fid);
    
    % Assign sizes
    validateattributes(offset,{'numeric'},{'nonnegative'});
    info.offset = offset;
    
    validateattributes(dims,{'numeric'},{'numel',3});
    info.dims = dims;
    info.byteOrder  = byteOrder;
    
    % Calculate the size of the data
    if info.bitPrecision
        info.dataSize = prod(info.dims) * (info.eltsize/8);
    else
        info.dataSize = prod(info.dims) * info.eltsize;
    end
    
    info.rowIndex  = ':';
    info.colIndex  = ':';
    info.bandIndex = ':';
    bUseMemMap = useMemMappedFile(info.inputClass);
    info = getDefaultIndices(info, bUseMemMap);
    
    % 'subset' is a string with 0-3 characters (r, c, b). The string
    % represents which dimension is being subset
    info.subset = '';
    
    % Analyze the parameters that specify the subset of interest
    if numel(varargin) > 0
        for i=1:length(varargin)
            % Determine the subsetting method
            methods = {'direct', 'range'};
            if length(varargin{i})==2
                %Default to 'Direct'
                method = 'direct';
                n = 2;
            else
                param = varargin{i}{2};
                match = find(strncmpi(param,methods,numel(param)));
                if ~isempty(match)
                    method = methods{match};
                else
                    error(message('MATLAB:imagesci:multibandread:badSubset', param));
                end
                n = 3;
            end
            % Determine the orientation of the subset
            dimensions = {'row', 'column', 'band'};
            param = varargin{i}{1};
            match = find(strncmpi(param,dimensions,numel(param)));
            if ~isempty(match)
                dim = dimensions{match};
            else
                error(message('MATLAB:imagesci:multibandread:unrecognizedDimSubsetString', varargin{ i }{ 1 }));
            end
            % Get the indices for the subset
            info.subset(i) = dimensions{match}(1); %build a string 'rcb'
            switch dim
                case 'row'
                    info.rowIndex = getIndices(method, i, n, varargin{:});
                case 'column'
                    info.colIndex = getIndices(method, i, n, varargin{:});
                case 'band'
                    info.bandIndex = getIndices(method, i, n, varargin{:});
            end
        end
    end
end
%*****************************************************
function ndx = getIndices(method, i, n, varargin)
    % Use a subsetting method
    if strcmp(method,'direct')
        ndx = varargin{i}{n};
    else
        switch length(varargin{i}{n})
            case 2
                ndx = feval('colon',varargin{i}{n}(1),varargin{i}{n}(2)); %#ok<*FVAL> 
            case 3
                ndx = feval('colon',varargin{i}{n}(1), ...
                    varargin{i}{n}(2), varargin{i}{n}(3));
            otherwise
                error(message('MATLAB:imagesci:multibandread:badSubsetRange'))
        end
    end
end
%*****************************************************
function info = getPixelInfo(fid, precision)
    % Returns size of each pixel.  Size is in bytes unless precision is
    % bitN or ubitN, in which case width is in bits.
    
    % Determine if precision is bitN or ubitN
    info.bitPrecision = ~isempty(strfind(precision,'bit')); %#ok<*STREMP> 
    
    % Reformat the precision string and determine if bit precision is in the
    % shorthand notation
    bitPrecisionWithShorthand = false;
    info.precision = precision(~isspace(precision));
    if strncmp(info.precision(1),'*',1)
        % For bit precision with *source shorthand, set precision to the input
        % precision.  fread will know how to determine the output class, which
        % is the smallest class that can contain the input.
        if info.bitPrecision
            info.precision = precision;
            bitPrecisionWithShorthand = true;
        else
            info.precision(1) = [];
            info.precision = [info.precision '=>' info.precision];
        end
    end
    
    % Determine the input and output types (classes)
    lastInputChar = strfind(info.precision,'=>')-1;
    if isempty(lastInputChar)
        lastInputChar=length(info.precision);
    end
    if bitPrecisionWithShorthand
        info.inputClass = info.precision(2:lastInputChar);
    else
        info.inputClass = precision(1:lastInputChar);
    end
    p = ftell(fid);
    tmp = fread(fid, 1, info.precision);
    info.eltsize = ftell(fid)-p;
    info.outputClass = class(tmp);
    
    % If bit precision with shorthand, set the final precision to the longhand 
    % notation, then parse the precision string to determine eltsize
    if info.bitPrecision
        if bitPrecisionWithShorthand
            info.precision = [info.inputClass '=>' info.outputClass];
        end
        info.eltsize = sscanf(info.inputClass(~isletter(info.inputClass)), '%d');
        if isempty(info.eltsize)
            error(message('MATLAB:imagesci:multibandread:badPrecision', info.precision));
        end
    end
end
%*****************************************************
function info = getDefaultIndices(info, bUseMemMap)
    if ~bUseMemMap
        if ischar(info.rowIndex) && info.rowIndex  == ':'
            info.rowIndex  = 1:info.dims(1);
        end
        if ischar(info.colIndex) && info.colIndex  == ':'
            info.colIndex  = 1:info.dims(2);
        end
        if ischar(info.bandIndex) && info.bandIndex  == ':'
            info.bandIndex  = 1:info.dims(3);
        end
    end
end
%*****************************************************
function rslt = useMemMappedFile(type)
    switch(type)
        case {'int8' 'uint8' 'int16' 'uint16' 'int32' 'uint32' ...
                'int64' 'uint64' 'single' 'double'}
            rslt = true;
        otherwise
            rslt = false;
    end
end
%*****************************************************
function wl = set_wl(binning)
    if binning
        wl = [320.32;324.11;327.91;331.71;335.5;339.3;343.09;346.88;350.68;... % Spectral binning (135 bands).
            354.46;358.25;362.04;365.83;369.61;373.4;377.18;380.96;384.74;...
            388.52;392.3;396.07;399.85;403.62;407.4;411.17;414.94;418.71;...
            422.48;426.24;430.01;433.78;437.53;441.3;445.06;448.82;452.58;...
            456.34;460.09;463.84;467.6;471.35;475.1;478.85;482.6;486.35;...
            490.1;493.84;497.59;501.33;505.07;508.81;512.55;516.29;520.02;...
            523.76;527.5;531.23;534.96;538.69;542.42;546.15;549.88;553.6;...
            557.33;561.05;564.78;568.5;572.22;575.93;579.65;583.37;587.08;...
            590.8;594.51;598.22;601.93;605.64;609.35;613.06;616.76;620.47;...
            624.17;627.87;631.57;635.27;638.97;642.67;646.36;650.06;653.75;...
            657.44;661.14;664.82;668.51;672.2;675.89;679.57;683.26;686.94;...
            690.62;694.3;697.98;701.66;705.34;709.01;712.68;716.36;720.03;...
            723.7;727.37;731.04;734.7;738.37;742.04;745.7;749.36;753.02;...
            756.68;760.34;764.0;767.66;771.31;774.96;778.62;782.27;785.92;...
            789.57;793.22;796.86;800.51;804.15;807.8;811.44;815.08;818.72]; 
    else
        wl = [319.36;321.26;323.16;325.06;326.96;328.86;330.76;332.66;334.56;... % No binning (270 bands).
            336.45;338.35;340.25;342.14;344.04;345.94;347.83;349.73;351.62;...
            353.52;355.41;357.31;359.2;361.09;362.99;364.88;366.77;368.67;...
            370.56;372.45;374.34;376.23;378.12;380.02;381.9;383.79;385.69;...
            387.58;389.46;391.35;393.24;395.13;397.02;398.9;400.79;402.68;...
            404.56;406.46;408.34;410.22;412.11;414.0;415.88;417.76;419.65;...
            421.54;423.42;425.3;427.19;429.06;430.95;432.84;434.72;436.6;...
            438.48;440.35;442.24;444.11;446.0;447.88;449.76;451.63;453.52;...
            455.4;457.28;459.16;461.03;462.9;464.78;466.66;468.54;470.42;...
            472.29;474.16;476.04;477.92;479.79;481.66;483.54;485.42;487.28;...
            489.16;491.04;492.9;494.78;496.65;498.52;500.4;502.26;504.13;...
            506.0;507.88;509.74;511.62;513.48;515.35;517.22;519.09;520.96;...
            522.83;524.7;526.56;528.43;530.3;532.16;534.03;535.89;537.76;...
            539.62;541.49;543.35;545.22;547.08;548.94;550.81;552.67;554.54;...
            556.4;558.26;560.12;561.98;563.84;565.71;567.57;569.42;571.28;...
            573.15;575.0;576.86;578.72;580.58;582.44;584.29;586.16;588.01;...
            589.87;591.72;593.58;595.44;597.29;599.15;601.0;602.86;604.72;...
            606.57;608.42;610.28;612.13;613.98;615.84;617.69;619.54;621.39;...
            623.24;625.1;626.94;628.8;630.65;632.5;634.35;636.2;638.04;639.9;...
            641.74;643.59;645.44;647.29;649.14;650.98;652.83;654.67;656.52;...
            658.37;660.21;662.06;663.9;665.75;667.59;669.44;671.28;673.12;...
            674.97;676.81;678.65;680.49;682.34;684.18;686.02;687.86;689.7;...
            691.54;693.38;695.22;697.06;698.9;700.74;702.58;704.42;706.25;...
            708.09;709.93;711.77;713.6;715.44;717.28;719.11;720.95;722.78;...
            724.62;726.45;728.29;730.12;731.96;733.79;735.62;737.46;739.29;...
            741.12;742.95;744.79;746.62;748.45;750.28;752.11;753.94;755.77;...
            757.6;759.43;761.26;763.08;764.91;766.74;768.57;770.4;772.22;...
            774.05;775.88;777.7;779.53;781.36;783.18;785.0;786.83;788.66;...
            790.48;792.3;794.13;795.95;797.78;799.6;801.42;803.24;805.06;...
            806.89;808.7;810.52;812.34;814.16;815.98;817.8;819.62];
    end
end
%*****************************************************
function hcube = smooth_hypercube(hcube,span)
    cube = hcube.DataCube;
    [rows,columns,bands] = size(cube);
    matrix = reshape(cube,rows*columns,bands)';
    f = waitbar(0,'Smothing cube. Please wait...');
    for i = 1:size(matrix,2)
        waitbar(i/size(matrix,2),f,'Smoothing cube. Please wait...')
        matrix(:,i) = smooth(matrix(:,i),span);
    end
    close(f);
    cube = reshape(matrix',rows,columns,bands);
    wl = hcube.Wavelength;
    hcube = hypercube(cube,wl);
end
%*****************************************************
function white = set_white(white_reflectance)
    switch white_reflectance
        case 'Multi_90'
            white = [0.94958574	0.94724384	0.9443403	0.94459232	0.9462478	0.94488912	0.94335204	0.94380255	0.94280032	0.94441593	...
                     0.94430661	0.94334806	0.943664	0.94293963	0.94332739	0.94272599	0.9429587	0.94392938	0.94453078	0.9460848	...
                     0.94527088	0.94428336	0.94475201	0.94535213	0.94555955	0.94542305	0.94484793	0.94437511	0.94459751	0.94418216	...
                     0.94478368	0.94471024	0.94471005	0.94466565	0.94419501	0.94424115	0.94408467	0.9440786	0.94401017	0.94374275	...
                     0.9433519	0.94327987	0.94313649	0.94317241	0.94249733	0.94219909	0.94150077	0.94181476	0.94195696	0.94125809	...
                     0.94139837	0.94186997	0.94073934	0.94099113	0.94055914	0.94057246	0.9403842	0.9401759	0.93898701	0.93962702	...
                     0.93902978	0.93941681	0.93915889	0.93888463	0.93859995	0.93823624	0.93789211	0.93680851	0.93830738	0.93716315	...
                     0.93774751	0.93801031	0.9367933	0.93770679	0.93520652	0.93568712	0.93417627	0.93695163	0.93623721	0.93590096	...
                     0.93408144	0.93555625	0.93310126	0.93463823	0.93559375	0.93420273	0.93398872	0.93368243	0.93353137	0.93361078	...
                     0.93498727	0.93194868	0.93313056	0.93236357	0.93084149];
        case 'Multi_50'
            white = [0.4723838	0.47130066	0.46934503	0.46897196	0.4708246	0.46901661	0.46774103	0.46839719	0.467774	0.46875328	...
                     0.46845749	0.467958	0.46792293	0.46754892	0.46736892	0.46685422	0.46647415	0.46682961	0.46697487	0.46733659	...
                     0.46685755	0.46654177	0.46682085	0.46705577	0.4673019	0.46725241	0.46707747	0.46717142	0.46727374	0.4671495	...
                     0.46728773	0.46754304	0.46760767	0.46765941	0.46758285	0.46786853	0.46793148	0.46808321	0.46818812	0.46802307	...
                     0.46788282	0.46802407	0.46790166	0.46805623	0.46789951	0.46783813	0.46772151	0.46777748	0.46786772	0.46755084	...
                     0.46763466	0.46806967	0.46762315	0.46776454	0.46755818	0.4677134	0.46739751	0.46760087	0.46733749	0.46759611	...
                     0.46686286	0.46709097	0.46718996	0.46708498	0.46687323	0.46675459	0.46672407	0.46628402	0.46705108	0.46671129	...
                     0.46637221	0.46686794	0.4666536	0.46685049	0.46548041	0.46612975	0.46453376	0.46661001	0.46634287	0.46551793	...
                     0.46602229	0.46573882	0.4636831	0.46556097	0.46669917	0.46532979	0.46458038	0.46463455	0.46434963	0.46466226	...
                     0.46523238	0.46414249	0.46435568	0.46472262	0.46401024];
        case 'Multi_20'
            white = [0.21392675	0.21340871	0.21254689	0.21189364	0.2121142	0.21185723	0.21077652	0.21100838	0.21062764	0.21091786	...
                     0.21083696	0.2104912	0.21044507	0.2101407	0.21028306	0.20981019	0.20964433	0.20972398	0.20979658	0.20990952	...
                     0.20966362	0.20957661	0.20978402	0.20987631	0.2100352	0.21000153	0.20972609	0.20969174	0.20979172	0.2098454	...
                     0.21021618	0.21032558	0.21051733	0.21056008	0.2103828	0.21052186	0.21052556	0.21074624	0.21077597	0.21077484	...
                     0.21093133	0.21108845	0.21122852	0.21131883	0.21120423	0.21137677	0.21157519	0.21161144	0.21168401	0.21174482	...
                     0.21197004	0.21215179	0.21207316	0.21227756	0.21242335	0.2124583	0.21251601	0.21264572	0.21252814	0.21263846	...
                     0.21264341	0.21273616	0.21285031	0.21300244	0.21302917	0.2132191	0.21321852	0.21309248	0.21348334	0.21341592	...
                     0.21332419	0.21344195	0.21353372	0.21365211	0.21365761	0.21390162	0.21338438	0.21406791	0.21419747	0.21402791	...
                     0.21428453	0.21419711	0.21341569	0.21432375	0.21447735	0.21466124	0.21433336	0.21400193	0.21431947	0.21426312	...
                     0.21481707	0.21443041	0.2150735	0.21463276	0.21501833];
        case 'Multi_5'
            white = [0.05773317	0.05714976	0.05685759	0.0566	0.05661062	0.05624815	0.05616047	0.05598794	0.05579115	0.05565513	...
                     0.05561771	0.05551106	0.05538459	0.05523579	0.05508688	0.05496392	0.05488017	0.05472607	0.05477159	0.05474116	...
                     0.05458616	0.05452788	0.0545308	0.05449468	0.05445831	0.0545173	0.05444861	0.05440789	0.05440832	0.05430464	...
                     0.05428624	0.05425417	0.05414626	0.05419017	0.05418609	0.05431517	0.05426087	0.05421678	0.05420416	0.05415117	...
                     0.05411587	0.05411848	0.0541009	0.05409916	0.05410793	0.05397056	0.05398327	0.05394388	0.05399287	0.05397952	...
                     0.05398466	0.05396197	0.05389377	0.05393616	0.05389164	0.05387357	0.05394144	0.05387318	0.0538686	0.05389293	...
                     0.05384927	0.05389825	0.05381367	0.05384674	0.05385136	0.053834	0.05385769	0.05371137	0.05378497	0.05378589	...
                     0.0538279	0.053885	0.05371464	0.05380113	0.05366965	0.05364658	0.05362681	0.05371396	0.05381123	0.05381967	...
                     0.05373976	0.05368556	0.05349006	0.05369128	0.0537186	0.05387582	0.05379846	0.05384944	0.05377229	0.05383468	...
                     0.05371257	0.05358403	0.05355925	0.05365518	0.05364144];
        case 'Sphere'
            white = [0.960807	0.954703	0.957531	0.955764	0.956448	0.956493	0.96019	0.961455	0.956325	0.957322	...
                     0.958326	0.959727	0.957249	0.961698	0.960664	0.960214	0.960913	0.961164	0.962808	0.961128	...
                     0.962872	0.962899	0.961912	0.963136	0.963271	0.962751	0.963209	0.961616	0.963259	0.963214	...
                     0.962608	0.962631	0.963231	0.963787	0.962812	0.963031	0.962552	0.963194	0.96294	0.963821	...
                     0.963182	0.962933	0.962903	0.963242	0.962465	0.962079	0.962773	0.96231	0.962831	0.962976	...
                     0.962989	0.96224	0.962501	0.962224	0.961619	0.961471	0.961286	0.961372	0.961392	0.960574	...
                     0.961258	0.960884	0.960734	0.960616	0.960787	0.960672	0.961364	0.960679	0.959627	0.960557	...
                     0.959152	0.959849	0.960282	0.958783	0.959405	0.959603	0.958373	0.959695	0.958605	0.959007	...
                     0.958096	0.956925	0.959055	0.957555	0.958449	0.957913	0.958662	0.958592	0.95692	0.957957	...
                     0.958559	0.957313	0.958731	0.957047	0.955923];
        case 'Teflon'
            white = [0.914575249	0.916116666	0.917594233	0.919060329	0.920539923	0.922045102	0.923580043	0.92513367	0.926684907	0.928197646	...
                     0.929615895	0.930895288	0.931997885	0.932890762	0.933550272	0.933978337	0.93416985	0.934121503	0.9338635	0.933455841	...
                     0.932956439	0.93241854	0.931880007	0.931351703	0.930829264	0.93029857	0.929743198	0.929152484	0.928523942	0.92786578	...
                     0.927202005	0.926556615	0.925951668	0.92541105	0.92494052	0.924539406	0.924194737	0.923877925	0.923552389	0.92319354	...
                     0.922789371	0.922343176	0.921869937	0.921380585	0.920881043	0.920370552	0.919843283	0.91929091	0.918704799	0.918084716	...
                     0.917440687	0.916789219	0.91615601	0.915569954	0.915052021	0.914611941	0.914244831	0.913915731	0.913588234	0.913232919	...
                     0.912831036	0.912389062	0.911918579	0.911431733	0.91093828	0.91044574	0.909960551	0.909491318	0.90904861	0.908632193	...
                     0.908227241	0.907795677	0.907271945	0.906597528	0.905727555	0.904633856	0.903332345	0.901871528	0.900299921	0.898666504	...
                     0.897019994	0.895390913	0.893795917	0.892243841	0.890731888	0.889257234	0.887818777	0.886417106	0.885043066	0.88368535	...
                     0.882330831	0.88096531	0.879578779	0.878161608	0.876702034];
        otherwise
            white = white_reflectance;
    end
end