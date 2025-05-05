function [Coordinates, Names]=Extract_minicubes_v4_2(pcube, Metadata, N, range,cubename,startnum,parentcubename,minicinfo,Manual,Coord_Ext)

%function that extracts N rectangular areas from a hybercube and stores
%them as individual mini-cubes. It returns the coordinates of the extracted
%areas and the names of the mini-cubes. 
% The range parameter --> is either VNIR or SWIR (both strings). 
% The cubename parameter--> is a string containing the
%feature name of the group of minicubes (e.g. 'lab_samples' or 'Royal'). 
% startnum -->is the number of the first minimube sample e.g. 532 
%parentcubename  --> is a string variable with the name of the parent cube
%minicinfo  --> is the cell array with the descriptions
%of the minicubes. Example: 'minicubes_info'.
%Manual--> logical input set to 1 if the extraction will be manual, or 0 if
%it will be automatic from a set of coordinates
% Coord_Ext --> matrix of Nx4 containing the extraction positions in case
% of automatic extraction. This input field must be [] if the extraction is
% manual.
%EXAMPLE1: [Coordinates, Names]=Extract_minicubes_v4(cube, Metadata, 2,
%'VNIR','Royal',00991,'MPD41a','info_minicubes.xlsx', 1,[]);
%EXAMPLE2: [Coordinates, Names]=Extract_minicubes_v4(cube, Metadata, 2,
%'VNIR','Royal',00991,'MPD41a','info_minicubes.xlsx', 0,[15 60 100 90;25 80 50 75]);


sp_info=pcube.DataCube; %parent cube
Met=Metadata; %Parent Metadata
wl=pcube.Wavelength; %Parent wavelength
switch range
    case 'VNIR'
        bands=[42,28,7];
    case 'SWIR'
        bands=[141,61,21];
end

f = create_maximized_figureint(1);
    imshow(cat(3,imadjust(sp_info(:,:,bands(1))),imadjust(sp_info(:,:,bands(2))),imadjust(sp_info(:,:,bands(3)))),[])
    
    for k=1:N
        title('Full cube');
        if Manual
             disp(['Extracting region ',num2str(k),' (press enter after cropping)...'])
             h = imrect(gca); %#ok<*IMRECT> 
             pause()
             pos = round(getPosition(h));
        else
            pos=Coord_Ext(k,:);
            h = images.roi.Rectangle(gca,'Position',pos,'StripeColor','r');
            disp(['Extracting region ',num2str(k),'. Coordinates:',num2str(pos(1,1)),', ',num2str(pos(1,2)),', ',num2str(pos(1,3)),', ',num2str(pos(1,4))]);
            pause(1);
        end
        cropped_cube(k).Data = sp_info(pos(2):pos(2)+pos(4)-1,pos(1):pos(1)+pos(3)-1,:);    
        
        Coordinates(k,:)=pos;
        Names(k).name=['0',num2str(startnum+k-1),'-',range,'-',cubename];
        Metadata.position=Coordinates(k,:);
        Metadata.parent_cube=parentcubename;
        Metadata.cubeinfo=strjoin([Met.cubeinfo,'Extract:',minicinfo(k,1)]);
        Metadata.name=['0',num2str(startnum+k-1),'-',range,'-',cubename];
        Metadata.height=Coordinates(k,4);
        Metadata.width=Coordinates(k,3);
        Metadata.number=startnum+k-1;
        Metadata.RGB=cropped_cube(k).Data(:,:,bands);
        Metadata.date = Met.date;
        Metadata.restored = Met.restored;
        Metadata.substrate = Met.substrate;
        cube=hypercube(cropped_cube(k).Data,pcube.Wavelength,Metadata);
        disp(['Saving mini-cube ',num2str(k)]);
        save(string(join([Names(k).name,'.mat'],'')),'cube','Metadata','wl');
        %         save('Extracted_coordinates_names.mat','Coordinates','Names');
        image = cat(3,imadjust(cropped_cube(k).Data(:,:,bands(1))),...
            imadjust(cropped_cube(k).Data(:,:,bands(2))),...
            imadjust(cropped_cube(k).Data(:,:,bands(3))));
        imwrite(image,string(join([Names(k).name,'.png'],'')))
    end
close(f)
end


%%
    function handler = create_maximized_figureint(sc)
    a = get(0,'MonitorPositions');
    if sc == 1
        handler = figure('units','pixels','outerposition',a(1,:));
    elseif sc ==2
        handler = figure('units','pixels','outerposition',a(2,:));
    end
    end