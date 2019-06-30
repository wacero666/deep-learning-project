MillionSong ='/MSD'; 
msd_data_path=[MillionSong,'/data/H'];
msd_addf_path=[MillionSong,'/AdditionalFiles'];
MSDsubset=''; % or '' for full set
msd_addf_prefix=[msd_addf_path,'/',MSDsubset];
% Check that we can actually read the dataset
assert(exist(msd_data_path,'dir')==7,['msd_data_path ',msd_data_path,' is not found.']);

% path to the Million Song Dataset code
msd_code_path='MSongsDB';
assert(exist(msd_code_path,'dir')==7,['msd_code_path ',msd_code_path,' is wrong.']);
% add to the path
addpath([msd_code_path,'/MatlabSrc/']);

% Build a list of all the files in the dataset
all_files = findAllFiles(msd_data_path);
cnt = length(all_files);
disp(['Number of h5 files found: ',num2str(cnt)]);

% Get info from the first file using our wrapper
h5 = HDF5_Song_File_Reader(all_files{1});
disp(['artist name is: ',h5.get_artist_name()]);
disp([' song title is: ',h5.get_title()]);
disp(['track id is: ',h5.get_track_id()]);

% Show all the available methods
methods('HDF5_Song_File_Reader')

% .. covers all the EN Analyze API fields

% Plot the first 200 chromas
%chromas = h5.get_segments_pitches();
%subplot(311)
%imagesc(chromas(:,1:200))
%axis xy
%colormap(1-gray)
%colorbar
%title('first 200 chromas');

% Resynthesize the first 30 seconds using chroma and timbre
sr = 16000;
dur = 60; % first 30s
%x = en_resynth(h5,dur,sr);
% Take a listen
%soundsc(x,sr);
% recognizable?
% Plot the spectrogram, for comparison
%subplot(312)
%specgram(x,1024,sr);
%caxis(max(caxis)+[-80 0]);
%saveas(gcf, "h5.png")

%disp(all_files(1:2));

% Get all artist names by mapping a function to return artist names
% over the cell array of data file names
%tic;
%all_artist_names = cellfun(@(f) get_artist_name(HDF5_Song_File_Reader(f)), ...
%                           all_files, 'UniformOutput', false);
%tend = toc;
%disp(['All names acquired in ',num2str(tend),' seconds.']);
%disp(['First artist name is: ',all_artist_names{1}]);
%disp(['There are ',num2str(length(unique(all_artist_names))), ...
%      ' unique artist names']);
% takes around 5 min on MacBook Pro to scan 10k files (30ms/file)

cellfun(@(f) SaveSpect(f, sr, dur, 1024), all_files, 'UniformOutput', false);

function SaveSpect(f, sr, dur, nttf)
   h5 = HDF5_Song_File_Reader(f);
   track_id = h5.get_track_id();
   x = en_resynth(h5, dur, sr);
   specgram(x, nttf, sr);
   caxis(max(caxis)+[-80 0]);
   axis off;
   set(gcf, 'units', 'normalized'); %Just making sure it's normalized
   export_fig (['images_H/',track_id,'.png'])
end




