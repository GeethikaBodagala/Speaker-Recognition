% Set the working directory to where your audio files are located
audio_files_path = 'C:\Users\HP\Desktop\speaker recognition';

% Change to the directory with audio files
cd(audio_files_path);

% List of required audio files
audioFiles = {'speaker1.wav', 'speaker2.wav', 'speaker3.wav','speaker4.wav' ,'test_speaker.wav'};

% Check the existence of each file
for i = 1:length(audioFiles)
    if exist(audioFiles{i}, 'file') == 2
        disp([audioFiles{i}, ' exists.']);
    else
        disp([audioFiles{i}, ' does NOT exist.']);
        error([audioFiles{i}, ' does NOT exist in the specified path.']);
    end
end

% Add the Voicebox toolbox to MATLAB path
voicebox_path = 'C:\Users\HP\Desktop\speaker recognition\sap-voicebox-master\voicebox';  % e.g., 'C:\Users\<YourUsername>\Downloads\voicebox'
addpath(voicebox_path);

% Save the path for future sessions
savepath;

% Number of speakers
numSpeakers = 4;

% Cell array to store templates for each speaker
templates = cell(1, numSpeakers);

% Load and process audio files for each speaker
for i = 1:numSpeakers
    filename = fullfile(audio_files_path, ['speaker', num2str(i), '.wav']);
    [audio, fs] = audioread(filename);
    % Extract MFCC features
    templates{i} = melcepst(audio, fs);
end

% Load and process the test audio
test_filename = fullfile(audio_files_path, 'test_speaker.wav');
[testAudio, fs] = audioread(test_filename);
testCoeffs = melcepst(testAudio, fs);

% Initialize variables for matching
minDistance = Inf;
identifiedSpeaker = -1;

% Function to compute DTW distance
function d = dtw(x, y)
    [m, n] = size(x);
    [p, q] = size(y);
    D = zeros(m + 1, p + 1) + Inf;
    D(1, 1) = 0;
    for i = 2:m + 1
        for j = 2:p + 1
            cost = sum((x(i-1, :) - y(j-1, :)).^2);
            D(i, j) = cost + min([D(i-1, j), D(i, j-1), D(i-1, j-1)]);
        end
    end
    d = D(m + 1, p + 1);
end

% Match test audio to each speaker template using DTW
for i = 1:numSpeakers
    distance = dtw(testCoeffs, templates{i});
    if distance < minDistance
        minDistance = distance;
        identifiedSpeaker = i;
    end
end

% Display the identified speaker
disp(['Identified Speaker: ', num2str(identifiedSpeaker)]);
