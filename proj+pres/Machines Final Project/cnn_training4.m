%% =========================================================================
%  TRAIN CNN FOR MOTOR FAULTS (FINAL REFINED)
%  - Fixed Augmentation (No vertical shifts/rotation)
%  - sharper windowing for BRB resolution
% =========================================================================

%% 1. User Settings
base_names = {
    "Ia",  "Ib",  "Ic";   % healthy
    "Ia1", "Ib1", "Ic1";  % brb
    "Ia2", "Ib2", "Ic2";  % bearing
    "Ia3", "Ib3", "Ic3";  % sws
    "Ia4", "Ib4", "Ic4";  % eccentricity
    "Ia5", "Ib5", "Ic5"   % VI
};
class_labels = ["healthy","brb","bearing","sws","eccentricity","VI"];

root_dir = 'Dataset_Strict_Split';
train_dir = fullfile(root_dir, 'Train');
val_dir   = fullfile(root_dir, 'Val');
test_dir  = fullfile(root_dir, 'Test');

if exist(root_dir,'dir'), rmdir(root_dir,'s'); end
mkdir(train_dir); mkdir(val_dir); mkdir(test_dir);

Fs_target = 1000;            
window_size = 30000;          % 3s @ 10kHz input
window_step = 5000;          
windows_per_signal = 3;      

% --- OPTIMIZED SPECTROGRAM SETTINGS ---
spec.window = 2048;           
spec.noverlap = 1536;         % Higher overlap for smoother image
spec.nfft = 4096;            
img_size = [224,224];
noise_std_frac = 0.01;       
split_ratios = [0.7, 0.15, 0.15];

%% 2. Discover & Split Data
ws_vars = evalin('base','who');
num_classes = size(base_names,1);
fprintf('Discovering signals...\n');

for ci = 1:num_classes
    label = char(class_labels(ci));
    Ia_base = char(base_names{ci,1});
    
    mkdir(fullfile(train_dir, label));
    mkdir(fullfile(val_dir, label));
    mkdir(fullfile(test_dir, label));
    
    patt = strcat('^', Ia_base, '_\d+$');
    matched_vars = {};
    for k = 1:numel(ws_vars)
        if ~isempty(regexp(ws_vars{k}, patt, 'once'))
            matched_vars{end+1} = ws_vars{k}; %#ok<SAGROW>
        end
    end
    matched_vars = sort(matched_vars);
    num_files = numel(matched_vars);
    
    if num_files == 0, continue; end
    
    n_train = floor(num_files * split_ratios(1));
    n_val   = floor(num_files * split_ratios(2));
    
    % 3. Generate Images
    for v = 1:num_files
        if v <= n_train
            current_save_dir = fullfile(train_dir, label);
        elseif v <= (n_train + n_val)
            current_save_dir = fullfile(val_dir, label);
        else
            current_save_dir = fullfile(test_dir, label);
        end
        
        Ia_name = matched_vars{v};
        suffix = extractAfter(Ia_name, strcat(Ia_base, '_'));
        Ib_name = strcat(char(base_names{ci,2}), '_', suffix);
        Ic_name = strcat(char(base_names{ci,3}), '_', suffix);
        
        if ~evalin('base', sprintf('exist(''%s'',''var'')', Ib_name)), continue; end
        
        Ia_raw = evalin('base', Ia_name); Ia_raw = Ia_raw(:);
        Ib_raw = evalin('base', Ib_name); Ib_raw = Ib_raw(:);
        Ic_raw = evalin('base', Ic_name); Ic_raw = Ic_raw(:);
        
        % Remove DC
        Ia_raw = Ia_raw - mean(Ia_raw);
        Ib_raw = Ib_raw - mean(Ib_raw);
        Ic_raw = Ic_raw - mean(Ic_raw);
        
        if length(Ia_raw) < window_size, continue; end
        
        for w = 1:windows_per_signal
            start_idx = 1 + (w-1) * window_step;
            if start_idx + window_size - 1 > length(Ia_raw), break; end
            
            Ia_win = Ia_raw(start_idx : start_idx + window_size - 1);
            Ib_win = Ib_raw(start_idx : start_idx + window_size - 1);
            Ic_win = Ic_raw(start_idx : start_idx + window_size - 1);
            
            % Add Noise
            sstd = noise_std_frac * median([std(Ia_win), std(Ib_win), std(Ic_win)]);
            if sstd > 0
                Ia_win = Ia_win + sstd*randn(size(Ia_win));
                Ib_win = Ib_win + sstd*randn(size(Ib_win));
                Ic_win = Ic_win + sstd*randn(size(Ic_win));
            end
            
            img = create_spectrogram(Ia_win, Ib_win, Ic_win, Fs_target, img_size, spec.window, spec.noverlap, spec.nfft);
            
            fname = sprintf('%s_file%s_win%02d.png', label, suffix, w);
            imwrite(img, fullfile(current_save_dir, fname));
        end
    end
end

%% 4. Load Datastores
imdsTrain = imageDatastore(train_dir, 'IncludeSubfolders',true, 'LabelSource','foldernames');
imdsVal   = imageDatastore(val_dir,   'IncludeSubfolders',true, 'LabelSource','foldernames');
imdsTest  = imageDatastore(test_dir,  'IncludeSubfolders',true, 'LabelSource','foldernames');

%% 5. Define Custom CNN
layers = [
    imageInputLayer([img_size 3], 'Name', 'input', 'Normalization', 'none')
    
    convolution2dLayer(3, 8, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    fullyConnectedLayer(numel(categories(imdsTrain.Labels)))
    softmaxLayer
    classificationLayer
];
lgraph = layerGraph(layers);

%% 6. Training Options (FIXED AUGMENTATION)
use_gpu = (gpuDeviceCount > 0);
if use_gpu, execEnv = 'gpu'; else, execEnv = 'cpu'; end

% --- FIX: DISABLING ROTATION & Y-SHIFT ---
% Frequency features (Y-axis) must not be shifted!
augmenter = imageDataAugmenter(...
    'RandXTranslation', [-20 20], ...   % Time shifting (Robustness to start time)
    'RandYTranslation', [-3 3], ...     % <--- UNCOMMENT/ADD THIS LINE
    'RandXReflection', false);    % Reflection might flip Phase order (RGB), safer to disable

augimdsTrain = augmentedImageDatastore(img_size, imdsTrain, 'DataAugmentation', augmenter);
augimdsVal   = augmentedImageDatastore(img_size, imdsVal);
num_train_images = numel(imdsTrain.Files);
iters_per_epoch = floor(num_train_images / 16);

options = trainingOptions('adam', ...
    'MiniBatchSize', 16, ...
    'MaxEpochs', 50, ...
    'InitialLearnRate', 1e-5, ...
    'L2Regularization', 0.01, ...
    'ValidationData', augimdsVal, ...
    'ValidationFrequency', iters_per_epoch, ...
    'ValidationPatience', 10, ...
    'Shuffle', 'every-epoch', ...
    'ExecutionEnvironment', execEnv, ...
    'Plots', 'training-progress', ...
    'Verbose', true);

%% 7. Train & Test
netTransfer = trainNetwork(augimdsTrain, lgraph, options);
augimdsTest = augmentedImageDatastore(img_size, imdsTest);
pred = classify(netTransfer, augimdsTest, 'ExecutionEnvironment', execEnv);
acc = sum(pred == imdsTest.Labels) / numel(imdsTest.Labels);
fprintf('FINAL TEST ACCURACY: %.2f%%\n', acc*100);
confusionchart(imdsTest.Labels, pred);
% --- SAVE THE TRAINED MODEL ---
save('cnn_trained_model4.mat', 'netTransfer', 'class_labels', 'acc');
fprintf('Model saved to cnn_trained_model4.mat\n');
%% Helper Function
function img = create_spectrogram(Ia, Ib, Ic, Fs_target, img_size, window, noverlap, nfft)
    downsample_factor = 10; 
    try
        Ia_low = decimate(double(Ia), downsample_factor);
        Ib_low = decimate(double(Ib), downsample_factor);
        Ic_low = decimate(double(Ic), downsample_factor);
    catch
        Ia_low = Ia(1:downsample_factor:end);
        Ib_low = Ib(1:downsample_factor:end);
        Ic_low = Ic(1:downsample_factor:end);
    end
    
    Ia_low = Ia_low - mean(Ia_low);
    Ib_low = Ib_low - mean(Ib_low);
    Ic_low = Ic_low - mean(Ic_low);
    
    rgb_img = zeros([img_size 3], 'uint8');
    phases = {Ia_low, Ib_low, Ic_low};
    
    % Pass 1: Global Max (0-400Hz)
    S_db_all = [];
    for ch = 1:3
        % Use Chebyshev window for sharper peaks (Better for BRB)
        win_vec = chebwin(window, 100); 
        [S, F, ~] = spectrogram(phases{ch}(:), win_vec, noverlap, nfft, Fs_target);
        
        valid_idx = find(F >= 0 & F <= 400); 
        S = S(valid_idx, :);
        S_db_all = [S_db_all; 10*log10(abs(S) + eps)]; 
    end
    
    Smax = max(S_db_all(:));
    Smin = Smax - 50;  
    
    % Pass 2: Generate
    for ch = 1:3
        win_vec = chebwin(window, 100);
        [S, F, ~] = spectrogram(phases{ch}(:), win_vec, noverlap, nfft, Fs_target);
        valid_idx = find(F >= 0 & F <= 400); 
        S = S(valid_idx, :);
        S_db = 10*log10(abs(S) + eps);
        S_db(S_db < Smin) = Smin;
        S_norm = (S_db - Smin) / (Smax - Smin);
        S_resized = imresize(S_norm, img_size);
        rgb_img(:,:,ch) = uint8(S_resized * 255);
    end
    img = rgb_img;
end