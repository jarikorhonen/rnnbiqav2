% ----------------------------------------------------------------------
%  trainAndTestRNNmodelv2.m
%
%  Written by Jari Korhonen, Shenzhen University
%
%  This function trains the RNN model version 2 to predict quality
%  of large resolution test images, using a sequence of low and
%  high resultion patch features as input.
% 
%  Usage: result = trainAndTestRNNmodelv2(XTrain,YTrain,XTest,YTest)
%  Inputs:
%      XTrain:    Training feature vector sequences
%      YTrain:    Training ground truth MOS vector
%      XTest:     Testing feature vector sequences
%      YTest:     Testing ground truth MOS vector
%
%  Output:
%      [

function result = trainAndTestRNNmodelv2(XTrain,YTrain,XTest,YTest)

% Preprocess input data
XTest1 = {};
XTest2 = {};
for i=1:length(XTest)
   XTest1{i} = XTest{i}(:,XTest{i}(1,:)==1);
    XTest2{i} = XTest{i}(:,XTest{i}(1,:)==2);
end
YTest = YTest;
X1 = padsequences(XTest1,2,'Length','longest','Direction','left','PaddingValue',0);
X1 = permute(X1,[1 3 2]);
X2 = padsequences(XTest2,2,'Length','longest','Direction','left','PaddingValue',0);
X2 = permute(X2,[1 3 2]);
dlXt1 = dlarray(X1,'CBT');
dlXt2 = dlarray(X2,'CBT');
dlYt = dlarray(YTest,'BC');
numFeatures = size(XTrain{1},1)-1;
numFeatures2 = numFeatures;

% Initialize training options
miniBatchSize = 16;
numEpochs = 6;
learnRate = 0.0002;
learnRateDropFactor = 1;
gradientDecayFactor = 0.9;
squaredGradientDecayFactor = 0.999;
l2Regularization = 0.00000001;
l2RegularizationMult = 1;
state = 0.3;

xDs = arrayDatastore(XTrain,'OutputType','same');
yDs = arrayDatastore(YTrain,'OutputType','same');
trainDs = combine(xDs,yDs);

clear("XTrain","YTrain");

numMiniBatchOutputs = 3;
mbq = minibatchqueue(trainDs,numMiniBatchOutputs,...
    'MiniBatchSize',miniBatchSize,...
    'MiniBatchFormat',{'CBT','CBT','BC'},...
    'MiniBatchFcn',@(x,t) preprocessMiniBatch(x,t));

% Initialize training visualization
figure
lineLossTrain = animatedline('Color',[0.85 0.325 0.098]);
lineLossTest = animatedline('Color',[0 0 0]);
ylim([0 inf])
xlabel("Iteration")
ylabel("Loss")
grid on
iteration = 0;
start = tic;

rng(555); % seed for random initialization


% Initialize pre-stage FC layer (fcpre)
sz = [numFeatures numFeatures];
numOut = numFeatures;
numIn = numFeatures;
parameters.fcpre.Weights = initializeGlorot(sz,numOut,numIn);
parameters.fcpre.Bias = initializeZeros([numOut 1]);

% Initialize 1st GRU layer for low resolution stream (gru_l1)
numHiddenUnits1 = 256;
sz = [3*numHiddenUnits1 numFeatures2];
parameters.gru_l1.Weights = initializeGlorot(sz,3*numHiddenUnits1,numFeatures2);
parameters.gru_l1.RecurrentWeights = initializeOrthogonal([3*numHiddenUnits1 numHiddenUnits1]);
parameters.gru_l1.Bias = initializeZerosGru(3*numHiddenUnits1);

% Initialize 2nd GRU layer for low resolution stream (gru_l2)
numHiddenUnits2 = 128;
sz = [3*numHiddenUnits2 numHiddenUnits1];
parameters.gru_l2.Weights = initializeGlorot(sz,3*numHiddenUnits2,numHiddenUnits1);
parameters.gru_l2.RecurrentWeights = initializeOrthogonal([3*numHiddenUnits2 numHiddenUnits2]);
parameters.gru_l2.Bias = initializeZerosGru(3*numHiddenUnits2);

% Initialize 3rd GRU layer for low resolution stream (gru_l3)
numHiddenUnits3 = 64;
sz = [3*numHiddenUnits3 numHiddenUnits2];
parameters.gru_l3.Weights = initializeGlorot(sz,3*numHiddenUnits3,numHiddenUnits2);
parameters.gru_l3.RecurrentWeights = initializeOrthogonal([3*numHiddenUnits3 numHiddenUnits3]);
parameters.gru_l3.Bias = initializeZerosGru(3*numHiddenUnits3);
 
% Initialize 1st GRU layer for high resolution stream (gru_h1)
numHiddenUnits1 = 256;
sz = [3*numHiddenUnits1 numFeatures2];
parameters.gru_h1.Weights = initializeGlorot(sz,3*numHiddenUnits1,numFeatures2);
parameters.gru_h1.RecurrentWeights = initializeOrthogonal([3*numHiddenUnits1 numHiddenUnits1]);
parameters.gru_h1.Bias = initializeZerosGru(3*numHiddenUnits1);

% Initialize 2nd GRU layer for high resolution stream (gru_h2)
numHiddenUnits2 = 128;
sz = [3*numHiddenUnits2 numHiddenUnits1];
parameters.gru_h2.Weights = initializeGlorot(sz,3*numHiddenUnits2,numHiddenUnits1);
parameters.gru_h2.RecurrentWeights = initializeOrthogonal([3*numHiddenUnits2 numHiddenUnits2]);
parameters.gru_h2.Bias = initializeZerosGru(3*numHiddenUnits2);

% Initialize 3rd GRU layer for high resolution stream (gru_h3)
numHiddenUnits3 = 64;
sz = [3*numHiddenUnits3 numHiddenUnits2];
parameters.gru_h3.Weights = initializeGlorot(sz,3*numHiddenUnits3,numHiddenUnits2);
parameters.gru_h3.RecurrentWeights = initializeOrthogonal([3*numHiddenUnits3 numHiddenUnits3]);
parameters.gru_h3.Bias = initializeZerosGru(3*numHiddenUnits3);

% Initialize the GRU layer for head (gruhead)
numHiddenUnits4 = 64;
sz = [3*numHiddenUnits4 numHiddenUnits3*2];
parameters.gruhead.Weights = initializeGlorot(sz,3*numHiddenUnits4,numHiddenUnits3*2);
parameters.gruhead.RecurrentWeights = initializeOrthogonal([3*numHiddenUnits4 numHiddenUnits4]);
parameters.gruhead.Bias = initializeZerosGru(3*numHiddenUnits4);

% Initialize 1st FC layer for head (fc1)
numIn = numHiddenUnits4;
numOut = numHiddenUnits4;
sz = [numOut numIn];
parameters.fc1.Weights = initializeGlorot(sz,numOut,numIn);
parameters.fc1.Bias = initializeZeros([numOut 1]);

% Initialize 2nd FC layer for head (fc2)
numIn = numHiddenUnits4;
numOut = 32;
sz = [numOut numIn];
parameters.fc2.Weights = initializeGlorot(sz,numOut,numIn);
parameters.fc2.Bias = initializeZeros([numOut 1]);

% Initialize 3rd FC layer for head (fc3)
numIn = 32;
numOut = 1;
sz = [numOut numIn];
parameters.fc3.Weights = initializeGlorot(sz,numOut,numIn);
parameters.fc3.Bias = initializeZeros([numOut 1]);

trailingAvg = [];
trailingAvgSq = [];

% Loop over epochs.
for epoch = 1:numEpochs
    
    shuffle(mbq);
    reset(mbq);
        
    % Loop over mini-batches.
    while hasdata(mbq)
    
        iteration = iteration + 1;
        
        [dlX1,dlX2,T] = next(mbq);
        
        % Compute loss and gradients.
        [gradients,loss] = dlfeval(@modelGradients,parameters,state,dlX1,dlX2,T);

        % L2 regularization of weights
        gradients.fcpre.Weights = dlupdate(@(g,w) g + l2Regularization*w, gradients.fcpre.Weights, parameters.fcpre.Weights);
        gradients.gru_l1.Weights = dlupdate(@(g,w) g + l2Regularization*w, gradients.gru_l1.Weights, parameters.gru_l1.Weights);
        gradients.gru_l2.Weights = dlupdate(@(g,w) g + l2Regularization*w, gradients.gru_l2.Weights, parameters.gru_l2.Weights);
        gradients.gru_l3.Weights = dlupdate(@(g,w) g + l2Regularization*w, gradients.gru_l3.Weights, parameters.gru_l3.Weights);
        gradients.gru_h1.Weights = dlupdate(@(g,w) g + l2Regularization*w, gradients.gru_h1.Weights, parameters.gru_h1.Weights);
        gradients.gru_h2.Weights = dlupdate(@(g,w) g + l2Regularization*w, gradients.gru_h2.Weights, parameters.gru_h2.Weights);
        gradients.gru_h3.Weights = dlupdate(@(g,w) g + l2Regularization*w, gradients.gru_h3.Weights, parameters.gru_h3.Weights);
        gradients.gruhead.Weights = dlupdate(@(g,w) g + l2Regularization*w, gradients.gruhead.Weights, parameters.gruhead.Weights);
        gradients.gru_l1.RecurrentWeights = dlupdate(@(g,w) g + l2Regularization*w, gradients.gru_l1.RecurrentWeights, parameters.gru_l1.RecurrentWeights);
        gradients.gru_l2.RecurrentWeights = dlupdate(@(g,w) g + l2Regularization*w, gradients.gru_l2.RecurrentWeights, parameters.gru_l2.RecurrentWeights);
        gradients.gru_l3.RecurrentWeights = dlupdate(@(g,w) g + l2Regularization*w, gradients.gru_l3.RecurrentWeights, parameters.gru_l3.RecurrentWeights);
        gradients.gru_h1.RecurrentWeights = dlupdate(@(g,w) g + l2Regularization*w, gradients.gru_h1.RecurrentWeights, parameters.gru_h1.RecurrentWeights);
        gradients.gru_h2.RecurrentWeights = dlupdate(@(g,w) g + l2Regularization*w, gradients.gru_h2.RecurrentWeights, parameters.gru_h2.RecurrentWeights);
        gradients.gru_h3.RecurrentWeights = dlupdate(@(g,w) g + l2Regularization*w, gradients.gru_h3.RecurrentWeights, parameters.gru_h3.RecurrentWeights);
        gradients.gruhead.RecurrentWeights = dlupdate(@(g,w) g + l2Regularization*w, gradients.gruhead.RecurrentWeights, parameters.gruhead.RecurrentWeights);
        gradients.fc1.Weights = dlupdate(@(g,w) g + l2Regularization*w, gradients.fc1.Weights, parameters.fc1.Weights);
        gradients.fc2.Weights = dlupdate(@(g,w) g + l2Regularization*w, gradients.fc2.Weights, parameters.fc2.Weights);
        gradients.fc3.Weights = dlupdate(@(g,w) g + l2Regularization*w, gradients.fc3.Weights, parameters.fc3.Weights);

        % Update parameters using adamupdate.
        [parameters,trailingAvg,trailingAvgSq] = adamupdate(parameters,gradients,trailingAvg,trailingAvgSq,...
            iteration,learnRate,gradientDecayFactor,squaredGradientDecayFactor,10^(-9));
        
        % Display the training progress.
        D = duration(0,0,toc(start),'Format','hh:mm:ss');
        addpoints(lineLossTrain,iteration,sqrt(double(extractdata(loss))))
        title("Epoch: " + epoch + ", Elapsed: " + string(D))
        if size(dlXt1,2)>2048
            if mod(iteration,250)==0 || iteration==1 
                for i=1:2048:size(dlXt1,2)
                    dlYp(i:min(size(dlXt1,2),i+2047)) = model(parameters,state, ...
                                                             dlXt1(:,i:min(size(dlXt1,2),i+2047),:),...
                                                             dlXt2(:,i:min(size(dlXt1,2),i+2047),:),1);
                end
                YPred = double(extractdata(dlYp))';      
                YTest = double(extractdata(dlYt))'; 
                addpoints(lineLossTest,iteration,sqrt(mse(YPred,YTest)));
                res = [corr(YTest,YPred,'type','Spearman') ...
                       corr(YTest,YPred,'type','Pearson') ...
                       sqrt(mse(YTest,YPred))]     
            end
        else
             if mod(iteration,100)==0 || iteration==1 
                dlYp = model(parameters,state,dlXt1,dlXt2,1);
                YPred = double(extractdata(dlYp))';      
                YTest = double(extractdata(dlYt))'; 
                addpoints(lineLossTest,iteration,sqrt(mse(YPred,YTest)));
                res = [corr(YTest,YPred,'type','Spearman') ...
                       corr(YTest,YPred,'type','Pearson') ...
                       sqrt(mse(YTest,YPred))]     
            end           
        end
        drawnow
    end

    % Update learning rate
    learnRate = learnRate * learnRateDropFactor;
    learnRateDropFactor = learnRateDropFactor * 0.5;

end % end of training loop

mbq = [];


if size(dlXt1,2)<2048
    dlYp = model(parameters,state,dlXt1,dlXt2,0);
else
    for i=1:2048:size(dlXt1,2)
        dlYp(i:min(size(dlXt1,2),i+2047)) = model(parameters,state, ...
                                                 dlXt1(:,i:min(size(dlXt1,2),i+2047),:),...
                                                 dlXt2(:,i:min(size(dlXt1,2),i+2047),:),0);
    end
end
YPred = double(extractdata(dlYp))';      
YTest = double(extractdata(dlYt))'; 
size(YPred)
size(YTest)
result = [corr(YTest,YPred,'type','Spearman') ...
          corr(YTest,YPred,'type','Pearson') ...
          sqrt(mse(YTest,YPred))]
% 
% plot(YTest,YPred);

end


function [X1,X2,T] = preprocessMiniBatch(src,trg)
    X1 = {};
    X2 = {};
    for i=1:length(src)
        X1{i} = src{i}(:,src{i}(1,:)==1);
        X2{i} = src{i}(:,src{i}(1,:)==2);
    end
    X1 = padsequences(X1,2,'Length','longest','Direction','left','PaddingValue',0);
    X1 = permute(X1,[1 3 2]);
    X2 = padsequences(X2,2,'Length','longest','Direction','left','PaddingValue',0);
    X2 = permute(X2,[1 3 2]);
    T = cat(1,trg{:});
end

function [gradients,loss] = modelGradients(parameters,state,dlX1,dlX2,T)
    dlY = model(parameters, state, dlX1, dlX2, 1);
    loss = huber(dlY,T,'TransitionPoint',1/5); 
    
    % Update gradients.
    gradients = dlgradient(loss,parameters);
end

% ------------------------------------------------------------------------------------------
function [dlY] = model(parameters,state,dlX1,dlX2,training)

    % Regularization
    if training
        mult = dlarray(rand(1,1,1),'CBT');
        dlX1 = dlX1.*(mult.*state+1-state/2);
        mult = dlarray(rand(1,1,1),'CBT');
        dlX2 = dlX2.*(mult.*state+1-state/2);
    end

    % Prescaling by FC layer
    dlY1 = (dlX1(2:end,:,:));
    dlY2 = (dlX2(2:end,:,:));
    dlY1 = fullyconnect(dlY1,parameters.fcpre.Weights,parameters.fcpre.Bias);
    dlY1 = relu((dlY1));
    dlY2 = fullyconnect(dlY2,parameters.fcpre.Weights,parameters.fcpre.Bias);
    dlY2 = relu((dlY2));

    % Low resolution stream
    H0 = dlarray(zeros(size(parameters.gru_l1.Bias,1)/3,size(dlY1,2)),'CB');
    dlY1 = gru(dlY1,H0,parameters.gru_l1.Weights,parameters.gru_l1.RecurrentWeights,parameters.gru_l1.Bias);
    H0 = dlarray(zeros(size(parameters.gru_l2.Bias,1)/3,size(dlY1,2)),'CB');
    dlY1 = gru(dlY1,H0,parameters.gru_l2.Weights,parameters.gru_l2.RecurrentWeights,parameters.gru_l2.Bias);
    H0 = dlarray(zeros(size(parameters.gru_l3.Bias,1)/3,size(dlY1,2)),'CB');
    dlY1 = gru(dlY1,H0,parameters.gru_l3.Weights,parameters.gru_l3.RecurrentWeights,parameters.gru_l3.Bias);
    dlY1 = dlY1(:,:,end);

    % High resolution stream
    H0 = dlarray(zeros(size(parameters.gru_h1.Bias,1)/3,size(dlY1,2)),'CB');
    dlY2 = gru(dlY2,H0,parameters.gru_h1.Weights,parameters.gru_h1.RecurrentWeights,parameters.gru_h1.Bias);
    H0 = dlarray(zeros(size(parameters.gru_h2.Bias,1)/3,size(dlY2,2)),'CB');
    dlY2 = gru(dlY2,H0,parameters.gru_h2.Weights,parameters.gru_h2.RecurrentWeights,parameters.gru_h2.Bias);
    H0 = dlarray(zeros(size(parameters.gru_h3.Bias,1)/3,size(dlY2,2)),'CB');
    dlY2 = gru(dlY2,H0,parameters.gru_h3.Weights,parameters.gru_h3.RecurrentWeights,parameters.gru_h3.Bias);
    dlY2 = dlY2(:,:,end);

    % Concatenate streams
    dlY = cat(1,dlY1,dlY2);

    % Head
    H0 = dlarray(zeros(size(parameters.gruhead.Bias,1)/3,size(dlY,2)),'CB');
    dlY = gru(dlY,H0,parameters.gruhead.Weights,parameters.gruhead.RecurrentWeights,parameters.gruhead.Bias);
    dlY = stripdims(dlY);
    dlY = dlarray(dlY,'CB');
    dlY = relu(dlY);
    dlY = fullyconnect(dlY,parameters.fc1.Weights,parameters.fc1.Bias);
    dlY = relu(dlY);
    dlY = fullyconnect(dlY,parameters.fc2.Weights,parameters.fc2.Bias);
    dlY = relu(dlY);
    dlY = fullyconnect(dlY,parameters.fc3.Weights,parameters.fc3.Bias);
    dlY = sigmoid(dlY);

end


% ------------------------------------------------------------------------------------------
function weights = initializeGlorot(sz,numOut,numIn)

Z = 2*rand(sz,'single') - 1;
bound = sqrt(6 / (numIn + numOut));

weights = bound * Z;
weights = dlarray(weights);

end

function weights = initializeHe(sz,numIn)

weights = randn(sz,'single') * sqrt(2/numIn);
weights = dlarray(weights);

end

function parameter = initializeZeros(sz)

parameter = zeros(sz,'single');
parameter = dlarray(parameter);

end

function parameter = initializeOrthogonal(sz)

Z = randn(sz,'single');
[Q,R] = qr(Z,0);

D = diag(R);
Q = Q * diag(D ./ abs(D));

parameter = dlarray(Q);

end

function parameter = initializeZerosGru(sz)

parameter = zeros(sz,1,'single');
parameter = dlarray(parameter);

end
