clear

load train_data.mat;

%  GOURP OF CLASSES WITH THEIR CORRESPONDINGS SAMPLES
w_n=mitbihtrain1(70000:73999,1:187);
w_s=mitbihtrain1(72472:74694,1:187);
w_v=mitbihtrain1(74695:78694,1:187);


% OVERSAMPLING THE CLASSES WITH LESS NUMBER OF SAMPLES SO WE HAVE THE SAME
% AMOUNT OF SAMPLES IN EVERY CLASS. WE ARE GOING TO DO 4000 FOR EACH CLASS
w_s_add=mitbihtrain1(72472:74248,1:187);
w_s=[w_s ; w_s_add];



train_data=[w_n;w_s;w_v];

% WE NEED TO CREATE A CELL ARRAY, EACH ROW CORRESPONDS TO A SAMPLE, THE
% CELL ARRAY SIZE WOULD BE N-BY-1 WHERE N EQUALS TO THE NUMBER OF SAMPLES,
% IN OUR CASE 16000
dataTrain={};
[rows_train_data,columns_train_data]=size(train_data)

for i=1:rows_train_data
    dataTrain{i,1}=train_data(i,:);
end


label_n=zeros(4000,1);
label_s=ones(4000,1);
label_v=2*ones(4000,1);

train_labels_vector=[label_n ; label_s;label_v];

train_labels=num2cell(train_labels_vector);


for i=1:length(train_labels)
    if train_labels{i}==0
        train_labels{i}='N';
    end
    
    if train_labels{i}==1
        train_labels{i}='S';
    end
    
    if train_labels{i}==2
        train_labels{i}='V';
    end
    
    
end

layers = [ ...
    sequenceInputLayer(1)
    bilstmLayer(100,'OutputMode','last')
    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer
    ]

options = trainingOptions('adam', ...
    'MaxEpochs',10, ...
    'MiniBatchSize', 150, ...
    'InitialLearnRate', 0.01, ...
    'SequenceLength', 187, ...
    'GradientThreshold', 1, ...
    'ExecutionEnvironment',"auto",...
    'plots','training-progress', ...
    'Verbose',false);

train_labels=categorical(train_labels);
net = trainNetwork(dataTrain,train_labels,layers,options);

trainPred = classify(net,dataTrain,'SequenceLength',1000);


% COMPUTING THE ACCURACY OF OUR MODEL, % OF RIGHT EXPECTED VALUES WITH THE
% TRAINING DATA
LSTMAccuracy = sum(trainPred == train_labels)/numel(train_labels)*100

% COMPUTATION OF THE CONFUSSION MATRIX, BEAR IN MIND THAT THE SOME OF THE
% TRAINING DATA IS REPEATED BECAUSE WE WANTED TO HAVE THE SAME AMOUNT OF
% TRAINING SAMPLES FOR EACH CLASS AND IN THE CASE OF CLASS F COULD BE NOT
% REALISTIC THE PERCENTAGE THAT WE OBTAIN
figure
ccLSTM = confusionchart(train_labels,trainPred);
ccLSTM.Title = 'Confusion Chart for LSTM';
ccLSTM.ColumnSummary = 'column-normalized';
ccLSTM.RowSummary = 'row-normalized';





% WE PROCCED NOW TO TEST THE TEST DATA

load test_data.mat

% WE FIRST ARRANGE THE DATA TO USE IT FOR THE LSTM FUNCTIONS
test_n=mitbihtest(17000:18118,1:187);
test_s=mitbihtest(18119:18674,1:187);
test_v=mitbihtest(18675:20122,1:187);

test_labels=mitbihtest(17000:20284,188);

test_data=[test_n;test_s;test_v];



test_labels=num2cell(test_labels);
dataTest={};


for i=1:length(test_data)
    dataTest{i,1}=test_data(i,:);
end


for i=1:length(test_labels)
    if test_labels{i}==0
        test_labels{i}='N';
    end
    
    if test_labels{i}==1
        test_labels{i}='S';
    end
    
    if test_labels{i}==2
        test_labels{i}='V';
    end
    
end

test_labels=categorical(test_labels);

testPred = classify(net,dataTest,'SequenceLength',1000);
LSTMAccuracy = sum(testPred == test_labels)/numel(test_labels)*100

figure
ccLSTM = confusionchart(test_labels,testPred);
ccLSTM.Title = 'Confusion Chart for LSTM';
ccLSTM.ColumnSummary = 'column-normalized';
ccLSTM.RowSummary = 'row-normalized';






