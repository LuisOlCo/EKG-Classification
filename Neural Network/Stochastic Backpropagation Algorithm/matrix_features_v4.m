% This program is the same as matrix_features_train_test_data_v2, but it
% has been modified so we can add a validation curve along with the
% training cuve, for that, we will remove the 20% of the training data
% aside for validation

clear
load train_data.mat



% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% We are going to create a matrix with the different features corresponding
% to each sample, the matrix will have n rows corresponding to the number
% of samples used and it will have b columns, corresponding to the number
% of fatures of each sample
%1. mean
%2. length
%3. (pk1-mean)/mean
%4. (pk2-mean)/mean
%5. (pk3-mean)/mean
%6. (dPk1-dPk2)/length
%7. (dPk1-dPk2)/length
%8. (dPk2-dPk3)/length

f1_mean=[];
f1_1_mode=[];
f2_length=[];
f3_pk1_mean=[];
f4_pk2_mean=[];
f5_pk3_mean=[];
f6_dpk1_dpk2=[];
f7_dpk1_dpk3=[];
f8_dpk2_dpk3=[];
f9_pk1_mean_2=[];
f10_pk2_mean_2=[];
f11_pk3_mean_2=[];
f12_peak1=[];
f13_peak2=[];
f14_peak3=[];


% TRAINING DATA: 4000 samples

train_data_o=mitbihtrain1(60000:63999,:);
% Samples of class 1 are repeated because we want to have the same amount
% of training samples for each class.
train_data_1=[mitbihtrain1(72472:73999,:);mitbihtrain1(72472:73999,:);mitbihtrain1(72472:73416,:)];
train_data_2=mitbihtrain1(74695:78694,:);
training_data=[train_data_o;train_data_1;train_data_2];
[number_training_samples,number_points_per_sample]=size(training_data);

label_class=[];
for pos=1:number_training_samples
    
    if training_data(pos,188)==0
        label_class(pos)=1;
    elseif training_data(pos,188)==1
        label_class(pos)=2;
    elseif training_data(pos,188)==2
        label_class(pos)=3;
    elseif training_data(pos,188)==3
        label_class(pos)=4;
    end
end

label_class=transpose(label_class);
training_data=training_data(:,1:187);
[number_training_samples,number_points_per_sample]=size(training_data);

% VALIDATION DATA: 695 samples

validation_data_o=mitbihtrain1(64000:64694,:);
validation_data_1=mitbihtrain1(74000:74694,:);
validation_data_2=mitbihtrain1(79000:79694,:);
validation_data=[validation_data_o;validation_data_1;validation_data_2;];
[number_validation_samples,number_points_per_sample]=size(validation_data);


label_class_validation=[];
for pos=1:number_validation_samples
    
    if validation_data(pos,188)==0
        label_class_validation(pos)=1;
    elseif validation_data(pos,188)==1
        label_class_validation(pos)=2;
    elseif validation_data(pos,188)==2
        label_class_validation(pos)=3;
    elseif validation_data(pos,188)==3
        label_class_validation(pos)=4;
    end
end

label_class_validation=transpose(label_class_validation);
validation_data=validation_data(:,1:187);
[number_validation_samples,number_points_per_sample]=size(validation_data);


% TEST DATA

load test_data.mat

test_data_o=mitbihtest(17000:18118,:);
test_data_1=mitbihtest(18119:18674,:);
test_data_2=mitbihtest(18675:20122,:);
test_data=[test_data_o;test_data_1;test_data_2];
[number_test_samples,number_points_per_test_sample]=size(test_data);


label_class_test_data=[];
for pos=1:number_test_samples
    
    if test_data(pos,188)==0
        label_class_test_data(pos)=1;
    elseif test_data(pos,188)==1
        label_class_test_data(pos)=2;
    elseif test_data(pos,188)==2
        label_class_test_data(pos)=3;
    elseif test_data(pos,188)==3
        label_class_test_data(pos)=4;
    end
end

label_class_test_data=transpose(label_class_test_data);
test_data=test_data(:,1:187);
[number_test_samples,number_points_per_test_sample]=size(test_data);



%f1_mean f2_length f3_pk1_mean f4_pk2_mean f5_pk3_mean f6_dpk1_dpk2 f7_dpk1_dpk3 f8_dpk2_dpk3 f9_pk1_mean_2 f10_pk2_mean_2 f11_pk3_mean_2 f12_peak1 f13_peak2 f14_peak3


Matrix_Feature_Training_Data=[];
for i=1:number_training_samples
    Matrix_Feature_Training_Data(i,:)=Feature_Extractor(training_data(i,:));
end

Matrix_Feature_Training_Data=[Matrix_Feature_Training_Data label_class];
save Matrix_Feature_Training_Data.mat



Matrix_Feature_Validation_Data=[];
for i=1:number_validation_samples
    Matrix_Feature_Validation_Data(i,:)=Feature_Extractor(validation_data(i,:));
end

Matrix_Feature_Validation_Data=[Matrix_Feature_Validation_Data label_class_validation];
save Matrix_Feature_Validation_Data.mat



Matrix_Feature_Test_Data=[];
for i=1:number_test_samples
    Matrix_Feature_Test_Data(i,:)=Feature_Extractor(test_data(i,:));
end

Matrix_Feature_Test_Data=[Matrix_Feature_Test_Data label_class_test_data];
save Matrix_Feature_Test_Data.mat



% 
function row_sample=Feature_Extractor(sample)
    
    f1_mean=[];f1_1_mode=[];f2_length=[];f3_pk1_mean=[];f4_pk2_mean=[];f5_pk3_mean=[];f6_dpk1_dpk2=[];f7_dpk1_dpk3=[];f8_dpk2_dpk3=[];f9_pk1_mean_2=[];
    f10_pk2_mean_2=[];f11_pk3_mean_2=[];f12_peak1=[];f13_peak2=[];f14_peak3=[];
    
    % Eliminate all the points with value zero after the last non zero
    % value
    sample_no_zeros=sample(find(sample,1,'first'):find(sample,1,'last'));
    m=0;
    l=0;
    
    % Feature 1: mean
    m=mean(sample_no_zeros);
    f1_mean=[f1_mean;m];
    
    % Feature 1.1: mode
    mod=mode(sample_no_zeros);
    f1_1_mode=[f1_1_mode;mod];
    
    % Feature 2: length
    l=length(sample_no_zeros);
    f2_length=[f2_length;l];
    
       
    [pks,locs,width,prominence] = findpeaks(sample_no_zeros);
    [value_peaks,index_peaks]=maxk(pks,3);
    
    if length(value_peaks)==0
        val1=0;
        val2=0;
        val3=0;
    elseif length(value_peaks)==1 
         val1=value_peaks(1);
         val2=value_peaks(1);
         val3=value_peaks(1);
     elseif length(value_peaks)==2
         val1=value_peaks(1);
         val2=value_peaks(2);
         val3=value_peaks(2);
    else
         val1=value_peaks(1);
         val2=value_peaks(2);
         val3=value_peaks(3);
     end    
    
        
   % Feature 3: Peak 1 - mean
     f3=val1/m;
     f3_pk1_mean=[f3_pk1_mean;f3];
     
        
   % Feature 4: Peak 2 - mean
     f4=val2/m;
     f4_pk2_mean=[f4_pk2_mean;f4];
     
        
   % Feature 5: Peak 3 - mean
     
     f5=val3/m;
     f5_pk3_mean=[f5_pk3_mean;f5];
    
     if length(index_peaks)==0
         d1=0;d2=0;d3=0;
     elseif length(index_peaks)==1 
         d1=index_peaks(1);
         d2=index_peaks(1);
         d3=index_peaks(1);
     elseif length(index_peaks)==2
         d1=index_peaks(1);
         d2=index_peaks(2);
         d3=index_peaks(2);
     elseif length(index_peaks)==3
         d1=index_peaks(1);
         d2=index_peaks(2);
         d3=index_peaks(3);
     end
     
    
     
    % Feature 6: DPk1-DPk2
    
    d12=(d1-d2)/l;
    f6_dpk1_dpk2=[f6_dpk1_dpk2;d12];
    
    
    % Feature 7: DP1-DPk3   
    
    d13=(d1-d3)/l;
    f7_dpk1_dpk3=[f7_dpk1_dpk3;d13];
    
    
    % Feature 8: DP2-DPk3
    
    d23=(d2-d3)/l;
    f8_dpk2_dpk3=[f8_dpk2_dpk3;d23];   
    
    
    % Feature 9: pk1_mean_2;
    
    f9=val1*val2;
    f9_pk1_mean_2=[f9_pk1_mean_2;f9];
    
    
    % Feature 10: pk2_mean_2;
    
    f10=val1*val3;
    f10_pk2_mean_2=[f10_pk2_mean_2;f10];
    
    % Feature 11: pk3_mean_2;
    
    f11=val2*val3;
    f11_pk3_mean_2=[f11_pk3_mean_2;f11];

    % Feature 12: pk1_peak1;
    
    f12=val1*val1;
    f12_peak1=[f12_peak1;f12];
    
    % Feature 13: pk2_peak2;
    
    f13=val2*val2;
    f13_peak2=[f13_peak2;f13];
    
    % Feature 14: pk3_peak3;
    
    f14=val3*val3;
    f14_peak3=[f14_peak3;f14];
    
    row_sample=[f1_mean,f2_length,f3_pk1_mean,f4_pk2_mean,f5_pk3_mean,f6_dpk1_dpk2,f7_dpk1_dpk3,f8_dpk2_dpk3];
   

end
