% Code for classifying the samples once we have the result from the Neural
% Network

clear

load Wkj.mat
load Wji.mat
load Matrix_Feature_Test_data.mat

test_data=Matrix_Feature_Test_Data;
[r_test_data,c_test_data]=size(test_data);
label_class_test_data=test_data(:,c_test_data);
test_data=test_data(:,1:c_test_data-1);
test_data=transpose(test_data);% the rows correspond to the features and the columns to the number of samples


% IT IS NECESSARY TO NORMALIZE THE DATA BEFORE USING IT AS INPUT IN OUR NN
mu_test=[];
stdd_test=[];

for nor=1:number_of_features
    
    mu_test=[mu_test;mean(test_data(nor,:))];
    a_test=std(test_data(nor,:));
    current_std_test=1./a_test;
    stdd_test=[stdd_test;current_std_test];
end
%because we are not normalizing we assign to norm_data our matrix
test_data=(test_data-mu_test).*stdd_test;

[r,number_of_test_samples]=size(test_data);
bias=ones(1,number_of_test_samples);
test_data=[bias;test_data];


% COMPUTATION OF THE OUPUT BY THE NEURAL NETWORK FOLLOWING THE
% FEEDFORWARD OPERATION

z_results=[];

for p=1:number_of_test_samples
    z_re=computation_result_z(test_data(:,p),a,b,c,d,nH,Wji,Wkj);
    z_results=[z_results; z_re];
end


[number_of_results,output_unit_value]=size(z_results);

% ONCE WE HAVE THE OUPUT BECAUSE THE NUMBER AT THE OUTPUT WILL BE UNLIKELY
% CLOSE TO 1 OR -1, WE NEED TO ROUND THEM SO WE CAN MAKE THE CLASSIFICATION
% CLEAR


%  METHOD FOR COMPUTING THE EFFIECIENCY
    
   for i=1:number_of_results
    a1=1-z_results(i,1);
    a2=1-z_results(i,2);
    a3=1-z_results(i,3);
    
    b1=-1-z_results(i,1);
    b2=-1-z_results(i,2);
    b3=-1-z_results(i,3);
    
    a=[a1 a2 a3];
    b=[b1 b2 b3];
    
    %dif1=(a1^2)+(b2^2)+(b3^2);
    %dif2=(a2^2)+(b1^2)+(b3^2);
    %dif3=(a3^2)+(b2^2)+(b1^2);
    
    dif=[sqrt((a1^2)+(b2^2)+(b3^2)), sqrt((a2^2)+(b1^2)+(b3^2)), sqrt((a3^2)+(b2^2)+(b1^2))];
    
    [value_2,index_2]=min(dif);
    
    if index_2==1
        z_results_label2(i)=1;
    elseif index_2==2
        z_results_label2(i)=2;
    elseif index_2==3
        z_results_label2(i)=3;
    end
    
   end
 

z_results_label2=transpose(z_results_label2);

m11=0;m12=0;m13=0;
m21=0;m22=0;m23=0;
m31=0;m32=0;m33=0;

for i=1:length(z_results_label2)
    if label_class_test_data(i)==1 && z_results_label2(i)==1
        m11=m11+1;
    elseif label_class_test_data(i)==2 && z_results_label2(i)==1
        m12=m12+1;
    elseif label_class_test_data(i)==3 && z_results_label2(i)==1
        m13=m13+1;
        
    elseif label_class_test_data(i)==1 && z_results_label2(i)==2
        m21=m21+1;
    elseif label_class_test_data(i)==2 && z_results_label2(i)==2
        m22=m22+1;
    elseif label_class_test_data(i)==3 && z_results_label2(i)==2
        m23=m23+1;
        
    elseif label_class_test_data(i)==1 && z_results_label2(i)==3
        m31=m31+1;
    elseif label_class_test_data(i)==2 && z_results_label2(i)==3
        m32=m32+1;
    elseif label_class_test_data(i)==3 && z_results_label2(i)==3
        m33=m33+1;
    end
   
end
counter2=0;
for i=1:length(z_results_label2)
    if z_results_label2(i)==label_class_test_data(i)
        counter2=counter2+1;
    end
 end


efficiency=counter2/length(z_results_label2)
confussion_matrix=[m11 m12 m13;m21 m22 m23;m31 m32 m33]
    
    
    
    % Computation of the Neural Network output Z and derivatives for
    % updates calculation
    % FEEDFORWARD OPERATION
        
    function z=computation_result_z(chosen_pattern,a,b,c,d,nH,Wji,Wkj)
    
         % Computation of netj, net activation input-hidden units
        netj=[];
        for i=1:nH
            netj(i)=dot((Wji(:,i)),chosen_pattern);
        end   
    
        % Computation of hidden units outputs (activation function and its
        % derivative)
        y=[];
        for i=1:nH
            y(i)=a*(tanh(b*netj(i)));
        end
    
        f_netj_derivative=[];
        for i=1:nH
            f_netj_derivative(i)=a*b*(1-(tanh(b*netj(i))^2));
        end
    
        % Computation of netk, net activation hidden-output units
        y=[1 y]; % bias it also go to the output units, so we need to include them again in this operation
        netk=[];
        for i=1:c
            netk(i)=dot((Wkj(i,:)),y);
        end 
    
        % Computation of the activation function corresponding to the output
        % units and it derivative
   
        z=[];
        for i=1:c
            z(i)=a*(tanh(b*netk(i)));
        end
    
        f_netk_derivative=[];
        for i=1:c
            f_netk_derivative(i)=a*b*(1-(tanh(b*netk(i))^2));
        end
        
    end
    