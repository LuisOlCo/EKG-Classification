clear
load Matrix_Feature_Training_Data.mat
load Matrix_Feature_Validation_Data.mat
load Matrix_Feature_Test_Data.mat

%TRAINING DATA
training_data=Matrix_Feature_Training_Data;
[r_training_data,c_training_data]=size(training_data);
label_class_training=training_data(:,c_training_data);
training_data=training_data(:,1:c_training_data-1);
training_data=transpose(training_data);
% the rows correspond to the features and the columns to the number of traning samples
[number_of_features,number_of_training_samples]=size(training_data);

% Normalizing training data
norm_training_data=normalizing_data(training_data,number_of_features);

% Computation of the target vector for training samples
target_vector_training_data=target_vector(label_class_training);


%VALIDATION DATA
val_data=Matrix_Feature_Validation_Data;
[r_val_data,c_val_data]=size(val_data);
label_class_val=val_data(:,c_val_data);
val_data=val_data(:,1:c_val_data-1);
val_data=transpose(val_data);
% the rows correspond to the features and the columns to the number of validation samples
[number_of_features,number_of_validation_samples]=size(val_data);

% Normalizing validation data
norm_validation_data=normalizing_data(val_data,number_of_features);

% Computation of the target vector for validation samples
target_vector_validation_data=target_vector(label_class_val);


%TEST DATA
test_data=Matrix_Feature_Test_Data;
[r_test_data,c_test_data]=size(test_data);
label_class_test=test_data(:,c_test_data);
test_data=test_data(:,1:c_test_data-1);
test_data=transpose(test_data);
% the rows correspond to the features and the columns to the number of test samples
[number_of_features,number_of_test_samples]=size(test_data);

% Normalizing test data
norm_test_data=normalizing_data(test_data,number_of_features);

% Computation of the target vector for validation samples
target_vector_test_data=target_vector(label_class_test);





%SETTINGS OF OUR NEURAL NETWORK
nH=100;  % nH = number of hidden units
d=number_of_features;    % d = number of input nodes
c=3;    % c = number of classes of outputs
theta=0.7;  %theta = convergence criterion
eta=0.1;    % eta = Convergence rate 
 
% Parameters of the activation function, f=a*tanh(b*net)
a=1.716;
b=2/3;
 

% WEIGHTS INITIALIZATION 
 
% Wji matrix with all the weights between hidden and input layer, it is
% neccesary to initializate it, if it is equal to zero iteration does not
% progress
xmin=-1/(sqrt(d));
xmax=1/(sqrt(d));
Wji= xmin+rand(d+1,nH)*(xmax-xmin); % that plus one is to include the bias weight in the matrix too
 
 
 
% Wkj matrix with all the weights between output and hidden layer
xmin=-1/(sqrt(nH));
xmax=1/(sqrt(nH));
Wkj=xmin+rand(c,nH+1)*(xmax-xmin); % that plus one is to include the bias weight in the matrix too
E=[];
E_validation=[];
E_test=[];
r=0;

while (r<100)
    
       
    Update_Wkj=0; Update_Wji=0;
    
        
    % Picking a random sample 
    randomIndex = randi(length(norm_training_data));
    chosen_pattern=norm_training_data(:,randomIndex);
    chosen_target=target_vector_training_data(:,randomIndex);
    
    % We have to add the bias to this chosen pattern to include bias in our
    % calculation
    chosen_pattern=[1; chosen_pattern];
    
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
    
    
    
    % Computation of the sensitivity delta_k
    delta_k=[];
    delta_k=(transpose(chosen_target)-z).*f_netk_derivative;
    
    
    % Computation of the sensitivity delta_j
    delta_j=[];
    for i=1:nH
        delta_j(i)=f_netj_derivative(i)*dot(Wkj(:,i),delta_k);
    end
    
    
    % Update for hidden-output weights
    
    Update_Wkj=eta*(transpose(delta_k)*y);

    % Update for input-hidden weights
     
    Update_Wji=eta*chosen_pattern*delta_j;
     
    Wkj=Wkj+Update_Wkj;
    Wji=Wji+Update_Wji;
     
    
    % Computation of the total training error
    
    J=0;
    for sample=1:number_of_training_samples
        Jp=0;
        chosen_sample=norm_training_data(:,sample);
        chosen_sample=[1;chosen_sample]; % Adding bias to the samples
        z=computation_result_z(chosen_sample,a,b,c,d,nH,Wji,Wkj);
        for output_unit=1:c
            Jp=Jp + 0.5*(target_vector_training_data(output_unit,sample)-z(output_unit))^2;
        end
        J = J + Jp;
     
    end
    
    % Computation of the total validation error
    
    J_validation=0;
    for sample=1:number_of_validation_samples
        Jp=0;
        chosen_sample=norm_validation_data(:,sample);
        chosen_sample=[1;chosen_sample]; % Adding bias to the samples
        z=computation_result_z(chosen_sample,a,b,c,d,nH,Wji,Wkj);
        for output_unit=1:c
            Jp=Jp + 0.5*(target_vector_validation_data(output_unit,sample)-z(output_unit))^2;
        end
        J_validation = J_validation + Jp;
     
    end
    
    
    % Computation of the total test error
    
    J_test=0;
    for sample=1:number_of_test_samples
        Jp=0;
        chosen_sample=norm_test_data(:,sample);
        chosen_sample=[1;chosen_sample]; % Adding bias to the samples
        z=computation_result_z(chosen_sample,a,b,c,d,nH,Wji,Wkj);
        for output_unit=1:c
            Jp=Jp + 0.5*(target_vector_test_data(output_unit,sample)-z(output_unit))^2;
        end
        J_test = J_test + Jp;
     
    end
    
    Error=J/number_of_training_samples
    E=[E Error];
    
    Error_validation=J_validation/number_of_validation_samples;
    E_validation=[E_validation Error_validation];
    
    Error_test=J_test/number_of_test_samples;
    E_test=[E_test Error_test];
    
    figure(1)
    plot(E)
    hold on
    plot(E_validation)
    hold on
    plot(E_test)
    hold off
    title('Stochastic Backpropagation Learning Curve')
    xlabel('Number of samples randomly used by the algorithm')
    ylabel('J/n')
    legend('Training','Validation','Test')
    drawnow
    
    figure(2)
    plot(E)
    hold on
    plot(E_validation)
    hold off
    title('Stochastic Backpropagation Learning Curve')
    xlabel('Number of samples randomly used by the algorithm')
    ylabel('J/n')
    legend('Training','Validation')
    drawnow

  
    if (J/number_of_training_samples)< theta
        break
        
    end
   
   
    %r=r+1;
end



save Wkj.mat
save Wji.mat

% This function returns a matrix with all target arrays for each sample 
% given the label_class_vector
function target=target_vector(label_class)
    target=[];
    for h=1:length(label_class)
        if label_class(h)==1
            target(:,h)=[1;-1;-1];
        elseif label_class(h)==2
            target(:,h)=[-1;1;-1];
        elseif label_class(h)==3
            target(:,h)=[-1;-1;1];
        end
    end
end

   
% This function gets any data and normalize it before being processed by
% the Neural Network.
function norm_data=normalizing_data(data,number_of_features)
    mu=[];
    stdd=[];
    norm_data=[];

    for nor=1:number_of_features
        current_mean=mean(data(nor,:));
        mu=[mu;current_mean];
        a=std(data(nor,:));
        current_std=1./a;
        stdd=[stdd;current_std];
    end

    %norm_data=(data-mu).*stdd;
    norm_data=(data-mu).*stdd;
    %[number_of_features,number_of_training_samples]=size(norm_data);

end



    
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
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



