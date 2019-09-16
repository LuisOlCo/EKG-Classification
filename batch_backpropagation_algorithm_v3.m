clear

load Matrix_Feature_Training_Data.mat

%TRAINING DATA
data=Matrix_Feature_Training_Data;
[r_data,c_data]=size(data);
label_class=data(:,c_data);
data=data(:,1:c_data-1);
data=transpose(data);
% the rows correspond to the features and the columns to the number of traning samples
[number_of_features,number_of_training_samples]=size(data);


%TARGET

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
   

%NORMALIZING TRAINING DATA

mu=[];
stdd=[];

for nor=1:number_of_features
    current_mean=mean(data(nor,:));
    mu=[mu;current_mean];
    a=std(data(nor,:));
    current_std=1./a;
    stdd=[stdd;current_std];
end

norm_data=(data-mu).*stdd;

[number_of_features,number_of_training_samples]=size(norm_data);

%SETTINGS OF OUR NEURAL NETWORK
nH=167;  % nH = number of hidden units
d=number_of_features;    % d = number of input nodes
c=3;    % c = number of classes of outputs
theta=0.16;  %theta = convergence criterion
eta=0.001;    % eta = Convergence rate 
 
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

r=0;

while (r<100)
    
       
    Update_Wkj=0; Update_Wji=0;
    for index_of_sample=1:number_of_training_samples
        
        chosen_pattern=norm_data(:,index_of_sample);
        chosen_target=target(:,index_of_sample);
 
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
    
        Update_Wkj=Update_Wkj + eta*(transpose(delta_k)*y);

        % Update for input-hidden weights
     
        Update_Wji=Update_Wji + eta*chosen_pattern*delta_j;
     
    end % findel rule del epoch
    
        
        
    Wkj=Wkj+Update_Wkj;
    Wji=Wji+Update_Wji;
     
    
    % Computation of the total training error
    
    J=0;
    for sample=1:number_of_training_samples
        Jp=0;
        chosen_sample=norm_data(:,sample);
        chosen_sample=[1;chosen_sample]; % Adding bias to the samples
        z=computation_result_z(chosen_sample,a,b,c,d,nH,Wji,Wkj);
        for output_unit=1:c
            Jp=Jp + 0.5*(target(output_unit,sample)-z(output_unit))^2;
        end
        J = J + Jp;
     
    end
    
    Error=J/number_of_training_samples
    E=[E Error];
    plot(E)
    title('Batch Backpropagation Learning Curve')
    ylabel('J/n')
    xlabel('Epochs')
    drawnow

  
    if (J/number_of_training_samples)< theta
        break
        
    end
   
    
    %r=r+1;
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
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



