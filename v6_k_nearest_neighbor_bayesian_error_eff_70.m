% K-Nearest-Neighbor & Bayesian Decision Rule

clear

load train_data.mat;
load test_data.mat;

%  GOURP OF CLASSES WITH THEIR CORRESPONDINGS SAMPLES, we have only take
%  640 samples of each class.
%w_n=mitbihtrain1(70000:70640,1:187);
%w_s=mitbihtrain1(72472:73112,1:187);
%w_v=mitbihtrain1(74695:75335,1:187);

w_n=mitbihtrain1(70000:72000,1:187);
[r_w_n,c_w_n]=size(w_n);
w_s=mitbihtrain1(72472:74471,1:187);
[r_w_s,c_w_s]=size(w_s);
w_v=mitbihtrain1(74695:76694,1:187);
[r_w_v,c_w_v]=size(w_v);


%[n,points]=size(w_n);
n=r_w_n+r_w_s+r_w_v;
%k=round(sqrt(n));
%  All training data has to have the same amount of samples for each class



test_data=mitbihtest(17500:20122,1:187);%17500:20122

[r_test_data,c_test_data]=size(test_data);
class_test_sample=mitbihtest(17500:20122,188)+1;
predicted_class=[];



%for j=1:r_test_data
for j=1:r_test_data
    
    test_sample=test_data(j,:);
    

    Prob_matrix=[]; % It is a matrix that represents the probability of one point of the sample to be part of one of the classes, let's say that test sample 1 
    % in time point 1 has four diferent probabilities for each class,
    % whichever probablityt is higher, then for that point that sample
    % belongs to that class that had higher probability. So we can say that
    % there is on Prob_matrix for each test sample.
    
    % Now we compute for test sample, and in this sample at especific point
    % inside of this sample the distance between the test sample and all
    % the training samples, those distance are gathered in one vector
    % called total_distances
    
    for i=1:length(test_sample)
        distance_n=[];distance_s=[];distance_v=[];distance_f=[];
        distance_n=distance_class(test_sample(i),w_n(:,i));
        distance_s=distance_class(test_sample(i),w_s(:,i));
        distance_v=distance_class(test_sample(i),w_v(:,i));
        %distance_f=distance_class(test_sample(i),w_f(:,i));
        %total_distance=[distance_n distance_s distance_v distance_f];
        total_distance=[distance_n distance_s distance_v];



%Now we are looking to find the number of k nearest samples belong to each
%class, the problem is that sometimes we have that we have a bigger number
%of k distances equal to zero then we have this condition, in the casr we
%have many distances equal to zero we will get as many ks as zeros
%distances are. In case there is no distances equal to zero, we will follow
%the standard procedure, we get the closets sqrt(n) elements and we
%estimate how many of them belong to a certain class,  to obtain the
%probability of that point that belongs to a test sample of belonging to a
%certain class
        
        nzeros=[numel(distance_n)-nnz(distance_n) numel(distance_s)-nnz(distance_s) numel(distance_v)-nnz(distance_v)];
        
        if nzeros>round(sqrt(n))
            k=sum(nzeros);
            Prob_matrix(:,i)=nzeros/k;
        else
            k=round(sqrt(n));
            total_distance=[distance_n distance_s distance_v distance_f];
            [values,index]=mink(total_distance,k);
            k_n=0;k_s=0;k_v=0;k_f=0;
            
            for p=1:length(index)
                
                if index(p)<=r_w_n
                    k_n=k_n+1;
                    
                elseif r_w_n+1 <= index(p) && index(p)<= r_w_n+r_w_s+1
                    k_s=k_s+1;
                    
                elseif r_w_n+r_w_s+2 <= index(p) && index(p)<= r_w_n+r_w_s+2+r_w_v
                    k_v=k_v+1;
                end
                
 
            end
            
            Prob_matrix(:,i)=[k_n k_s k_v]/k;
            
        end
        

    
    end
    
    
% Once we have the matrix with all the posteriori porbabilities regarding
% the test sample we need to decide which class entails the less risk

    [r_Prob_matrix,c_Prob_matrix]=size(Prob_matrix);

% Now we create a matrix with the error for each selection of the class 
% and for every time point of the test sample   
    
    error_matrix=ones(r_Prob_matrix,c_Prob_matrix) - Prob_matrix;
    total_error=sum(error_matrix,2)/sum(error_matrix,'all');
    
    [min_error,index_error]=min(total_error)

% the most probable class for each point
    predicted_class(j)=index_error;
    
    

end


m11=0;m12=0;m13=0;
m21=0;m22=0;m23=0;
m31=0;m32=0;m33=0;

for i=1:length(predicted_class)
    if class_test_sample(i)==1 && predicted_class(i)==1
        m11=m11+1;
    elseif class_test_sample(i)==2 && predicted_class(i)==1
        m12=m12+1;
    elseif class_test_sample(i)==3 && predicted_class(i)==1
        m13=m13+1;
        
    elseif class_test_sample(i)==1 && predicted_class(i)==2
        m21=m21+1;
    elseif class_test_sample(i)==2 && predicted_class(i)==2
        m22=m22+1;
    elseif class_test_sample(i)==3 && predicted_class(i)==2
        m23=m23+1;
        
    elseif class_test_sample(i)==1 && predicted_class(i)==3
        m31=m31+1;
    elseif class_test_sample(i)==2 && predicted_class(i)==3
        m32=m32+1;
    elseif class_test_sample(i)==3 && predicted_class(i)==3
        m33=m33+1;
    end
   
end





counter=0;
for i=1:length(predicted_class)
    if predicted_class(i)==class_test_sample(i)
        counter=counter+1;
    end
end
confussion_matrix=[m11 m12 m13;m21 m22 m23;m31 m32 m33]
eff=counter/length(predicted_class)
% Computation of the size of the window for each class
function V=size_window(distance,k,test_sample,training_samples)

    [B,I]=mink(distance,k);
    vector=[test_sample];
    
    for i=1:k
        vector=[vector training_samples(I((i)))];
    end
    
    min_val_window=min(vector);
    max_val_window=max(vector);
    V=max_val_window-min_val_window;

end

% compute the distances between the test sample and the training samples
function distances=distance_class(test_sample,training_samples)
    [r,c]=size(training_samples);
    distances=[];
    for i=1:r
    distances(i)=abs(test_sample-training_samples(i));
    end
    

end









