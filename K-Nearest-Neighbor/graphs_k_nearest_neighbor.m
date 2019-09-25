% This program plots the pictures used or the explanation of the
% K-nearest-neighbor algorithm


clear

load train_data.mat


w_n=mitbihtrain1(70000:70640,1:187);
w_n1=mitbihtrain1(70000:70010,1:187);
w_s=mitbihtrain1(72472:73112,1:187);
w_s1=mitbihtrain1(72472:72473,1:187);
w_v=mitbihtrain1(74695:75335,1:187);
w_v1=mitbihtrain1(74695:74704,1:187);


w_n2=mitbihtrain1(70000:70030,1:20);
[r,c]=size(w_n2);


w_n1=mitbihtrain1(70000:70005,1:20);
w_s1=mitbihtrain1(72472:72477,1:20);
w_v1=mitbihtrain1(74695:74700,1:20);

[r_w_n,c_w_n]=size(w_n1);
x=1:1:c_w_n;
figure(1)
for i=1:r_w_n
    
    scatter(x,w_n1(i,:),'*','black')
    hold on
    scatter(x,w_s1(i,:),'*','green')
    hold on
    scatter(x,w_v1(i,:),'*','red')
    hold on
end
hold off

figure(2)
t=1:1:c;
for i=1:r

    scatter(t,w_n2(i,:),'*','black')
    hold on
end
hold off




