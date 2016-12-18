clear
load('theta_k_10_es_3_lr_0.010_maxepoch_500.mat','thetas');
c=thetas(1,:);
c=c(2:end)';
b=thetas(:,1);
b=b(2:end);
W=thetas(2:end,2:end);
X=zeros(size(W));
for t=1:100
    random_image=binornd(1,0.5,784,1);
    x_prev=random_image;
    for k=1:1000
        [j,h_x,x_bar]=sampler_unif(x_prev,W,b,c);
        x_prev=x_bar;
    end
    X(t,:)=x_bar';
end
fname=strcat('5c','theta_k_10_es_3_lr_0.010_maxepoch_500','.png')
displayData(X,fname);  