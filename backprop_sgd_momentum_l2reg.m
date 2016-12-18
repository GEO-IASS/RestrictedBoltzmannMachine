clear
trainset=load('digitstrain.txt');

valset=load('digitsvalid.txt');
Xval=valset(:,1:size(valset,2)-1);
Xval=double(Xval>=0.5);
yval=valset(:,size(valset,2));

input_layer_size=784;
hidden_layer_size=100;
num_labels=10;
m=size(trainset,1);
max_epochs=100;
learning_rate=0.1;
beta=0;
epsilon=0.001;
lambda=0;
%lambda=0.0001;
track_train_cost=zeros(max_epochs,1);
track_train_accuracy=zeros(max_epochs,1);
track_val_cost=zeros(max_epochs,1);
track_val_accuracy=zeros(max_epochs,1);
track_wts=zeros(max_epochs,1);
wtsname='dae_theta_es_5_lr_0.010_maxepoch_100';
load('dae_theta_es_5_lr_0.010_maxepoch_100.mat','thetas');
b=thetas(:,1);
b=b(2:end);
W=thetas(2:end,2:end);

hidden_wts=[b';W'];
layer2_wts=initialize_wts(hidden_layer_size+1,num_labels);
tic;
prev_J_train=100;
convergence_epoch=0;

for epoch=1:max_epochs
    shuffled_trainset=trainset(randperm(m),:);
    Xtrain=shuffled_trainset(:,1:size(shuffled_trainset,2)-1);
    Xtrain=double(Xtrain>=0.5);
    ytrain=shuffled_trainset(:,size(shuffled_trainset,2));
    X=[ones(m,1) Xtrain];
    Y=zeros(m,num_labels);
    for c=1:num_labels,
        Y(:,c)= (ytrain==c-1);
    end
    J=0;
    prev_hidden_grad_term=zeros(size(hidden_wts));
    prev_layer2_grad_term=zeros(size(layer2_wts));
    for t=1:m
        a1=X(t,:);
        z2=a1*hidden_wts;
        a2=sigmoid(z2);
        a2=[1 a2];
        z3=a2*layer2_wts;
        sfmax=exp(z3);
        a3=sfmax./sum(sfmax);
        yt=Y(t,:);
        J=J-(yt*transpose(log(a3)));
        del3=a3-yt;
        z2=[0 z2];
        del2=(del3*transpose(layer2_wts)).*sigmoidGradient(z2);
        del2=del2(2:end);
        hidden_l2_reg=hidden_wts(2:end,:);
        hidden_l2_reg=[zeros(1,size(hidden_l2_reg,2));hidden_l2_reg];
        layer2_l2_reg=layer2_wts(2:end,:);
        layer2_l2_reg=[zeros(1,size(layer2_l2_reg,2));layer2_l2_reg];
        layer2_grad_term=transpose(a2)*del3+lambda*layer2_l2_reg;
        hidden_grad_term=transpose(a1)*del2+lambda*hidden_l2_reg;
        current_layer2_grad_term=layer2_grad_term+beta.*prev_layer2_grad_term;
        current_hidden_grad_term=hidden_grad_term+beta.*prev_hidden_grad_term;
        hidden_wts=hidden_wts-learning_rate.*current_hidden_grad_term;
        layer2_wts=layer2_wts-learning_rate.*current_layer2_grad_term;
    end
    [J_train,accuracy_train]=predict(Xtrain,ytrain,hidden_wts,layer2_wts);
    track_train_cost(epoch)=J_train;
    track_train_accuracy(epoch)=accuracy_train;
    theta=[hidden_wts(:);layer2_wts(:)];
    track_wts(epoch)=sqrt(sum(theta.^2));
    [J_val,accuracy_val]=predict(Xval,yval,hidden_wts,layer2_wts);
    track_val_cost(epoch)=J_val;
    track_val_accuracy(epoch)=accuracy_val;
    if (mod(epoch,50)==0)
        epoch
    end
    if prev_J_train-J_train<=epsilon && convergence_epoch==0
        convergence_epoch=epoch
    elseif prev_J_train-J_train>epsilon && epoch==max_epochs
            max_epochs=max_epochs+1
    end
    prev_J_train=J_train;
end        
toc
elapsedTime=toc;
elapsedTime=elapsedTime/60
to_save_to_file=[convergence_epoch;zeros(max_epochs-1,1)]; 
output=[to_save_to_file track_train_cost track_train_accuracy track_wts track_val_cost track_val_accuracy];
fname=strcat('sgd_bin_',wtsname,'.csv');
%fname=strcat('sgd_l2reg_%d.csv',max_epochs)
csvwrite(fname,output);
%to_visualize=hidden_wts(2:end,:);
%displayData(transpose(to_visualize));