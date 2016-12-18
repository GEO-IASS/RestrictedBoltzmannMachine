clear
trainset=load('digitstrain.txt');

valset=load('digitsvalid.txt');
Xval=double(valset(:,1:size(valset,2)-1)>=0.5);

input_layer_size=784;
hidden_layer_size=100;
m=size(trainset,1);

max_epochs=500;
early_stopper_limit=3;
learning_rate=0.01;
CD=10;

track_train_error=zeros(max_epochs,1);
track_val_error=zeros(max_epochs,1);
track_wts=zeros(max_epochs,1);
W=initialize_wts(hidden_layer_size,input_layer_size);
c=zeros(input_layer_size,1);
b=zeros(hidden_layer_size,1);

tic;
prev_J_val=10000;
W_final=zeros(size(W));
c_final=zeros(size(c));
b_final=zeros(size(b));
early_stopper=0;
convergence_flag=0;

detail_string=sprintf('k_%d_es_%d_lr_%.3f_maxepoch_%d',CD,early_stopper_limit,learning_rate,max_epochs);
x_prev=binornd(1,0.5,1,size(trainset,2)-1)';
for epoch=1:max_epochs
    shuffled_trainset=trainset(randperm(m),:);
    Xtrain=shuffled_trainset(:,1:size(shuffled_trainset,2)-1);
    ytrain=shuffled_trainset(:,size(shuffled_trainset,2));
    X=double(Xtrain>=0.5);
    for t=1:m
        x=X(t,:)';
        %x_prev=x;
        [j,first_h,x_data]=sampler_unif(x,W,b,c);
        for k=1:CD
            [j,h,x_bar]=sampler_unif(x_prev,W,b,c);
            %if k==1
            %    first_h=h;
            %end
            x_prev=x_bar;
        end
        [j_not_req,h_bar,x_not_req]=sampler_unif(x_bar,W,b,c);
        x_prev=x_not_req;
        W=W+learning_rate*(first_h*x'-h_bar*x_bar');
        b=b+learning_rate*(first_h-h_bar);
        c=c+learning_rate*(x-x_bar);
    end
    J=test_rbm(X,W,b,c);
    track_train_error(epoch)=J;
    theta=[W(:)];
    track_wts(epoch)=sqrt(sum(theta.^2));
    J_val=test_rbm(Xval,W,b,c);
    track_val_error(epoch)=J_val;
    if (mod(epoch,50)==0)
        epoch
    end
    if prev_J_val<J_val
        if early_stopper==0
            W_final=W;
            b_final=b;
            c_final=c;
            thetas=[b_final W_final];
            thetas=[[1 c_final'];thetas];
            fname=strcat('theta_',detail_string,'.mat');
            save(fname,'thetas');
        end
        if convergence_flag==0
            convergence_flag=1;   
        end
        early_stopper=early_stopper+1;
        if early_stopper>=early_stopper_limit
            break
        end
    else
        if convergence_flag==1
           convergence_flag=0;
           early_stopper=0;
        end
    end
    prev_J_val=J_val;
end
toc
elapsedTime=toc;
elapsedTime=elapsedTime/60   
output=[track_train_error track_wts track_val_error];
fname=strcat('rbm_dev_',detail_string,'.csv');        
csvwrite(fname,output);
fname=strcat('img_',detail_string,'.png');
displayData(W_final,fname);

