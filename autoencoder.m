clear
trainset=load('digitstrain.txt');

valset=load('digitsvalid.txt');
Xval=valset(:,1:size(valset,2)-1);

input_layer_size=784;
hidden_layer_size=100;
m=size(trainset,1);
max_epochs=500;
early_stopper_limit=3;
learning_rate=0.01;

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

detail_string=sprintf('hls_%d_es_%d_lr_%.3f_maxepoch_%d',hidden_layer_size,early_stopper_limit,learning_rate,max_epochs);
for epoch=1:max_epochs
    shuffled_trainset=trainset(randperm(m),:);
    Xtrain=shuffled_trainset(:,1:size(shuffled_trainset,2)-1);
    X=double(Xtrain>=0.5);
    for t=1:m
        x=X(t,:)';
        z2=b+W*x;
        h_x=sigmoid(z2);%a2
        z3=c+W'*h_x;
        x_bar=sigmoid(z3);%a3
        loss=x'*log(x_bar)+(1-x)'*log(1-x_bar);
        del3=x_bar-x;
        del2=(W*del3).*sigmoidGradient(z2);
        x_bar_to_h_wts_term=del3*h_x';
        hidden_grad_term=del2*x';
        b_grad_term=del2;
        c_grad_term=del3;
        W_grad_term=x_bar_to_h_wts_term'+hidden_grad_term;
        W=W-learning_rate*W_grad_term;
        b=b-learning_rate*b_grad_term;
        c=c-learning_rate*c_grad_term;
    end
    J=test(X,W,b,c);
    track_train_error(epoch)=J;
    theta=[W(:)];
    track_wts(epoch)=sqrt(sum(theta.^2));
    J_val=test(Xval,W,b,c);
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
            fname=strcat('ae_theta_',detail_string,'.mat');
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
fname=strcat('ae_dev_',detail_string,'.csv');        
csvwrite(fname,output);
fname=strcat('ae_img_',detail_string,'.png');
displayData(W_final,fname);
