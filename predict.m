function [J,accuracy]=predict(X,y_true,hidden_wts,layer2_wts)
m = size(X, 1);
X=[ones(m,1) X];
z2=X*hidden_wts;
a2=sigmoid(z2);
a2=[ones(size(a2,1),1) a2];
z3=a2*layer2_wts;
sfmax=exp(z3);
sfmax_denom=sum(sfmax,2);
a3=diag(1./sfmax_denom)*sfmax;
[maxprobs,labels]=max(a3,[],2);
y_pred=labels-1;

accuracy=mean(double(y_pred == y_true)) * 100;

Y=zeros(m,10);
for c=1:10
 Y(:,c)= (y_true==c-1);
end

J_extra=Y*transpose(log(a3));
J=-1*trace(J_extra)/m;


