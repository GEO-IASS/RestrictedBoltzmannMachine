function [J]=test(X,W,b,c)
m=size(X,1);
J=0;
for t=1:m
    x=X(t,:)';
    [cross_entropy,h_x,x_bar]=sampler_unif(x,W,b,c);
    J=J+cross_entropy;
end
J=J/m;


