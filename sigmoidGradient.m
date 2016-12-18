function g = sigmoidGradient(z)
g = zeros(size(z));
g1=sigmoid(z);
g2=1.-g1;
g=g1.*g2;