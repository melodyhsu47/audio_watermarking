%X = host (original) signal; 
%y = audio data; fs = sample rate; L = number of audio samples
%possible audio samples:
%Turbine-16-44p1-mono-22secs.wav'
%Ambiance-16-44p1-mono-12secs.wav'
%'Counting-16-44p1-mono-15secs.wav'
%'TrainWhistle-16-44p1-mono-9secs.wav'


[X,fs] = audioread('Ambiance-16-44p1-mono-12secs.wav');
info = audioinfo('Ambiance-16-44p1-mono-12secs.wav');

%[Y,fs2] = audioread('Ambiance-16-44p1-mono-12secs.wav');
%info2 = audioinfo('Ambiance-16-44p1-mono-12secs.wav');

%[Y,fsy] = audioread('TrainWhistle-16-44p1-mono-9secs.wav');
%infoY = audioinfo('TrainWhistle-16-44p1-mono-9secs.wav');

t = 0:seconds(1/fs):seconds(info.Duration);
t = t(1:end-1); %because orig. t has one more value than # samples


%ONLY CHANGE VALUES DOWN HERE%
%---------------------------------------------------------------------%
L = length(t); %number of audio samples
V = [1 1 1 1 1 0 0 1 1 0 1 0 1 1 1 0]; %synchronization code
P = length(V); %length of synchronization code (# bits)
X1 = X(1:15360);
X2 = X(length(X1)+1:end);
n = length(X1)/P; %length of each segment in X1
Su = 0.1; %mean weight
Ss = 0.8; %std weight
dm = 0.01; %min quantization step
dM = 0.1; %max quantization step

M = 32; %dimensions of square watermark
I = imread('testpat1.png'); %watermark image; %circlesBrightDark.png, testpat1.png
I = imresize(I, [M M]);
W = imbinarize(I);
figure
imshow(W);
%---------------------------------------------------------------------%
%DON'T CHANGE ANYTHING BELOW HERE


%SYNCHRONIZATION CODE EMBEDDING
samples = zeros(P,n);
for m=1:P
    for k=1:n
        sample = X1(k+(m-1)*n);
        samples(m,k) = sample;
    end
end
means_X1 = mean(samples,2);

synced = zeros(P,n);
for m=1:P
    for k=1:n
        if V(m) == 1
            synced(m,k) = samples(m,k) - (means_X1(m) - abs(means_X1(m)));
        else
            synced(m,k) = samples(m,k) - (means_X1(m) + abs(means_X1(m)));
        end
    end
end

%WATERMARK EMBEDDING
[a,d] = dwt(X2,'haar');
coeff = a';%obtain low level coefficients for X2
u = sqrt(length(coeff)/(M*M)); %dimension of 2D matrix

Dj = zeros(u,u,M*M); %2D blocks of low level coeffs
SVj = zeros(1,u,M*M); %singular values of blocks
zj = zeros(1,M*M); %norms of vectors of SVs
mDj = zeros(1,M*M); %means of each Dj block
sDj = zeros(1,M*M); %stds of each Dj block
weightj = zeros(1,M*M); %weights of each Dj block
Uj = zeros(u,u,M*M); %U matrices (from svd)
Sj = zeros(u,u,M*M); %S matrices (from svd)
Vj = zeros(u,u,M*M); %V matrices (from svd)

for j=1:(M*M)
    D = reshape(coeff(1+(u^2)*(j-1):(u^2)*j),u,u)';
    SV = svd(D)';
    %----------------
    [U,S,V] = svd(D);
    Uj(:,:,j) = U;
    Sj(:,:,j) = S;
    Vj(:,:,j) = V;
    %----------------
    z = norm(SV);
    uD = mean(D,'all');
    sD = std(D,1,'all');
    weight = (Su*uD)+(Ss*sD); %weight of block
    
    Dj(:,:,j)= D;
    SVj(:,:,j) = SV;
    zj(j) = z;
    mDj(j) = uD;
    sDj(j) = sD;
    weightj(j) = weight;
end

weightM = max(weightj); weightm = min(weightj);
dj = zeros(1,M*M); %quantization steps
Cj = zeros(1,M*M); %watermarked code bit

for j=1:M*M
    delta = dm+(dM-dm)*((weightj(j)-weightm)/(weightM-weightm)); %quantization step for Dj
    dj(j) = delta;
    C = floor(zj(j)/delta);
    Cj(j) = C;
end

for i=1:M
    for j=1:M
        if W(i,j) == 1 && (mod(Cj(j),2) == 1)
            Cj(j) = Cj(j)+1;
        elseif W(i,j) == 0 && (mod(Cj(j),2) == 0)
            Cj(j) = Cj(j)+1;
        else
            Cj(j) = Cj(j)+0;
        end
    end
end

Gj = zeros(u,u,M*M); %new singular values matrix
zpj = zeros(1,M*M);
for j=1:M*M
    zp = dj(j)*Cj(j)+(dj(j)/2); %altered norm
    zpj(j) = zp; %add altered norm to norm vector
    G = SVj(1,:,j)*(zp/zj(j)); %calculating new SV using ratio of norms
    G = diag(G);
    Gj(:,:,j) = G;
end

reconstructed = zeros(1,u*u*M*M);
for j=1:M*M
    reconstruct = Uj(:,:,j)*Gj(:,:,j)*Vj(:,:,j);
    reconstruct = reconstruct';
    reconstruct = reconstruct(:);
    reconstruct = reconstruct'; %finished reconstructed dim is 16*16 = 256
    reconstructed(1+(u*u)*(j-1):(u*u)*(j)) = reconstruct;
end
reconstructed = reconstructed';
reconstructed(isnan(reconstructed))=0;
    

%reconstruct = Uj.*Gj.*Vj; %inverse SVD with new singular values
%reconstruct = reconstruct(:);
%reconstruct(isnan(reconstruct))=0;
watermarked = idwt(reconstructed,d,'haar');
figure
subplot(2,1,1)
plot(1:length(X2),X2)
xlabel('Time'); ylabel('Signal'); title('Original signal');
subplot(2,1,2)
plot(watermarked)
ylim([-3 3]);
title('Reconstructed watermarked signal');

SNR = 10*log10(sum(X2.^2)/(sum((X2-watermarked).^2)));

%WATERMARK EXTRACTION
[a2,d2] = dwt(watermarked,'haar');
watermark_ext = zeros(1,M*M);
coeff2 = a2';
for j=1:M*M
    D = reshape(coeff2(1+(u^2)*(j-1):(u^2)*j),u,u)';
    sv = svd(D)';
    zt = norm(sv);
    C = floor(zt/dj(j));
    if mod(C,2) == 0
        watermark_ext(j) = 1;
    else
        watermark_ext(j) = 0;
    end
end

%imshow(reshape(watermark_ext,M,M)')
        
