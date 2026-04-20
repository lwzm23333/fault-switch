function [u, u_hat, omega] = vmd(f, alpha, tau, K, DC, init, tol)
% VMD 官方原版函数
% 输入：
% f       原始信号
% alpha   惩罚因子
% tau     时间步长
% K       模态数
% DC      是否含直流分量
% init    初始化方式
% tol     收敛阈值

if ~exist('alpha','var'), alpha = 2500; end
if ~exist('tau','var'), tau = 0; end
if ~exist('K','var'), K = 6; end
if ~exist('DC','var'), DC = 0; end
if ~exist('init','var'), init = 1; end
if ~exist('tol','var'), tol = 1e-7; end

% 镜像扩展
f = f(:);
T = length(f);
t = (1:T)/T;
f_hat = fftshift(fft(f));

% 频率域
omega = 2*pi*t;

% 初始化
u_hat = zeros(K, length(f_hat));
if init == 1
    omega_plus = (1:K)*pi/K;
elseif init == 2
    omega_plus = sort(exp(log(1e-6) + (log(0.5)-log(1e-6))*rand(K,1)));
else
    omega_plus = zeros(K,1);
end
if DC
    omega_plus(1) = 0;
end

lambda_hat = zeros(size(f_hat));
n = 0;
error = 1;

while error > tol
    u_hat_prev = u_hat;
    
    for k = 1:K
        % 更新第k个模态
        sum_u = sum(u_hat([1:k-1,k+1:K],:),1);
        u_hat(k,:) = (f_hat + alpha*u_hat(k,:)*(omega - omega_plus(k))^2 - sum_u) ...
            ./ (1 + alpha*(omega - omega_plus(k))^2);
        
        % 非直流分量更新中心频率
        if ~DC || k > 1
            omega_plus(k) = (omega*(abs(u_hat(k,:)).^2))/sum(abs(u_hat(k,:)).^2);
        end
    end
    
    % 更新拉格朗朗乘子
    lambda_hat = lambda_hat + tau*(f_hat - sum(u_hat,1));
    
    % 计算误差
    error = norm(u_hat - u_hat_prev)/norm(u_hat);
    n = n + 1;
end

% 反变换
u = zeros(K, T);
for k = 1:K
    u(k,:) = real(ifft(ifftshift(u_hat(k,:))));
end
u = u';
u_hat = u_hat';
omega = omega_plus/(2*pi);
end