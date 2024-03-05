%%
clc
clear 
close all

load('iddata-20.mat');

nk=1; 
na=2; nb=2;
ts = val.Ts;
uid = id.u;
yid = id.y;
uval = val.u;
yval = val.y;

m=3;
%example
%na=2; nb=na;
%dk=[-3,-6,1,2];
%regressors = reg_generator_pred(uid, yid, m, na, nb);
%matrix = power_matrix(dk, m, []);

%% PREDICTION MODEL

for m=1:3 %the system's order won't be higher than 3
    for na=1:10
        nb=na;
        [modelpred, yid_pred, yval_pred] = predictionARX(id,val,m,na,nb);

        %mse error id data - prediction
        mseid_pred = 0;
        for i= 1:length(yid)
            mseid_pred = mseid_pred + (yid_pred(i)-yid(i))^2;
        end
        mseid_pred = 1/length(yid) * mseid_pred;

        %mse error val data - prediction
        mseval_pred = 0;
        for i= 1:length(yval)
            mseval_pred = mseval_pred + (yval_pred(i)-yval(i))^2;
        end
        mseval_pred = 1/length(yval) * mseval_pred;

        MSEid_pred(m,na) = mseid_pred;
        MSEval_pred(m,na) = mseval_pred;
    end 
end
minmsevalpred = min(min(MSEval_pred));

[modelpredmin, yid_pred, yval_pred] =predictionARX(id, val, 1, 10, 10);

% for mse minimum val: m=1, na=9
% mse min id: m=3, na=7

figure
subplot(121)
plot(modelpredmin.y); 
hold on; plot(yid);
legend('model','yid');
title('Pred model vs yid for mse=0.0123');
subplot(122)
plot(modelpredmin.y); 
hold on; plot(yval);
legend('model','yval');
title('Pred model vs yval for mse=0.0177');


%SIMULATION MODEL
for m=1:3
    for na=1:10
        [modelsim, yid_sim, yval_sim]=simulationARX(id,val,m,na,nb);

        %mse error identification data- simulation
        % !!! NOT NECESARY
        mseid_sim = 0;
        for i= 1:length(yid)
            mseid_sim = mseid_sim + (yid_sim(i)-yid(i))^2;
        end
        mseid_sim = 1/length(yid) * mseid_sim;

        %mse error validation data- simulation
        mseval_sim = 0;
        for i= 1:length(yval)
            mseval_sim = mseval_sim + (yval_sim(i)-yval(i))^2;
        end
        mseval_sim = 1/length(yval) * mseval_sim;

        MSEid_sim(m,na) = mseid_sim;
        MSEval_sim(m,na) = mseval_sim;

    end
end

[modelsimmin, yid_sim, yval_sim] =simulationARX(id, val, 1, 10, 10);

figure
subplot(121)
plot(modelsimmin.y); 
hold on; plot(yid);
legend('model','yid');
title('Sim-model vs yid for mse=0.0628');

subplot(122)
plot(modelsimmin.y); 
hold on; plot(yval);
legend('model','yval');
title('Sim-model vs yval for mse=0.0475');

%%mse plots
% figure
% subplot(121);
% plot(MSEid_pred);
% hold on
% plot(MSEval_pred);
% legend('mse id', 'mse val');
% title('MSE for m=2- PREDICTION');
% subplot(122);
% plot(MSEid_sim);
% hold on
% plot(MSEval_sim);
% legend('mse id', 'mse val');
% title('MSE for m=2- SIMULATION');

%% FUNCTIONS NEEDED

function [model,yid_sim, yval_sim] = simulationARX(id,val,m,na,nb) %returns the model and the MSE
    ts = val.Ts;    
    uid = id.u;
    yid = id.y;
    uval = val.u;

    phi_predid = reg_generator_pred(uid, yid, m, na, nb);  
    theta = phi_predid \ yid; %theta for prediction identification data

    %simulation model
    yid_sim = reg_generator_sim(uid,m,na,nb,theta);          
    yval_sim = reg_generator_sim(uval,m,na,nb,theta);        

    model = iddata(yval_sim, uval, ts);  
end


function [model,yid_pred,yval_pred] = predictionARX(id, val, m, na, nb) %returns the model and the MSE
    ts = val.Ts;    
    uid = id.u;
    yid = id.y;
    uval = val.u;
    yval = val.y;

    %prediction model
    phi_predid = reg_generator_pred(uid, yid, m, na, nb);  
    theta = phi_predid \ yid;
    
    yid_pred = phi_predid*theta; %yhat prediction id

    phi_predval = reg_generator_pred(uval, yval, m, na, nb);
    yval_pred = phi_predval*theta; %yhat prediction val

    model = iddata(yval_pred, uval, ts);
end


function [regressors] = reg_generator_pred(u, y, m, na, nb)
    mat = power_matrix(zeros(1,na+nb), m, []);    
    regressors = zeros(length(u),length(mat));
    for i=1:length(u)
        dk = ones(1,na+nb);
        for a=1:na
            if(i>a)
                dk(a) = -y(i-a);
            else
                dk(a) = 0;
            end
        end
        for b=1:nb
            if(i>b)
                dk(na+b) = u(i-b);
            else
                dk(na+b) = 0;
            end
        end %calculated the delayed phi
        phi_new = ones(1,size(mat,1));
        l = size(mat,1);
        for k = 1:l
            for j = 1:length(dk)
                phi_new(k) = phi_new(k) * dk(j)^mat(k,j);
            end
        end
        regressors(i,:) = phi_new; 
        %all the delayed elements, to the correct powers from the power matrix
    end
end

%regression function for simulation:
function [yhat_sim] = reg_generator_sim(u, m, na, nb, theta)
    yhat_sim = zeros(length(u),1);
    mat = power_matrix(zeros(1,na+nb), m, []);    
    for i=1:length(u)
        dk = zeros(1,na+nb);
        for a=1:na
            if(i>a)
                dk(a) = -yhat_sim(i-a);
            else
                dk(a) = 0;
            end
        end
        for b=1:nb
            if(i>b)
                dk(na+b) = u(i-b);
            else
                dk(na+b) = 0;
            end
        end %calculated the delayed phi (dk)
        %phi must be a row!!!
        phi_new = ones(1,size(mat,1));
        l = size(mat,1);
        for k = 1:l
            for j = 1:length(dk)
                phi_new(k) = phi_new(k) * dk(j)^mat(k,j);
            end
        end
        %all the delayed elements, to the correct powers from the power matrix
        yhat_sim(i) = phi_new*theta;
    end
end

function [matrix] = power_matrix(d, m,matrix)
    n = length(d);
    matrix = zeros(1, n);

    for i = 1:m
        aux = zeros(size(matrix, 1) * n, n);
        row_index = 1;

        for j = 1:size(matrix, 1)
            for k = 1:n
                aux(row_index, :) = matrix(j, :);
                aux(row_index, k) = aux(row_index, k) + 1;
                row_index = row_index + 1;
            end
        end
        matrix = unique(aux, 'rows');
    end
end
