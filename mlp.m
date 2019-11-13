%% MLP - dataset: Fisheriris
%author = @leilamr
close all; 
clear all;
clc

%load dataset
load fisheriris
    
%convert targets: 1 = "setosa", 2 = "versicolor" e 3 = "virginica"
classnames = unique(species);
for i=1:3
    class(strcmp(species, classnames{i})) = i;
end
class = class';

%inputs data
inputData = [class meas];

%definition of classes of iris plant types
a=[+1 -1 -1];
b=[-1 +1 -1];
c=[-1 -1 +1];

avgTest = 0;

%% cross-validation: KFold (k = 10) manual
j = 10;
% size kx = 15x5
k1 = [inputData(1:3,:); inputData(33:44,:)];
k2 = [inputData(4:6,:); inputData(45:46,:)];
k3 = [inputData(7:9,:); inputData(57:68,:)];
k4 = [inputData(10:12,:); inputData(69:80,:)];    
k5 = [inputData(13:15,:); inputData(87:92,:)];
k6 = [inputData(16:18,:); inputData(93:104,:)];
k7 = [inputData(19:21,:); inputData(105:116,:)];
k8 = [inputData(22:24,:); inputData(117:128,:)];
k9 = [inputData(25:28,:); inputData(129:139,:)];
k10 = [inputData(29:32,:); inputData(140:150,:)];

%above folds exchange
for i=1:j
    if i == 1
        %make train matrix(135x5)
        trainMatrix = [k1;k2;k3;k4;k5;k6;k7;k8;k9];
        testMatrix = k10;
    end
    
    if i == 2
        %make train matrix(135x5)
        trainMatrix = [k1;k2;k3;k4;k5;k6;k7;k8;k10];
        testMatrix = k9;
    end
    
    if i == 3
        %make train matrix(135x5)
        trainMatrix = [k1;k2;k3;k4;k5;k6;k7;k9;k10];
        testMatrix = k8;
    end
    
    if i == 4
        %make train matrix(135x5)
        trainMatrix = [k1;k2;k3;k4;k5;k6;k8;k9;k10];
        testMatrix = k7;
    end
    
     if i == 5
        %make train matrix(135x5)
        trainMatrix = [k1;k2;k3;k4;k5;k7;k8;k9;k10];
        testMatrix = k6;
     end
    
      if i == 6
        %make train matrix(135x5)
        trainMatrix = [k1;k2;k3;k4;k6;k7;k8;k9;k10];
        testMatrix = k5;
      end
      
       if i == 7
        %make train matrix(135x5)
        trainMatrix = [k1;k2;k3;k5;k6;k7;k8;k9;k10];
        testMatrix = k4;
       end
    
     if i == 8
        %make train matrix(135x5)
        trainMatrix = [k1;k2;k4;k5;k6;k7;k8;k9;k10];
        testMatrix = k3;
     end
    
      if i == 9
        %make train matrix(135x5)
        trainMatrix = [k1;k3;k4;k5;k6;k7;k8;k9;k10];
        testMatrix = k2;
      end
    
       if i == 10
        %make train matrix(135x5)
        trainMatrix = [k2;k3;k4;k5;k6;k7;k8;k9;k10];
        testMatrix = k1;
       end
    
       %% start MLP
       %matrix for Train (135x4)
       inputTrain = trainMatrix(:,2:5);
       
       
       %target (135x1)
       target = trainMatrix(:,1);
       
       %normalization of data-input
       inputTrain = mapminmax(inputTrain);
       
       %normalization of target
       outputTrain = zeros(size(target,1),3);

       for j=1:size(target,1)
           if target(j,1) == 1
               outputTrain(j,:) = a;
           end
           if target(j,1) == 2
               outputTrain(j,:) = b;
           end
           if target(j,1) == 3
               outputTrain(j,:) = c;
           end
       end
       
       %matrix transposed input and output network
       inputTrain = inputTrain'; %(4x135)
       outputTrain = outputTrain'; %(3x135)
       
       %% make network
       n_hidden_nodes = 65;
       net = feedforwardnet(n_hidden_nodes);

       net.trainParam.epochs = 1000; 
       net.trainParam.max_fail = 500;

       net.trainParam.min_grad = 0.000000000000001;

       net.divideParam.trainRatio = 1;
       net.divideParam.valRatio = 0;
       net.divideParam.testRatio = 0;

       net.layers{1}.transferFcn='logsig';
       net.layers{2}.transferFcn='tansig';

       %% train network
       
       [net,xTest, yTest] = train(net,inputTrain,outputTrain);

       %% test network performance
       inputTest = testMatrix(:,2:5); %(15x4)
       targetTest = testMatrix(:,1);
       
       % normalization of data-test
       inputTest = mapminmax(inputTest);
       outputTest = zeros(size(targetTest,1),3);

       for j=1:size(targetTest,1)
           if targetTest(j,1) == 1
               outputTest(j,:) = a;
           end
           if targetTest(j,1) == 2
               outputTest(j,:) = b;
           end
           if targetTest(j,1) == 3
               outputTest(j,:) = c;
           end
       end
       
       %matrix transposed input and output network
       inputTest = inputTest'; %(4x15)
       outputTest = outputTest'; %(3x15)
       
       %network prediction result
       result = sim(net,inputTest);
       [per con] = confusion(outputTest,result);
       perTest = 100 * (1 - per);
       
       avgTest = avgTest + perTest;
       
       %plot and save confusion matrix
       
        x(i)=plotconfusion(result,outputTest);
    
        if i==1
            saveas(x(i),'confusion_1.jpg');  
        end
        
        if i==2
            saveas(x(i),'confusion_2.jpg');  
        end
        
        if i==3
            saveas(x(i),'confusion_3.jpg');  
        end
        
        if i==4
            saveas(x(i),'confusion_4.jpg');  
        end
        
        if i==5
            saveas(x(i),'confusion_5.jpg');  
        end
        
        if i==6
            saveas(x(i),'confusion_6.jpg');  
        end
        
        if i==7
            saveas(x(i),'confusion_7.jpg');  
        end
        
        if i==8
            saveas(x(i),'confusion_8.jpg');  
        end
        
        if i==9
        saveas(x(i),'confusion_9.jpg');  
        end
        
        if i==10
            saveas(x(i),'confusion_10.jpg');  
        end
end
 avgTest=avgTest/10;
    
 fprintf('network hit average (kfold = 10): %.3f%%\n', avgTest);
    
    


