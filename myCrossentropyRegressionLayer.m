classdef myCrossentropyRegressionLayer < nnet.layer.RegressionLayer
               
    properties
        %ClassWeights
    end
    methods
        function layer = myCrossentropyRegressionLayer(name)
			
            % Set layer name
            layer.Name = name;
            %layer.ClassWeights = W;

            % Set layer description.
            layer.Description = "Custom cross-entropy regression layer";
        end
    
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the cross
            % entropy loss between the predictions Y and the training
            % targets T.
            
            N = size(Y,4);
            Y = squeeze(Y);
            T = squeeze(T);
            
            %W = layer.ClassWeights';
            
%             n = length(T(:))/2;
%             W = repelem(W, [n n]);
%             W = reshape(W, size(T));
    
            prod = (T.*log(Y));
            loss = -sum(prod(:))/N;
        end
        
        function dLdY = backwardLoss(layer, Y, T)
            % dLdY = backwardLoss(layer, Y, T) returns the derivatives of
            % the weighted cross entropy loss with respect to the
            % predictions Y.
            
            [h,w,K,N] = size(Y);
            Y = squeeze(Y);
            T = squeeze(T);
            %W = layer.ClassWeights';
            
%             n = length(T(:))/2;
%             W = repelem(W, [n n]);
%             W = reshape(W, size(T));
			
            dLdY = -(T./Y)/N;
            dLdY = reshape(dLdY,[h w K N]);
        end
        
    end
end