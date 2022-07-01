% Custom Softmax layer

classdef myRandLayer < nnet.layer.Layer
    % Example custom PReLU layer.

    properties 
        % Layer learnable parameters
    end
    
    methods
        function layer = myRandLayer(name) 
            % layer = preluLayer(numChannels, name) creates a PReLU layer
            % for 2-D image input with numChannels channels and specifies 
            % the layer name.

            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = "Custom randomization layer";
        end
        
%         function dLdX = backward(layer, X, Z, dLdZ, memory)
%             dLdX = dLdZ;
% %             dLdX = Z.*dLdZ;
% %             s = sum(dLdX,3);
% %             for i=1:size(Z,4)
% %                 dLdX(1,1,:,i) = dLdX(1,1,:,i)-Z(1,1,:,i).*s(1,1,i);
% %             end
%             n = size(Z,3);           
%             m = size(Z,4);
%             for i=1:m
%                 for j=1:n
%                     J=dLdZ(1,1,j,i)*Z(1,1,j,i)*(1-Z(1,1,j,i));
%                     for k=1:n
%                         if k~=j
%                             J=J-dLdZ(1,1,k,i)*Z(1,1,j,i).*Z(1,1,k,i);
%                         end
%                     end
%                     dLdX(1,1,j,i)=J;
%                 end
%             end
%             %dLdX = cast(dLdX,'single');
%         end        

        function Z = forward(layer, X)
            Z = X.*(rand(1)*0.3+0.85);
        end
        
        function Z = predict(layer, X)
            Z = X; 
        end
    end
end