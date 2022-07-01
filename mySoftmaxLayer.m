% Custom Softmax layer

classdef mySoftmaxLayer < nnet.layer.Layer
    % Example custom PReLU layer.

    properties (Learnable)
        % Layer learnable parameters
            
    end
    
    methods
        function layer = mySoftmaxLayer(name) 
            % layer = preluLayer(numChannels, name) creates a PReLU layer
            % for 2-D image input with numChannels channels and specifies 
            % the layer name.

            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = "Custom softmax layer";
        end
        
        function dLdX = backward(layer, X, Z, dLdZ, memory)
            
            dims = length(size(Z));
            dLdX = dLdZ;
            if dims>2
                
                n = size(Z,3);           
                m = size(Z,4);
                for i=1:m
                    for j=1:n
                        J=dLdZ(1,1,j,i)*Z(1,1,j,i)*(1-Z(1,1,j,i));
                        for k=1:n
                            if k~=j
                                J=J-dLdZ(1,1,k,i)*Z(1,1,j,i).*Z(1,1,k,i);
                            end
                        end
                        dLdX(1,1,j,i)=J;
                    end
                end
            else
                
                n = size(Z,1);           
                m = size(Z,2);
                for i=1:m
                    for j=1:n
                        J=dLdZ(j,i)*Z(j,i)*(1-Z(j,i));
                        for k=1:n
                            if k~=j
                                J=J-dLdZ(k,i)*Z(j,i).*Z(k,i);
                            end
                        end
                        dLdX(j,i)=J;
                    end
                end                
            end
            
            %dLdX = cast(dLdX,'single');
        end        

        
        function Z = predict(layer, X)
            % Z = predict(layer, X) forwards the input data X through the
            % layer and outputs the result Z.
            
            dims = length(size(X));
            if dims>2
                X = X(1,1,:,:);
                Z = exp(X)./sum(exp(X));
                Z(1,1,:,:) = Z;
            else
                Z = exp(X)./sum(exp(X));
            end
        end
    end
end