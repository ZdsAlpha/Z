# Z Layers
Z Layers are alternative layers to vanilla layers.
Unlike vanilla layers, Z layers can perform non-linear transformations.
So it does not require any activation function.

Z Layer is formed by using N parallel vanilla layers (N considered number of routes) and performing SoftPooling operation between outputs of each layer.
This lets layer learn any arbitrary non-linear transformation.
The complexity of transformation can be increased by increasing number of routes.

Since Z Layers consist of N parallel vanilla layers, these layers do not increase composition of model. 
In vanilla models it is required to stack multiple layers to learn a non-linear transformation.
But stacking too many layers makes it harder for model to converge.
As more layers are added the composition of models keeps increasing.
More layers require more time to train. This is why we use skip-connections, dense layers etc to reduce composition.
(Here composition means minimum number of linear transformation from input to output in a model's graph)

A single Z layer can approximate 2 to 3 stacked vanilla layers.
This also means it has lesser parameters and high information density.

It is not guaranteed to work as its still under research. Feel free to share your results.
