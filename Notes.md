# Notes


## Intuition
- High level dimensions understanding: 
    - Image => (batch_size, 28, 28)
    - Conv Layer => (batch_size, 256, 20, 20)
    - PrimaryCaps => (batch_size, 1152, 8) composed of 32 capsules, each (batch_size, 8, 6, 6)
    - DigitCaps => (batch_size, 10, 16)

- The primary caps layer has 32 capsules, each capsule is a convolution layer with input channels of 256, and output channel of 8. 
- Given that the output of the convolution inside a primary capsule is [], we really want a 3d of [batch_size, 6x6x32, 8], so we resize it, then squash the values.
- A way to think about convolution capsule layer, is that they are a list of convolution layers stacked up.


- For weights W inside DigitCaps, used for Dynamic Routing, the dimensions are (1, 1152, 10, 16, 8). The intuition is:
    - (1) is to be broadcastable to the `batch_size`. 
    - (1152, 10) is number of primary capsules in the lower level, each one is connected to the 10 digit capsules, and must sum to 1. Each on of the 1152 has to have a weight of agreement with each one of the 10 digit caps. 
    - (16, 8) This is useful for the transformation matrix, we'll need to map the initial (8) dimensional vectors to (16). #TODO. This one needs more refinement. 
- `torch.matmul` when the tensors have more than 2 dimensions, it uses the last two dimensions to do the matrix multiplication, the other dimensions are done like batching. That is, they are just "broadcast".


### Comparing CapsNet