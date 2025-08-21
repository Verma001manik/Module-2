from minitorch.tensor_data import index_to_position,to_index

from minitorch.tensor_ops import tensor_map

def square(x):
    return x*x
import numpy as np

# our storages
in_storage = [1.0, 2.0]                  # shape (2, 1)
in_shape = [2, 1]
in_strides = [1, 1]                      # row-major (simplified)

out_storage = [0.0] * 6                  # shape (2, 3)
out_shape = [2, 3]
out_strides = [3, 1]                     # row-major

fn = lambda x: x * 2

map_fn = tensor_map(fn)

# call it
map_fn(out_storage, out_shape, out_strides,
       in_storage, in_shape, in_strides)

print(out_storage)


'''
 fn: reduction function mapping two floats to float
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        reduce_dim (int): dimension to reduce out

    
    
    what are we trying to do?
    we need to reduce a given dimension
    what do we have , information?
    we are given a_shape, strides,storage, reduce_dim

    we have only one ,a_storage.
    but the func takes 2 float values

    one we can get from a_storage but who will give the another?

    lets make guess?
    can out_storage give us?
    no , why? because we need to build that thing ourselves
    out is at present empty and our answers need to go there

    so what else? who else can give?
    if out cant give and there is no other person who can give us the 2nd float value
    it should be a_storage itself
    but how will it be able to give us 

    what is it that we are trying to do 

    cant use the prev tensor map, tensor zip methods because , we were finding for one idx, 
    now we need 2 idxs 
    and we cant use the same idx twice because of they being the same thing deterministic 

    there is some way we need to get the 2 idx

    for each idx and shape , the unravel method gives the row,col position indexes for that index
    that means we would need 2 differnt indexes , 
    if we get 2 different indexes i think the problem is solved by using the previous method iff we have the 2 idxs 

    the question is how to get the idx
    
    or the question is how to not get the idxs

    lets say the strides are (3,1)
    then if the given dim is 0 
    we could just use 3* 

    we cant forget that the matrix is stored in a 1 dim array 

    can we find 2 indexes , such that they belong to the given dimension? 
    yes 
    if we want to find the another row element we just have to skip +3 places or strides[0]  places
    if we want to find another col element we just have to skip +1 places or strides[1] places to get

    can we just use 
    
    for i in range(len(storage)- given_dimension):
        
        row_elements = []
        for j in range(len(storage)- given_dimension, given_dimension):
            el = storage[j]
            row_elements.append(el)
        
    
    we thus find all the row elements or given dimension elements

    
    

    ok so the final line must be something like this



    a_idx_flat = sum(i*j for i,j in zip(a_md_idx, strides/strides[given_dimension]))
    b_idx_flat = sum(i*j for i , j in zip(b_md_idx, strides/strides[given_dimension]))


    out[idx] =  fn(a_storage[a_idx_flat], a_storage[b_idx_flat])



    the broadcast_index does from big to small
    and outputs to small_index

    previously the small shape was given shape or the input shape not the output shape
    but here we have the output shape as a small shape after reduction

    should we choose this as the small index or not

    if we do not choose this index,then we will not be able to broadcast_index 
    because it requires big to small not small to big 

    so we have to use it , there is  not stronger counter arguement right now to not use it
    if we use it 
    we can say that the output shape is the smallest shape 
    and by this logic we can say that the input shape is bigger shape than the output shape 
    so we can we use these are respective params for broadcast_index

    if thats possible the question is how to use the current 2 indexs answers as the next singular input
    with the other ? 


    the one thing i can think of is , 
    we can storage back the fn output to the storage itself but at i+1 or the 2nd params position idx
    then go on till nomore storages left 
    
'''