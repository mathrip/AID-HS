import nibabel as nb
import os, sys
import numpy as np
import argparse
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='create mask 1% and 99% of body and tail of hippocampus')
    parser.add_argument('-i', '--input',
                        help='gii file',
                        required=False,
                        default=os.getcwd())
    parser.add_argument('-o', '--output',
                        help='new mgz filename.',
                        required=False,
                        default=None)
    args = parser.parse_args()

file_input = args.input
file_mask = args.output

# load mesh template
unfold= nb.load(file_input)
vertices = unfold.agg_data('pointset')
faces = unfold.agg_data('triangle')

# create mask 1% and 99% body and tail
n=len(vertices)
mask = np.zeros(len(vertices))
x_sorted=np.argsort(vertices[:,0])
x_sorted_mask=x_sorted[int(0.01*n):int(0.99*n)]
mask[x_sorted_mask]=1


#save mask 
mask = mask.astype(np.int32)
data_array = nb.gifti.gifti.GiftiDataArray(data=mask)
imgii = nb.gifti.GiftiImage(darrays=[data_array])
nb.save(imgii, file_mask)
