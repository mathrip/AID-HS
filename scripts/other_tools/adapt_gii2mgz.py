import os
import sys
import logging
import argparse
import nibabel as nb 
import subprocess as sub
from glob import glob
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Adapt gii to display in freeview')
    parser.add_argument('-i', '--input',
                        help='gii file',
                        required=False,
                        default=os.getcwd())
    parser.add_argument('-o', '--output',
                        help='new mgz filename.',
                        required=False,
                        default=None)
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output

    surf= nb.load(input_file)
    vertices = surf.agg_data('pointset')
    faces = surf.agg_data('triangle')
    
    nb.freesurfer.io.write_geometry(output_file,vertices,faces)

