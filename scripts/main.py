import argparse
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
from urllib.error import URLError
import os

@st.cache
def streamlit():
    st.write("""
    # My first app!
    Hello, world!
    """)



if __name__ == '__main':
    print('Entered MAIN')
    parser = argparse.ArgumentParser(description='Different RecSys algorithms.')
    parser.add_argument('--algo', metavar='ALGO', type=str, choices=['CF', 'MF'])
    parser.add_argument('--delim', metavar='DELIMITER', type=str, default='\t',
                        help='the delimiter to use when parsing input files')
    parser.add_argument('--batch', metavar='BATCH_SIZE', type=int, default=25000,
                        help='the batch size to use when doing gradient descent')
    parser.add_argument('--no-early', default=False, action='store_true',
                        help='disable early stopping')
    parser.add_argument('--early-stop-max-epoch', metavar='EARLY_STOP_MAX_EPOCH', type=int, default=40,
                        help='the maximum number of epochs to let the model continue training after reaching a '
                             'minimum validation error')
    parser.add_argument('--max-epochs', metavar='MAX_EPOCHS', type=int, default=1000,
                        help='the maximum number of epochs to allow the model to train for')
    parser.add_argument('--threshold', metavar='THRESHOLD', type=int, default=3)
    parser.add_argument('--distance-measures', metavar='DISTANCE_MEASURES', type=float, default=1.0 )

    args = parser.parse_args()
    algo = args.algo
    delimiter = args.delimiter
    batch_size = args.batch
    use_early_stop = not args.no_early
    max_epochs = args.max_epochs
    early_stop_max_epoch = args.early_stop_max_epoch
    threshold = args.threshold
    distance_measures = args.distance_measures
    streamlit()


