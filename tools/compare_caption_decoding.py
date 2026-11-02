import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'03_Assignment3'))
from cs231n.coco_utils import load_coco_data
from cs231n.beam_search import compare_decoding
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('cell_type',choices=['rnn','lstm']);args=parser.parse_args()
    compare_decoding(load_coco_data(),args.cell_type)
