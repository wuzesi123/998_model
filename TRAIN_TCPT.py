import argparse, glob
from chem_process_rag.visual.train_observable import train
p=argparse.ArgumentParser(description="Train optional TCPT observable-state head on canonical NPZ files")
p.add_argument("--glob",default="data/training/*.npz",help="NPZ must contain features[T,D], timestamps[T], labels[T]")
p.add_argument("--out",default="checkpoints/tcpt_observable.pt"); p.add_argument("--epochs",type=int,default=20); p.add_argument("--lr",type=float,default=2e-4)
a=p.parse_args(); files=glob.glob(a.glob); print("files",len(files)); train(files,a.out,a.epochs,a.lr)
