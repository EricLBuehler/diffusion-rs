import numpy as np

bnb = np.load("bnb_txt_in.npy")
unq = np.load("unq_txt_in.npy")

print(np.mean(np.abs(bnb, unq)))
print(bnb-unq)