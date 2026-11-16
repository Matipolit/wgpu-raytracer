"""Generate original toroidal void-and-cluster ranked spatial jitter masks."""
from pathlib import Path
import numpy as np
N=64
axis=np.minimum(np.arange(N),N-np.arange(N))
kernel=np.exp(-(axis[:,None]**2+axis[None,:]**2)/(2*1.5**2))
def shifted(index):
 y,x=divmod(int(index),N)
 return np.roll(np.roll(kernel,y,0),x,1)
def generate(seed):
 rng=np.random.default_rng(seed);mask=np.zeros((N,N),bool)
 mask.ravel()[rng.choice(N*N,N*N//10,replace=False)]=True
 density=np.fft.ifft2(np.fft.fft2(mask)*np.fft.fft2(kernel)).real
 for _ in range(20000):
  tight=np.argmax(np.where(mask,density,-np.inf));mask.ravel()[tight]=False;density-=shifted(tight)
  hole=np.argmin(np.where(~mask,density,np.inf));mask.ravel()[hole]=True;density+=shifted(hole)
  if tight==hole:break
 else:raise RuntimeError('relaxation did not settle')
 original=mask.copy();original_density=density.copy();ranks=np.zeros((N,N),np.uint16)
 for rank in range(int(mask.sum())-1,-1,-1):
  tight=np.argmax(np.where(mask,density,-np.inf));ranks.ravel()[tight]=rank;mask.ravel()[tight]=False;density-=shifted(tight)
 mask=original;density=original_density
 for rank in range(int(mask.sum()),N*N):
  hole=np.argmin(np.where(~mask,density,np.inf));ranks.ravel()[hole]=rank;mask.ravel()[hole]=True;density+=shifted(hole)
 assert len(np.unique(ranks))==N*N
 return np.floor((ranks.astype(np.float64)+.5)/(N*N)*65536).astype(np.uint32)
x=generate(273);y=generate(1171)
path=Path(__file__).resolve().parents[1]/'assets/sampling/blue-noise-64.bin'
path.write_bytes((x|(y<<16)).astype('<u4').tobytes())
print(path)
