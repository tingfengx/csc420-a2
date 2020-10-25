CSC420 Assignment #2
Tingfeng Xia (1003780884)

Since you are here, you should have already unzipped
the Assignment 2 folder...

To run my code, ```cd``` into the ```Assignment 1``` 
folder, and run ```jupyter notebook```. The notebook 
containing my code is named ```a2_code.ipynb```. All 
my code for this assignment is inside this ipython 
notebook, and all code inside are clearly labeled for 
which question they were answering.

Special Note:
My code uses Numba JIT to speed up numPy computation. 
It drastically increases my code's running speed, (up 
to about thirty times faster). However, I have myself 
encountered several times that Numba JIT stop working
and complaining about an internal bug. If this happens, 
you can try to restart the jupyter notebook, or turn 
off the Numba JIT acceleration entirely, by deleting 
```from numba import jit``` in the first cell, and 
delete all the ```@jit``` decorators. 
