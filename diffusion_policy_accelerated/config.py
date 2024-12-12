from enum import Enum
from contextlib import contextmanager

class InferenceMode(Enum):
    NORMAL = 'normal'
    ACCELERATED = 'accelerated'

@contextmanager
def inference_mode_context(new_mode):
    '''
    Sets global inference mode using a context manager. 
    '''
    global INFERENCE_MODE
    old_mode = INFERENCE_MODE
    INFERENCE_MODE = new_mode
    try:
        yield
    finally:
        INFERENCE_MODE = old_mode

INFERENCE_MODE = InferenceMode.NORMAL

TENSOR_SHAPES = \
[
(256,256, 16), 
(2048,512, 4), 
(1024,1024, 4), 
(1024,1024, 4), 
(256,256, 16), 
(256,256, 8), 
(1024,256, 8), 
(256,256, 8), 
(1024,1024, 4), 
(256,512, 8), 
(512,512, 4), 
(256,256, 8), 
(512,512, 8), 
(256,256, 16), 
(1024,1024, 4), 
(512,512, 8), 
(512,512, 8), 
(1024,1024, 4),  
(1024,1024, 4), 
(256,256, 16), 
(512,512, 4), 
(512,1024, 4), 
(1024,1024, 4), 
(512,512, 4)
]