from .root import *
from .SG3 import *
from .SG4 import *
from .SG5 import *
from .SG1 import *
from .SG2 import *

def get_root_model():
    return get_root('4')

def get_SG3_model():
    return get_SG3('3')

def get_SG4_model():
    return get_SG4('2')

def get_SG5_model():
    return get_SG5('2')

def get_SG1_model():
    return get_SG1('2')

def get_SG2_model():
    return get_SG2('2')
