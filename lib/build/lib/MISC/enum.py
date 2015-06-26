__author__ = 'aviv'

def enum(**enums):
    return type('Enum', (), enums)