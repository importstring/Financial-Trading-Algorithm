""""
This is just a file to store some utility functions that are used in the LLM Insights project.
Imports for this file are in function to make things more compact
"""
def walk_directories(path):
     import os
     directories = (directory for directory in os.walk(path))