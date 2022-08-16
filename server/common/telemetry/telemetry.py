from dstool.database import add_event

def push_event(data, tags=[], previous=None):
    add_event(data, ['shift'].extend(tags), previous)