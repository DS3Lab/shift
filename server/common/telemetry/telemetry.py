from dstool.database import push_couch

@push_couch
def add_event(event_type, event_data, elapsed):
    data={
        "event_type": event_type,
        "event_data": event_data,
        "elapsed_time": elapsed,
    }
    return data
