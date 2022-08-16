import tensorflow_datasets as tfds
from shift.io.dataset.vtab_preprocessors.clevr import _closest_object_preprocess_fn, _count_preprocess_fn
from shift.io.dataset.vtab_preprocessors.kitti import _closest_vehicle_distance_pp
from shift.io.dataset.vtab_preprocessors.dsprites import dsprites_location_fn, dsprites_orientation_fn

vtab_to_preprocess = {
    "vtab-kdp-train": 'kitti',
    'vtab-clevr-train': 'clevr_count',
    'vtab-cdp-train': 'clevr_distance',
    'vtab-dlp-train': 'dsprites_location',
    'vtab-dop-train': 'dsprites_orientation'
}

vtab_required_preprocessors = [
    'kitti',
    'clevr_count',
    'clevr_distance',
    'dsprites_location',
    'dsprites_orientation',
]

vtab_preprocess_num_classes = {
    'kitti': 4,
    'clevr_count': 8,
    'clevr_distance': 6,
    'dsprites_location': 16,
    'dsprites_orientation': 16,
}

preprocessing_map = {
    'kitti': _closest_vehicle_distance_pp,
    'clevr_count': _count_preprocess_fn,
    'clevr_distance': _closest_object_preprocess_fn,
    'dsprites_location': dsprites_location_fn,
    'dsprites_orientation': dsprites_orientation_fn,
}

def load_data_from_preproc(
                           preprocess_spec,
                           train_split,
                           val_split,
                           test_split,
                           train_slice,
                           val_slice,
                           test_slice,
                           ):
    print(vtab_required_preprocessors)
    if preprocess_spec in vtab_required_preprocessors:
        if preprocess_spec == 'kitti':
            load_name = 'kitti'
        elif preprocess_spec in ['clevr_count', 'clevr_distance']:
            load_name = 'clevr'
        elif preprocess_spec in ['dsprites_location', 'dsprites_orientation']:
            load_name = 'dsprites'
        else:
            raise ValueError('Unknown preprocess_spec: {}'.format(preprocess_spec))
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
                load_name,
                split=[
                    f'{train_split}[{train_slice}]',
                    f'{val_split}[{val_slice}]',
                    f'{test_split}[{test_slice}]'],
                # as_supervised=True,
                shuffle_files=True,
                with_info=True,
            )
        ds_train = ds_train.map(preprocessing_map[preprocess_spec])
        ds_val = ds_val.map(preprocessing_map[preprocess_spec])
        ds_test = ds_test.map(preprocessing_map[preprocess_spec])
        return ds_train, ds_val, ds_test, ds_info
    else:
        raise ValueError(f'Preprocess spec specified but not supported: {preprocess_spec}')