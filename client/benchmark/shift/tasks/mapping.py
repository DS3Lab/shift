shift_to_aftp_map = {
    'vtab-sna-train': {
        'name': 'smallnorb',
        'config': {
            # order matters!
            'lr': 0.01,
            'feature_name': 'image',
            'label_name': 'label_azimuth',
        },
        'num_classes': 18,
        'train_slice': ':800',
    },
    'vtab-diabet-train': {
        'name': 'diabetic_retinopathy_detection',
        'config': {
            'lr': 0.01,
            'feature_name': 'image',
            'label_name': 'label',
            'preprocess': 'empty',
        },
        'num_classes': 5,
        'train_slice': ':800'
    },
    'vtab-cdp-train': {
        'name': 'clevr',
        'config': {
            'lr': 0.01,
            'feature_name': 'image',
            'label_name': 'label_orientation',
            'preprocess': 'clevr_distance'
        },
        'num_classes': 6,
        'train_slice': ':800',
        'test_split':'validation'
    },
    'vtab-kdp-train': {
        'name': 'kitti',
        'config': {
            'lr': 0.01,
            'feature_name': 'image',
            'label_name': 'label_orientation',
            'preprocess': 'kitti'
        },
        'num_classes': 4,
        'train_slice': ':800'
    },
    'vtab-eurosat-train': {
        'name': 'eurosat',
        'config': {
            'lr': 0.01,
        },
        'num_classes': 10,
        'train_slice': ':800',
        'train_split':'train',
        'test_split':'train',
        'test_slice': ':100%'
    },
    'vtab-dop-train': {
        'name': 'dsprites',
        'config': {
            'lr': 0.01,
            'feature_name': 'image',
            'label_name': 'label_orientation',
            'preprocess': 'dsprites_orientation',
        },
        'num_classes': 16,
        'train_slice': ':800',
        'test_split':"train",
    },
    'vtab-cifar-train': {
        'name': 'cifar100',
        'config': {
            'lr': 0.01,
        },
        'num_classes': 100,
        'train_slice': ':800'
    },
    'vtab-clevr-train': {
        'name': 'clevr',
        'config': {
            'lr': 0.01,
            'feature_name': 'image',
            'label_name': 'label_orientation',
            'preprocess': 'clevr_count'
        },
        'num_classes': 8,
        'train_slice': ':800',
        'test_split':"validation",
    },
    'vtab-resisc-train': {
        'name': 'resisc45',
        'config': {
            'lr': 0.01,
        },
        'num_classes': 45,
        'train_slice': ':800',
        'train_split':'train',
        'test_split':'train',
        'test_slice': ':100%'
    },
    'vtab-flowers-train': {
        'name': 'oxford_flowers102',
        'config': {
            'lr': 0.01,
        },
        'num_classes': 102,
        'train_slice': ':800'
    },
    'vtab-svhn-train': {
        'name': 'svhn_cropped',
        'config': {
            'lr': 0.01,
        },
        'num_classes': 10,
        'train_slice': ':800',
        'train_split':'train',
        'test_split':'test',
        'test_slice': ':100%'
    },
    'vtab-caltech101-train': {
        'name': 'caltech101',
        'config': {
            'lr': 0.01,
            'feature_name': 'image',
            'label_name': 'label',
            'preprocess': 'empty'
        },
        'num_classes': 102,
        'train_slice': ':800',
        'train_split':'train',
        'test_split':'test',
        'test_slice': ':100%'
    },
    'vtab-pet-train': {
        'name': 'oxford_iiit_pet',
        'config': {
            'lr': 0.01,
            'feature_name': 'image',
            'label_name': 'label',
            'preprocess': 'empty'
        },
        'num_classes': 37,
        'train_slice': ':800'
    },
    'vtab-dlp-train': {
        'name': 'dsprites',
        'config': {
            'lr': 0.01,
            'feature_name': 'image',
            'label_name': 'label_x_position',
            'preprocess': 'dsprites_location',
        },
        'num_classes': 16,
        'train_slice': ':800',
        'test_split':"train",
    },
    'vtab-snb-train': {
        'name': 'smallnorb',
        'config': {
            'lr': 0.01,
            'feature_name': 'image',
            'label_name': 'label_elevation',
        },
        'num_classes': 9,
        'train_slice': ':800'
    },
    'vtab-pc-train': {
        'name': 'patch_camelyon',
        'config': {
            'lr': 0.01,
        },
        'num_classes': 2,
        'train_slice': ':800'
    },
    'vtab-dmlab-train': {
        'name': 'dmlab',
        'config': {
            'lr': 0.01,
        },
        'num_classes': 6,
        'train_slice': ':800'
    },
    'vtab-dtd-train': {
        'name': 'dtd',
        'config': {
            'lr': 0.01,
        },
        'num_classes': 47,
        'train_slice': ':800'
    },
    'vtab-sun-train': {
        'name': 'sun397',
        'config': {
            'lr': 0.01,
        },
        'num_classes': 397,
        'train_slice': ':800',
        'train_split':'train',
        'test_split':'test',
        'test_slice':':100%',
    },
}

dataset_names = ('vtab_caltech', 'vtab_cifar', 'vtab_clevrdist', 'vtab_clevrcnt',
                 'vtab_retinopathy', 'vtab_dmlab', 'vtab_dspritespos', 'vtab_dspritesori', 'vtab_dtd',
                 'vtab_eurosat', 'vtab_kitti', 'vtab_flowers', 'vtab_pets', 'vtab_camelyon', 'vtab_resisc',
                 'vtab_smallnorbazi', 'vtab_smallnorbele', 'vtab_sun397', 'vtab_svhn')

ds_mapping = ["Caltech101",
              "CIFAR-100",
              "Clevr-Dist",
              "Clevr-Count",
              "Retinopathy",
              "DMLab",
              "dSpr-Loc",
              "dSpr-Orient",
              "DTD",
              "EuroSAT",
              "KITTI-Dist",
              "Flowers102",
              "Pets",
              "Camelyon",
              "Resisc45",
              "sNORB-Azim",
              "sNORB-Elev",
              "Sun397",
              "SVHN"]

ds_to_vtab_mapping = {
    "Caltech101": "vtab-caltech101-train",
    "CIFAR-100": "vtab-cifar-train",
    "Clevr-Dist": "vtab-cdp-train",
    "Clevr-Count": "vtab-clevr-train",
    "Retinopathy": "vtab-diabet-train",
    "DMLab": "vtab-dmlab-train",
    "dSpr-Loc": "vtab-dlp-train",
    "dSpr-Orient": "vtab-dop-train",
    "DTD": "vtab-dtd-train",
    "EuroSAT": "vtab-eurosat-train",
    "KITTI-Dist": "vtab-kdp-train",
    "Flowers102": "vtab-flowers-train",
    "Pets": "vtab-pet-train",
    "Camelyon": "vtab-pc-train",
    "Resisc45": "vtab-resisc-train",
    "sNORB-Azim": "vtab-sna-train",
    "sNORB-Elev": "vtab-snb-train",
    "Sun397": "vtab-sun-train",
    "SVHN": "vtab-svhn-train"
}