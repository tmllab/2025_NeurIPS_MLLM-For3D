zero_shot_setting = {}
ScanNet_setting = {0: [],
                   2: [5, 12],  #{5: 'sofa', 12: 'desk'},
                   4: [5, 12, 9, 16], #{5: 'sofa', 12: 'desk', 9: 'bookshelf', 16: 'toilet'},
                   6: [5, 12, 9, 16, 18, 3], #{5: 'sofa', 12: 'desk', 9: 'bookshelf', 16: 'toilet', 18: 'bathtub', 3: 'bed'},
                   8: [5, 12, 9, 16, 18, 3, 13, 8], #{5: 'sofa', 12: 'desk', 9: 'bookshelf', 16: 'toilet', 18: 'bathtub', 3: 'bed', 13: 'curtain', 8: 'window'},
                   10: [5, 12, 9, 16, 18, 3, 13, 8, 7, 11], # {5: 'sofa', 12: 'desk', 9: 'bookshelf', 16: 'toilet', 18: 'bathtub', 3: 'bed', 13: 'curtain', 8: 'window', 7: 'door', 11: 'counter'}
                   }

S3DIS_setting = {0: [],
                 2: ['sofa', 'beam'],
                 4: ['sofa', 'beam', 'column', 'window'],
                 6: ['sofa', 'beam', 'column', 'window', 'bookshelf', 'board']}

semanticKITTI_setting = {0: [],
                         2: ['motorcycle', 'truck'],
                         4: ['motorcycle', 'truck', 'bicyclist', 'traffic-sign'],
                         6: ['motorcycle', 'truck', 'bicyclist', 'traffic-sign', 'car', 'terrain'],
                         8: ['motorcycle', 'truck', 'bicyclist', 'traffic-sign', 'car', 'terrain', 'vegetation', 'sidewalk']}

nuScenes_setting = {0: [],
                         2: [5, 8], #[5: "Motorcycle", 8: "trailer"],
                         4: [5, 8, 13, 7], # [5: "Motorcycle", 8: "trailer", 13: "terrain", 7: "traffic-cone"],
                         6: [5, 8, 13, 7, 1, 3], # [5: "Motorcycle", 8: "trailer", 13: "terrain", 7: "traffic-cone", 1: "bicycle", 3: "car"],
                         8: [5, 8, 13, 7, 1, 3, 15, 12]} # [5: "Motorcycle", 8: "trailer", 13: "terrain", 7: "traffic-cone", 1: "bicycle", 3: "car", 15: "vegetation", 12: "sidewalk"]}
