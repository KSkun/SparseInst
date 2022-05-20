import time

import cv2
import numpy as np
import torch

from demo import get_parser, setup_cfg
from sparseinst import SparseInst, VisualizationDemo

if __name__ == '__main__':
    image = cv2.imread('../hust1.jpg')
    image = image[:, :, ::-1]
    image = image.astype("float32").transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)
    image = torch.tensor(image)

    args = ['--config-file',
            'configs/sparse_inst_r50_giam.yaml',
            '--opt',
            'MODEL.WEIGHTS',
            'sparse_inst_r50_giam_ceaffc.pth',
            'INPUT.MIN_SIZE_TEST',
            '512']
    args = get_parser().parse_args(args)
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)

    with torch.no_grad():
        model = demo.predictor.model
        model.eval()
        outputs = model(image)[0]
        start_time = time.time()
        outputs = model(image)[0]
        print('runing time: %s seconds' % (time.time() - start_time))
        outputs = tuple(outputs[i].to('cpu').numpy() for i in range(3))

    output_image = cv2.imread('../hust1.jpg')
    for i in range(outputs[0].shape[0]):
        mask = outputs[0][i]
        score = outputs[1][i]
        clazz = outputs[2][i]
        if score < 0.5 or clazz != 0:
            continue

        rand_color = np.random.randint(0, 255, (3), dtype=np.uint8)
        for x in range(512):
            for y in range(512):
                if mask[x][y]:
                    output_image[x][y] = np.add(np.multiply(output_image[x][y], 0.5),
                                                np.multiply(rand_color, 0.5))
    print(outputs)
    cv2.imwrite('hust1-torch.jpg', output_image)
