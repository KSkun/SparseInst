import time

import cv2
import ncnn
import numpy as np
import threading


def masking_and_show(resized, outputs):
    output_image = resized.copy()
    for i in range(100):
        mask = outputs[0][i]
        score = outputs[1][i]
        clazz = outputs[2][i]
        if score < 0.5 or clazz != 0:
            continue

        mask = np.divide(mask.astype(float), 2)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        rand_color = np.array([255, 0, 0], dtype=np.uint8)
        # rand_color = np.random.randint(0, 255, (3), dtype=np.uint8)
        rand_color.shape = (1, 1, 3)
        rand_color = np.broadcast_to(rand_color, (512, 512, 3))
        output_image = np.add(np.multiply(mask, rand_color),
                              np.multiply(np.subtract(1, mask), output_image)).astype(np.uint8)

    cv2.imshow('SparseInst', output_image)
    if cv2.waitKey(1) == ord('q'):
        return


if __name__ == '__main__':
    ncnn_model = ncnn.Net()
    ncnn_model.load_param('si_mb_opt.param')
    ncnn_model.load_model('si_mb_opt.bin')

    ex = ncnn_model.create_extractor()
    ex.set_light_mode(True)
    ex.set_num_threads(4)

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        resized = cv2.resize(frame, (512, 512))
        image = resized.copy()
        image = image[:, :, ::-1]
        image = image.astype("float32").transpose(2, 0, 1)
        image = np.expand_dims(image, axis=0)
        mat_image = ncnn.Mat.from_pixels(
            image, ncnn.Mat.PixelType.PIXEL_BGR2RGB, 512, 512
        )
        start_time = time.time()
        ex.input('image', ncnn.Mat.from_pixels(image.data, ncnn.Mat.PixelType))
        mat_out1 = ex.extract("masks")
        mat_out2 = ex.extract("scores")
        mat_out3 = ex.extract("labels")
        print('inference time: %s sec' % (time.time() - start_time))
        outputs = (np.array(mat_out1), np.array(mat_out2), np.array(mat_out3))
        threading.Thread(target=masking_and_show, args=(resized, outputs)).run()

    cap.release()
    cv2.destroyAllWindows()
