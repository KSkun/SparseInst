import cv2
import onnxruntime as ort
import numpy as np

if __name__ == '__main__':
    ort_sess = ort.InferenceSession('sparse_inst_opt.onnx',
                                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

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
        outputs = ort_sess.run(None, {'image': image})

        output_image = resized.copy()
        for i in range(100):
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

        cv2.imshow('SparseInst', output_image)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
