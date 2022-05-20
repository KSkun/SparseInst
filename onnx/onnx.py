import onnxruntime as ort
import cv2
import numpy as np

if __name__ == '__main__':
    image = cv2.imread('../../hust1.jpg')
    image = image[:, :, ::-1]
    image = image.astype("float32").transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)

    ort_sess = ort.InferenceSession('../sparse_inst.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    outputs = ort_sess.run(None, {'image': image})

    output_image = cv2.imread('../../hust1.jpg')
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
    print(outputs)
    cv2.imwrite('hust1-result.jpg', output_image)
