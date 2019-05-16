import { ad } from "tns-core-modules/utils/utils";
declare var org: any;

const BYTES_PER_CHANNEL = 4;
export class Tflite {
    labelProb;
    labels;
    inputSize = 0;
    tfLite;

    loadModel(modelPath: string, labelsPath: string, numThreads: number) {
        return new Promise((resolve, reject) => {
            try {
                const assetManager = ad.getApplicationContext().getAssets();
                const fileDescriptor = assetManager.openFd(modelPath);
                const inputStream = new java.io.FileInputStream(
                    fileDescriptor.getFileDescriptor()
                );
                const fileChannel = inputStream.getChannel();
                const startOffset = fileDescriptor.getStartOffset();
                const declaredLength = fileDescriptor.getDeclaredLength();
                const buffer = fileChannel.map(
                    java.nio.channels.FileChannel.MapMode.READ_ONLY,
                    startOffset,
                    declaredLength
                );
                const tfliteOptions = new org.tensorflow.lite.Interpreter.Options();
                tfliteOptions.setNumThreads(numThreads);

                this.tfLite = new org.tensorflow.lite.Interpreter(
                    buffer,
                    tfliteOptions
                );

                this.loadLabels(assetManager, labelsPath);

                resolve();
            } catch (e) {
                reject(e);
            }
        });
    }

    loadLabels(assetManager, path) {
        try {
            const br = new java.io.BufferedReader(
                new java.io.InputStreamReader(assetManager.open(path))
            );
            this.labels = [];
            let line = br.readLine();
            while (line !== null) {
                this.labels.push(line);
                line = br.readLine();
            }
            this.labelProb = [];
            br.close();
        } catch (e) {
            throw "Failed to read label file: " + e;
        }
    }

    GetTopN(numResults, threshold) {
        const pq = new java.util.PriorityQueue(
            1,
            new java.util.Comparator({
                compare(lhs, rhs) {
                    return java.lang.Double.compare(
                        rhs.getDouble("confidence"),
                        lhs.getDouble("confidence")
                    );
                },
                equals(eq) {
                    return true;
                }
            })
        );

        for (let i = 0; i < this.labels.length; i++) {
            const confidence = this.labelProb[0][i];
            if (confidence > threshold) {
                const res = {
                    index: i,
                    label: this.labels.length > i ? this.labels[i] : "unknown",
                    confidence
                };
                pq.add(res);
            }
        }

        const results = [];
        const recognitionsSize = Math.min(pq.size(), numResults);
        for (let i = 0; i < recognitionsSize; i++) {
            results.push(pq.poll());
        }
        return results;
    }

    feedInputTensorImage(path, mean, std) {
        const tensor = this.tfLite.getInputTensor(0);
        this.inputSize = tensor.shape()[1];
        const inputChannels = tensor.shape()[3];

        const inputStream = new java.io.FileInputStream(
            path.replace("file://", "")
        );
        const bitmapRaw = android.graphics.BitmapFactory.decodeStream(
            inputStream
        );

        const matrix = this.getTransformationMatrix(
            bitmapRaw.getWidth(),
            bitmapRaw.getHeight(),
            this.inputSize,
            this.inputSize,
            false
        );

        const intValues = [];
        const bytePerChannel =
            tensor.dataType() == org.tensorflow.lite.DataType.UINT8
                ? 1
                : BYTES_PER_CHANNEL;
        const imgData = java.nio.ByteBuffer.allocateDirect(
            1 * this.inputSize * this.inputSize * inputChannels * bytePerChannel
        );
        imgData.order(java.nio.ByteOrder.nativeOrder());

        const bitmap = android.graphics.Bitmap.createBitmap(
            this.inputSize,
            this.inputSize,
            android.graphics.Bitmap.Config.ARGB_8888
        );
        const canvas: android.graphics.Canvas = new android.graphics.Canvas(
            bitmap
        );
        canvas.drawBitmap(bitmapRaw, matrix, null);
        bitmap.getPixels(
            intValues,
            0,
            bitmap.getWidth(),
            0,
            0,
            bitmap.getWidth(),
            bitmap.getHeight()
        );

        let pixel = 0;
        for (let i = 0; i < this.inputSize; i++) {
            for (let j = 0; j < this.inputSize; j++) {
                const pixelValue = intValues[pixel++];
                if (tensor.dataType() == org.tensorflow.lite.DataType.FLOAT32) {
                    imgData.putFloat(
                        (((pixelValue >> 16) & 0xff) - mean) / std
                    );
                    imgData.putFloat((((pixelValue >> 8) & 0xff) - mean) / std);
                    imgData.putFloat(((pixelValue & 0xff) - mean) / std);
                } else {
                    imgData.put((pixelValue >> 16) & 0xff);
                    imgData.put((pixelValue >> 8) & 0xff);
                    imgData.put(pixelValue & 0xff);
                }
            }
        }

        return imgData;
    }

    getTransformationMatrix(
        srcWidth,
        srcHeight,
        dstWidth,
        dstHeight,
        maintainAspectRatio
    ) {
        const matrix = new android.graphics.Matrix();

        if (srcWidth != dstWidth || srcHeight != dstHeight) {
            const scaleFactorX = dstWidth / srcWidth;
            const scaleFactorY = dstHeight / srcHeight;

            if (maintainAspectRatio) {
                const scaleFactor = Math.max(scaleFactorX, scaleFactorY);
                matrix.postScale(scaleFactor, scaleFactor);
            } else {
                matrix.postScale(scaleFactorX, scaleFactorY);
            }
        }

        matrix.invert(new android.graphics.Matrix());
        return matrix;
    }
}
