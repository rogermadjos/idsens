/* eslint-disable @typescript-eslint/no-explicit-any */
import tf from '@tensorflow/tfjs-node-gpu';
import R from 'ramda';

export type Layer = tf.Tensor | tf.Tensor[] | tf.SymbolicTensor | tf.SymbolicTensor[];

export function convolutional(
  input: Layer,
  args: {
    filtersShape: [number, number, number, number];
    downsample?: boolean;
    activate?: boolean;
    batchNormalization?: boolean;
  },
): Layer {
  const {
    filtersShape, batchNormalization, downsample, activate,
  } = R.mergeDeepLeft(args, {
    downsample: false,
    activate: true,
    batchNormalization: true,
  });

  let conv = input;

  let strides = 1;
  let padding: 'same' | 'valid' = 'same';

  if (downsample) {
    conv = tf.layers.zeroPadding2d({
      padding: [[1, 0], [1, 0]],
    }).apply(conv);

    strides = 2;
    padding = 'valid';
  }

  conv = tf.layers.conv2d({
    filters: filtersShape[3],
    kernelSize: filtersShape[0],
    strides,
    padding,
    useBias: !batchNormalization,
    kernelRegularizer: tf.regularizers.l2({ l2: 0.0005 }),
    kernelInitializer: tf.initializers.randomNormal({ stddev: 0.01 }),
    biasInitializer: tf.initializers.constant({ value: 0 }),
  }).apply(conv);

  if (batchNormalization) {
    conv = tf.layers.batchNormalization().apply(conv);
  }

  if (activate) {
    conv = tf.layers.leakyReLU({ alpha: 0.1 }).apply(conv);
  }

  return conv;
}

export function residualBlock(
  input: Layer,
  args: {
    numChannels: number;
    numFiltersOne: number;
    numFiltersTwo: number;
  },
): Layer {
  let conv = convolutional(input, { filtersShape: [1, 1, args.numChannels, args.numFiltersOne] });
  conv = convolutional(conv, { filtersShape: [3, 3, args.numFiltersOne, args.numFiltersTwo] });

  return tf.layers.add().apply([input as any, conv as any]);
}
