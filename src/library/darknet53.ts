/* eslint-disable no-restricted-syntax */
import R from 'ramda';
import { Layer, convolutional, residualBlock } from './common';

export default function (input: Layer): { routeOne: Layer, routeTwo: Layer, routeThree: Layer } {
  let conv = input;

  conv = convolutional(conv, { filtersShape: [3, 3, 3, 32] });
  conv = convolutional(conv, { filtersShape: [3, 3, 32, 64], downsample: true });
  conv = residualBlock(conv, { numChannels: 64, numFiltersOne: 32, numFiltersTwo: 64 });

  conv = convolutional(conv, { filtersShape: [3, 3, 64, 128], downsample: true });

  R.range(0, 2).forEach(() => {
    conv = residualBlock(conv, { numChannels: 128, numFiltersOne: 64, numFiltersTwo: 128 });
  });

  conv = convolutional(conv, { filtersShape: [3, 3, 128, 256], downsample: true });

  R.range(0, 8).forEach(() => {
    conv = residualBlock(conv, { numChannels: 256, numFiltersOne: 128, numFiltersTwo: 256 });
  });

  const routeOne = conv;
  conv = convolutional(conv, { filtersShape: [3, 3, 256, 512], downsample: true });

  R.range(0, 8).forEach(() => {
    conv = residualBlock(conv, { numChannels: 512, numFiltersOne: 256, numFiltersTwo: 512 });
  });

  const routeTwo = conv;
  conv = convolutional(conv, { filtersShape: [3, 3, 512, 1024], downsample: true });

  R.range(0, 4).forEach(() => {
    conv = residualBlock(conv, { numChannels: 1024, numFiltersOne: 512, numFiltersTwo: 1024 });
  });

  return {
    routeOne,
    routeTwo,
    routeThree: conv,
  };
}

R.range(2);
