/* eslint-disable @typescript-eslint/no-explicit-any */
import fs from 'fs';
import path from 'path';
import { promisify } from 'util';
import Bluebird from 'bluebird';
import ora from 'ora';
import R from 'ramda';
import download from '../library/download';

async function main() {
  const spinner = ora().start();

  spinner.text = 'Parsing ./dataset/raw.json';
  const data = JSON.parse(await promisify(fs.readFile)(path.resolve(__dirname, '../dataset/raw.json'), 'utf8'));

  await promisify(fs.rmdir)(path.resolve(__dirname, '../dataset/images'), { recursive: true });

  await promisify(fs.mkdir)(path.resolve(__dirname, '../dataset/images'));

  spinner.text = 'Downloading images';
  await Bluebird.map(data, async (item: Record<string, unknown>) => {
    const id = item.ID as string;
    const image = item['Labeled Data'] as string;

    await download(image, path.resolve(__dirname, `../dataset/images/${id}`));
  }, { concurrency: 5 });

  spinner.text = 'Creating ./dataset/labels.json';
  const labels = R.compose(
    R.flatten as any,
    R.map((item: any) => R.map((object) => ({
      id: object.featureId,
      image: item.image,
      box: object.bbox,
    }), item.objects)),
    R.map((item: any) => ({
      image: item.ID,
      objects: R.path(['Label', 'objects'], item),
    })),
  )(data);
  await promisify(fs.writeFile)(path.resolve(__dirname, '../dataset/labels.json'), JSON.stringify(labels, null, '  '));

  spinner.stop();
}

main().catch((err) => {
  console.error(err);
  process.exit(-1);
});
