import fs from 'fs';
import fetch from 'node-fetch';

export default async function (url: string, path: string): Promise<void> {
  const stream = fs.createWriteStream(path);

  const response = await fetch(url);

  await new Promise((resolve) => {
    response.body.pipe(stream).on('close', () => resolve());
  });

  stream.close();
}
