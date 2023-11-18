#!/usr/bin/env bash
rm -rf data-bin/iwslt14.tokenized.de-en_dev
cp -r data-bin/iwslt14.tokenized.de-en data-bin/iwslt14.tokenized.de-en_dev
rm data-bin/iwslt14.tokenized.de-en_dev/train.*
rm data-bin/iwslt14.tokenized.de-en_dev/test.*
mv data-bin/iwslt14.tokenized.de-en_dev/valid.de-en.en.idx data-bin/iwslt14.tokenized.de-en_dev/test.de-en.en.idx
mv data-bin/iwslt14.tokenized.de-en_dev/valid.de-en.de.idx data-bin/iwslt14.tokenized.de-en_dev/test.de-en.de.idx
mv data-bin/iwslt14.tokenized.de-en_dev/valid.de-en.en.bin data-bin/iwslt14.tokenized.de-en_dev/test.de-en.en.bin
mv data-bin/iwslt14.tokenized.de-en_dev/valid.de-en.de.bin data-bin/iwslt14.tokenized.de-en_dev/test.de-en.de.bin