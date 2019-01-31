#!/usr/bin/env bash
# make sure all the pythons files are the root level in the tar files, not under any folder

rm BiLSTM_CRF.tar.*
tar zcvf BiLSTM_CRF.tar.gz *