# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import lmdb
import os
import pickle
from functools import lru_cache
import logging
import shutil
import time

logger = logging.getLogger(__name__)


# async ckp copy
def lmdb_data_copy_fun(src_path_dir, target_path_dir, epoch, split):
    db_path_src = os.path.join(src_path_dir, "{}_part_{}.lmdb".format(split, epoch))
    db_path_tgt = os.path.join(target_path_dir, "{}_part_{}.lmdb".format(split, epoch))

    if os.path.exists(db_path_tgt):
        return

    if not os.path.exists(db_path_src):
        logger.warning(f"please not that {db_path_src} not exists.")
        return

    shutil.copyfile(db_path_src, db_path_tgt)

    last_db_path_tgt = os.path.join(target_path_dir, "{}_part_{}.lmdb".format(split, epoch-2))
    if os.path.exists(last_db_path_tgt):
        os.remove(last_db_path_tgt)

    logger.info(f"finished async copy file from {db_path_src} to {db_path_tgt}")
    return


class LMDBDataset():
    def __init__(self, db_dir, split, epoch=1, max_epoch=1, lmdb_copy_thread=None, tmp_data_dir="/temp/"):
        self.db_path = os.path.join(db_dir, "{}.lmdb".format(split))
        assert os.path.isfile(self.db_path), "{} not found".format(self.db_path)
        env = self.connect_db(self.db_path)
        with env.begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))

    def connect_db(self, lmdb_path, save_to_self=False):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        if not save_to_self:
            return env
        else:
            self.env = env

    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if not hasattr(self, "env"):
            self.connect_db(self.db_path, save_to_self=True)
        datapoint_pickled = self.env.begin().get(f"{idx}".encode("ascii"))
        data = pickle.loads(datapoint_pickled)
        return data