For how to launch normal inference, see

/coc/testnvme/xzhang3205/vla-adaptation/inference/run_one_gr00t_n15.py
/coc/testnvme/xzhang3205/vla-adaptation/inference/run_gr00t_n15.sbatch

This is for your reference when writing static inference code. For static inference you should always use model `/coc/testnvme/xzhang3205/vla-adaptation/checkpoints/gr00t/gr00t-n1.5` and its statistic.json. You must not recompute statistics.json -- use that