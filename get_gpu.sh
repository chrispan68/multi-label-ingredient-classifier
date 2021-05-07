salloc -t 16:00:00 --gres=gpu:1 -c 4 -A allcs --mem=24G srun --pty  $SHELL -l
