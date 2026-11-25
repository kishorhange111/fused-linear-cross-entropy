GPU **Tesla T4**. Head: d = 896, V = 151,936 (Qwen2.5-0.5B), fp16, loss forward + backward.

### Peak extra memory (GiB) and time (ms)

| tokens | fp32 logits alone | naive | fused c=512 | fused c=1024 | fused c=2048 | fused c=4096 |
|---|---|---|---|---|---|---|
| 2048 | 1.16 | 3.22 / 115 | 1.16 / 215 | 1.70 / 192 | 2.57 / 183 | 2.57 / 184 |
| 4096 | 2.32 | 6.69 / 252 | 1.16 / 427 | 1.70 / 384 | 3.15 / 366 | 4.89 / 380 |
| 8192 | 4.64 | 13.64 / 551 | 1.16 / 845 | 1.70 / 749 | 3.15 / 709 | 6.05 / 722 |
| 16384 | 9.27 | OOM / OOM | 1.16 / 1649 | 1.70 / 1471 | 3.15 / 1403 | 6.05 / 1447 |

### Qwen2.5-0.5B training step, seq 512, gradient checkpointing, fp16 autocast, SGD

| loss | max batch (powers of 2) | peak GiB | tokens/s |
|---|---|---|---|
| naive | 4 | 9.38 | 4003 |
| fused | 8 | 12.91 | 3705 |
