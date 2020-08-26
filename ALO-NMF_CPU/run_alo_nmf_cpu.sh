./nmf -est_nmf_cpu -K 64 -tile_size 8 -data ../20newsgroups.txt -matrix_type 2 -V 26214 -D 11314 -niters 100
# ./nmf -est_nmf_cpu -K 256 -tile_size 16 -data ../20newsgroups.txt -matrix_type 2 -V 26214 -D 11314 -niters 100
# ./nmf -est_nmf_cpu -K 1024 -tile_size 64 -data ../20newsgroups.txt -matrix_type 2 -V 26214 -D 11314 -niters 100

# ./nmf -est_nmf_cpu -K 64 -tile_size 8 -data ../PIE.txt -matrix_type 1 -V 11554 -D 4096 -niters 100
# ./nmf -est_nmf_cpu -K 256 -tile_size 16 -data ../PIE.txt -matrix_type 1 -V 11554 -D 4096 -niters 100
# ./nmf -est_nmf_cpu -K 1024 -tile_size 64 -data ../PIE.txt -matrix_type 1 -V 11554 -D 4096 -niters 100

# K = 64, T = 8
# K = 128, T = 16
# K = 256, T = 16
# K = 512, T = 32
# K = 1024, T = 64

# matrix_type = 1 // Dense
# matrix_type = 2 // Sparse

# 20newsgroups.txt
# V = 26214 D = 11314
# TDT2.txt
# V = 36771 D = 10212
# p2p.txt
# V = 36675(36682) D = 36682
# MovieLens10M.txt
# V = 71567 D = 10677
# PIE.txt
# V = 11554 D = 4096
