# ./nmf -est_nmf_cpu -K 64 -tile_size 8 -data 20newsgroups.txt -matrix_type 2 -V 26214 -D 11314 -niters 100
./nmf -est_nmf_cpu -K 256 -tile_size 16 -data 20newsgroups.txt -matrix_type 2 -V 26214 -D 11314 -niters 10
# ./nmf -est_nmf_cpu -K 1024 -tile_size 64 -data 20newsgroups.txt -matrix_type 2 -V 26214 -D 11314 -niters 1000
# ./nmf -est_nmf_cpu -K 64 -tile_size 8 -data TDT2.txt -matrix_type 2 -V 36771 -D 10212 -niters 1000
# ./nmf -est_nmf_cpu -K 128 -tile_size 16 -data TDT2.txt -matrix_type 2 -V 36771 -D 10212 -niters 100
# ./nmf -est_nmf_cpu -K 256 -tile_size 16 -data TDT2.txt -matrix_type 2 -V 36771 -D 10212 -niters 1000
# ./nmf -est_nmf_cpu -K 512 -tile_size 32 -data TDT2.txt -matrix_type 2 -V 36771 -D 10212 -niters 100
# ./nmf -est_nmf_cpu -K 1024 -tile_size 64 -data TDT2.txt -matrix_type 2 -V 36771 -D 10212 -niters 1000
# ./nmf -est_nmf_cpu -K 64 -tile_size 8 -data MovieLens10M.txt -matrix_type 2 -V 71567 -D 10677 -niters 1000
# ./nmf -est_nmf_cpu -K 256 -tile_size 16 -data MovieLens10M.txt -matrix_type 2 -V 71567 -D 10677 -niters 1000
# ./nmf -est_nmf_cpu -K 1024 -tile_size 64 -data MovieLens10M.txt -matrix_type 2 -V 71567 -D 10677 -niters 1000
# ./nmf -est_nmf_cpu -K 64 -tile_size 8 -data p2p.txt -matrix_type 2 -V 36675 -D 36682 -niters 1000
# ./nmf -est_nmf_cpu -K 256 -tile_size 16 -data p2p.txt -matrix_type 2 -V 36675 -D 36682 -niters 1000
# ./nmf -est_nmf_cpu -K 1024 -tile_size 64 -data p2p.txt -matrix_type 2 -V 36675 -D 36682 -niters 100
# ./nmf -est_nmf_cpu -K 64 -tile_size 8 -data PIE.txt -matrix_type 1 -V 11554 -D 4096 -niters 1000
# ./nmf -est_nmf_cpu -K 128 -tile_size 16 -data PIE.txt -matrix_type 1 -V 11554 -D 4096 -niters 100
# ./nmf -est_nmf_cpu -K 256 -tile_size 32 -data PIE.txt -matrix_type 1 -V 11554 -D 4096 -niters 100
# ./nmf -est_nmf_cpu -K 512 -tile_size 32 -data PIE.txt -matrix_type 1 -V 11554 -D 4096 -niters 100
# ./nmf -est_nmf_cpu -K 1024 -tile_size 64 -data PIE.txt -matrix_type 1 -V 11554 -D 4096 -niters 1000

# K = 64, T = 8
# K = 128, T = 16
# K = 256, T = 16
# K = 512, T = 32
# K = 1024, T = 64

# ./nmf -est_nmf_cpu -K 256 -tile_size 2 -data MovieLens10M.txt -matrix_type 2 -V 71567 -D 10677 -niters 10
# ./nmf -est_nmf_cpu -K 256 -tile_size 4 -data MovieLens10M.txt -matrix_type 2 -V 71567 -D 10677 -niters 10
# ./nmf -est_nmf_cpu -K 256 -tile_size 8 -data MovieLens10M.txt -matrix_type 2 -V 71567 -D 10677 -niters 10
# ./nmf -est_nmf_cpu -K 256 -tile_size 16 -data MovieLens10M.txt -matrix_type 2 -V 71567 -D 10677 -niters 10
# ./nmf -est_nmf_cpu -K 256 -tile_size 32 -data MovieLens10M.txt -matrix_type 2 -V 71567 -D 10677 -niters 10
# ./nmf -est_nmf_cpu -K 256 -tile_size 64 -data MovieLens10M.txt -matrix_type 2 -V 71567 -D 10677 -niters 10
# ./nmf -est_nmf_cpu -K 256 -tile_size 128 -data MovieLens10M.txt -matrix_type 2 -V 71567 -D 10677 -niters 10
# ./nmf -est_nmf_cpu -K 256 -tile_size 256 -data MovieLens10M.txt -matrix_type 2 -V 71567 -D 10677 -niters 10


# ./nmf -est_nmf_cpu -K 256 -tile_size 2 -data p2p.txt -matrix_type 2 -V 36675 -D 36682 -niters 10
# ./nmf -est_nmf_cpu -K 256 -tile_size 4 -data p2p.txt -matrix_type 2 -V 36675 -D 36682 -niters 10
# ./nmf -est_nmf_cpu -K 256 -tile_size 8 -data p2p.txt -matrix_type 2 -V 36675 -D 36682 -niters 10
# ./nmf -est_nmf_cpu -K 256 -tile_size 16 -data p2p.txt -matrix_type 2 -V 36675 -D 36682 -niters 10
# ./nmf -est_nmf_cpu -K 256 -tile_size 32 -data p2p.txt -matrix_type 2 -V 36675 -D 36682 -niters 10
# ./nmf -est_nmf_cpu -K 256 -tile_size 64 -data p2p.txt -matrix_type 2 -V 36675 -D 36682 -niters 10
# ./nmf -est_nmf_cpu -K 256 -tile_size 128 -data p2p.txt -matrix_type 2 -V 36675 -D 36682 -niters 10
# ./nmf -est_nmf_cpu -K 256 -tile_size 256 -data p2p.txt -matrix_type 2 -V 36675 -D 36682 -niters 10


# completed constructor::A::36675x36682

# matrix_type = 1 // Dense
# matrix_type = 2 // Sparse

# p2p.txt
# V = 36675(36682) D = 36682
# MovieLens10M.txt
# V = 71567 D = 10677
# 20newsgroups.txt
# V = 26214 D = 11314
# TDT2.txt
# V = 36771 D = 10212
# Reuters.txt
# V = 18933 D = 8293
# ATandT.txt
# V = 400 D = 10304
# PIE.txt
# V = 11554 D = 4096

# Dense_Synthetic_2000_1000_PLNMF.txt
# V = 100000 D = 50000 // Not working!
# V = 80000 D = 40000 // Not working!
# V = 70000 D = 35000 // Not working!
# V = 70000 D = 30000 // Working!
# V = 60000 D = 30000 // Working!

# nips.txt
# V = 12419 D = 1500
# nytimes.txt
# V = 102660 D = 300000
# pubmed.txt
# V = 141043 D = 8200000