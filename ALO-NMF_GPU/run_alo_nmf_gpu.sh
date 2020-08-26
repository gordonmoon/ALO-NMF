# ./nmf -est_nmf_gpu -K 1024 -tile_size 64 -data MovieLens10M.txt -matrix_type 2 -V 71567 -D 10677 -niters 1000
./nmf -est_nmf_gpu -K 1024 -tile_size 64 -data p2p.txt -matrix_type 2 -V 36675 -D 36682 -niters 1000
# ./nmf -est_nmf_gpu -K 1024 -tile_size 64 -data 20newsgroups.txt -matrix_type 2 -V 26214 -D 11314 -niters 1000
# ./nmf -est_nmf_gpu -K 64 -tile_size 8 -data TDT2.txt -matrix_type 2 -V 36771 -D 10212 -niters 1000
# ./nmf -est_nmf_gpu -K 64 -tile_size 8 -data PIE.txt -matrix_type 1 -V 11554 -D 4096 -niters 1000


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
# V = 70000 D = 30000 // Not working (wrong relative error and time)
# V = 60000 D = 40000 // Not working!
# V = 60000 D = 35000 // Not working!
# V = 60000 D = 30000 // Working!
# V = 50000 D = 30000 // 
# V = 40000 D = 30000 // Working!
# V = 30000 D = 30000 // 
# V = 30000 D = 20000 // Working!
# V = 28000 D = 20000 // Working!
# V = 27000 D = 20000 // 
# V = 25000 D = 20000 // Working!
# V = 20000 D = 20000 // Working!


# nips.txt
# V = 12419 D = 1500

# nytimes.txt
# V = 102660 D = 300000

# pubmed.txt
# V = 141043 D = 8200000