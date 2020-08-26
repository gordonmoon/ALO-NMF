void cuda_mul_W_old_temp_q(double* _d_W_old, double* _d_W_new, double* _d_temp_q, int V, int K);
void phase_two_H(double* _d_H_old, double* _d_H_new, double* _d_temp_s, double* _d_temp_r, int t, int tile_id, int Tile_size, int D, int K, double eps);
void phase_two_W(double* _d_W_old, double* _d_W_new, double* _d_temp_q, double* _d_temp_p, double* _d_ss_col, int t, int tile_id, int Tile_size, int V, int K, double eps);
void cuda_div_W_new_col(double* _d_W_new, double* _d_ss_col, int t, int V);