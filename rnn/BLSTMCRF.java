
import java.util.Arrays;
import java.util.Random;


public class BLSTMCRF {

	//forward state
	double w_ni_f[][] = null;
	double w_nf_f[][] = null;
	double w_no_f[][] = null;
	double w_nz_f[][] = null;	
	double w_hi_f[][] = null;
	double w_hf_f[][] = null;
	double w_ho_f[][] = null;
	double w_hz_f[][] = null;
	
	double w_out_f[][] = null;
	
	double w_ni_f_[][] = null;
	double w_nf_f_[][] = null;
	double w_no_f_[][] = null;
	double w_nz_f_[][] = null;	
	double w_hi_f_[][] = null;
	double w_hf_f_[][] = null;
	double w_ho_f_[][] = null;
	double w_hz_f_[][] = null;
	
	double w_out_f_[][] = null;
	
	double b_i_f[] = null;
	double b_f_f[] = null;
	double b_o_f[] = null;
	double b_z_f[] = null;
	
	double b_out[] = null;
	
	double b_i_f_[] = null;
	double b_f_f_[] = null;
	double b_o_f_[] = null;
	double b_z_f_[] = null;
	
	double b_out_[] = null;
	
	double input_i_f[][]  = null;
	double forget_f_f[][] = null;
	double block_z_f[][]  = null;
	double cell_c_f[][]   = null;
	double cell_c_ts_f[][]= null;
	double output_o_f[][] = null;
	double output_y_f[][] = null;	
	double output_out[][] = null;
	double hidden_pre_f[] = null;
	double cell_pre_f[]   = null;

	double output_[][] = null;
	double crf_output[][] = null;
	
	double error_out[][] = null;	
	double error_y_f[][] = null;
	double error_o_f[][] = null;
	double error_c_f[][] = null;
	double error_z_f[][] = null;
	double error_f_f[][] = null;
	double error_i_f[][] = null;
	
	double cache_w_ni_m_f[][] = null;
	double cache_w_ni_v_f[][] = null;
	double cache_w_nf_m_f[][] = null;
	double cache_w_nf_v_f[][] = null;
	double cache_w_no_m_f[][] = null;
	double cache_w_no_v_f[][] = null;
	double cache_w_nz_m_f[][] = null;
	double cache_w_nz_v_f[][] = null;
	double cache_w_hi_m_f[][] = null;
	double cache_w_hi_v_f[][] = null;
	double cache_w_hf_m_f[][] = null;
	double cache_w_hf_v_f[][] = null;
	double cache_w_ho_m_f[][] = null;
	double cache_w_ho_v_f[][] = null;
	double cache_w_hz_m_f[][] = null;
	double cache_w_hz_v_f[][] = null;
	
	double cache_b_i_m_f[] = null;
	double cache_b_i_v_f[] = null;
	double cache_b_f_m_f[] = null;
	double cache_b_f_v_f[] = null;
	double cache_b_o_m_f[] = null;
	double cache_b_o_v_f[] = null;
	double cache_b_z_m_f[] = null;
	double cache_b_z_v_f[] = null;
	
	double cache_w_out_m_f[][] = null;
	double cache_w_out_v_f[][] = null;
	double cache_b_out_m[] = null;
	double cache_b_out_v[] = null;
	//back state
	double w_ni_b[][] = null;
	double w_nf_b[][] = null;
	double w_no_b[][] = null;
	double w_nz_b[][] = null;	
	double w_hi_b[][] = null;
	double w_hf_b[][] = null;
	double w_ho_b[][] = null;
	double w_hz_b[][] = null;
	
	double w_out_b[][] = null;
	
	double w_ni_b_[][] = null;
	double w_nf_b_[][] = null;
	double w_no_b_[][] = null;
	double w_nz_b_[][] = null;	
	double w_hi_b_[][] = null;
	double w_hf_b_[][] = null;
	double w_ho_b_[][] = null;
	double w_hz_b_[][] = null;
	
	double w_out_b_[][] = null;
	
	double b_i_b[] = null;
	double b_f_b[] = null;
	double b_o_b[] = null;
	double b_z_b[] = null;
	
	double b_i_b_[] = null;
	double b_f_b_[] = null;
	double b_o_b_[] = null;
	double b_z_b_[] = null;
	
	double input_i_b[][]  = null;
	double forget_f_b[][] = null;
	double block_z_b[][]  = null;
	double cell_c_b[][]   = null;
	double cell_c_ts_b[][]= null;
	double output_o_b[][] = null;
	double output_y_b[][] = null;
	
	double hidden_pre_b[] = null;
	double cell_pre_b[]   = null;
	
	double error_y_b[][] = null;
	double error_o_b[][] = null;
	double error_c_b[][] = null;
	double error_z_b[][] = null;
	double error_f_b[][] = null;
	double error_i_b[][] = null;
	
	double cache_w_ni_m_b[][] = null;
	double cache_w_ni_v_b[][] = null;
	double cache_w_nf_m_b[][] = null;
	double cache_w_nf_v_b[][] = null;
	double cache_w_no_m_b[][] = null;
	double cache_w_no_v_b[][] = null;
	double cache_w_nz_m_b[][] = null;
	double cache_w_nz_v_b[][] = null;
	double cache_w_hi_m_b[][] = null;
	double cache_w_hi_v_b[][] = null;
	double cache_w_hf_m_b[][] = null;
	double cache_w_hf_v_b[][] = null;
	double cache_w_ho_m_b[][] = null;
	double cache_w_ho_v_b[][] = null;
	double cache_w_hz_m_b[][] = null;
	double cache_w_hz_v_b[][] = null;
	
	double cache_b_i_m_b[] = null;
	double cache_b_i_v_b[] = null;
	double cache_b_f_m_b[] = null;
	double cache_b_f_v_b[] = null;
	double cache_b_o_m_b[] = null;
	double cache_b_o_v_b[] = null;
	double cache_b_z_m_b[] = null;
	double cache_b_z_v_b[] = null;
	
	double cache_w_out_m_b[][] = null;
	double cache_w_out_v_b[][] = null;
	
	double trs[][] = null;

	int input_len;
	int hidden_len;
	int output_len;
	int N;
	
	int epochs;
	int mini_batch;
	int unfold_size;
	int unfold_count = 120;
	double decay_rate;
	double learning_rate;
	double eps;
	double drop;
	double dropout_i[][] = null;
	double dropout_h[][] = null;
	
	double beta_adam1;
	double beta_adam2;
	
	boolean isadam;
	DoubleMax dm = null;
	public BLSTMCRF(int input_len, int hidden_len, int output_len,
				int epochs, int mini_batch, int unfold_size,
				double decay_rate, double learning_rate, double eps,
				double beta_adam1, double beta_adam2, double drop ){
		this.input_len = input_len;
		this.hidden_len = hidden_len;
		this.output_len = output_len;
		this.epochs = epochs;
		this.mini_batch = mini_batch;
		this.unfold_size = unfold_size;
		this.decay_rate = decay_rate;
		this.learning_rate = learning_rate;
		this.eps = eps;
		this.beta_adam1 = beta_adam1;
		this.beta_adam2 = beta_adam2;
		this.drop = drop;
	}
	public void net(boolean isadam){
		this.isadam = isadam;
		w_ni_f = new double[hidden_len][input_len];
		w_nf_f = new double[hidden_len][input_len];
		w_no_f = new double[hidden_len][input_len];
		w_nz_f = new double[hidden_len][input_len];		
		w_hi_f = new double[hidden_len][hidden_len];
		w_hf_f = new double[hidden_len][hidden_len];
		w_ho_f = new double[hidden_len][hidden_len];
		w_hz_f = new double[hidden_len][hidden_len];
		w_out_f = new double[output_len][hidden_len];
		
		w_ni_f_ = new double[hidden_len][input_len];
		w_nf_f_ = new double[hidden_len][input_len];
		w_no_f_ = new double[hidden_len][input_len];
		w_nz_f_ = new double[hidden_len][input_len];		
		w_hi_f_ = new double[hidden_len][hidden_len];
		w_hf_f_ = new double[hidden_len][hidden_len];
		w_ho_f_ = new double[hidden_len][hidden_len];
		w_hz_f_ = new double[hidden_len][hidden_len];
		w_out_f_ = new double[output_len][hidden_len];
		
		b_i_f = new double[hidden_len];
		b_f_f = new double[hidden_len];
		b_o_f = new double[hidden_len];
		b_z_f = new double[hidden_len];
		b_out = new double[output_len];
		
		b_i_f_ = new double[hidden_len];
		b_f_f_ = new double[hidden_len];
		b_o_f_ = new double[hidden_len];
		b_z_f_ = new double[hidden_len];
		b_out_ = new double[output_len];
		
		input_i_f  = new double[unfold_count][hidden_len];
		forget_f_f = new double[unfold_count][hidden_len];
		block_z_f  = new double[unfold_count][hidden_len];
		cell_c_f   = new double[unfold_count][hidden_len];
		cell_c_ts_f= new double[unfold_count][hidden_len];
		output_o_f = new double[unfold_count][hidden_len];
		output_y_f = new double[unfold_count][hidden_len];
		output_out = new double[unfold_count][output_len];
		output_ = new double[unfold_count][output_len];
		crf_output = new double[unfold_count][output_len];
		hidden_pre_f = new double[hidden_len];
		cell_pre_f   = new double[hidden_len];
		
		error_out = new double[unfold_count][output_len];
		error_y_f = new double[unfold_count][hidden_len];
		error_o_f = new double[unfold_count][hidden_len];
		error_c_f = new double[unfold_count][hidden_len];
		error_z_f = new double[unfold_count][hidden_len];
		error_f_f = new double[unfold_count][hidden_len];
		error_i_f = new double[unfold_count][hidden_len];
		
		cache_w_ni_m_f = new double[hidden_len][input_len];
		cache_w_ni_v_f = new double[hidden_len][input_len];
		cache_w_nf_m_f = new double[hidden_len][input_len];
		cache_w_nf_v_f = new double[hidden_len][input_len];
		cache_w_no_m_f = new double[hidden_len][input_len];
		cache_w_no_v_f = new double[hidden_len][input_len];
		cache_w_nz_m_f = new double[hidden_len][input_len];
		cache_w_nz_v_f = new double[hidden_len][input_len];
		
		cache_w_hi_m_f = new double[hidden_len][hidden_len];
		cache_w_hi_v_f = new double[hidden_len][hidden_len];
		cache_w_hf_m_f = new double[hidden_len][hidden_len];
		cache_w_hf_v_f = new double[hidden_len][hidden_len];
		cache_w_ho_m_f = new double[hidden_len][hidden_len];
		cache_w_ho_v_f = new double[hidden_len][hidden_len];
		cache_w_hz_m_f = new double[hidden_len][hidden_len];
		cache_w_hz_v_f = new double[hidden_len][hidden_len];
		
		cache_b_i_m_f = new double[hidden_len];
		cache_b_i_v_f = new double[hidden_len];
		cache_b_f_m_f = new double[hidden_len];
		cache_b_f_v_f = new double[hidden_len];
		cache_b_o_m_f = new double[hidden_len];
		cache_b_o_v_f = new double[hidden_len];
		cache_b_z_m_f = new double[hidden_len];
		cache_b_z_v_f = new double[hidden_len];
		
		cache_w_out_m_f = new double[output_len][hidden_len];
		cache_w_out_v_f = new double[output_len][hidden_len];
		cache_b_out_m = new double[output_len];
		cache_b_out_v = new double[output_len];
		
		w_ni_b = new double[hidden_len][input_len];
		w_nf_b = new double[hidden_len][input_len];
		w_no_b = new double[hidden_len][input_len];
		w_nz_b = new double[hidden_len][input_len];		
		w_hi_b = new double[hidden_len][hidden_len];
		w_hf_b = new double[hidden_len][hidden_len];
		w_ho_b = new double[hidden_len][hidden_len];
		w_hz_b = new double[hidden_len][hidden_len];
		w_out_b = new double[output_len][hidden_len];
		
		w_ni_b_ = new double[hidden_len][input_len];
		w_nf_b_ = new double[hidden_len][input_len];
		w_no_b_ = new double[hidden_len][input_len];
		w_nz_b_ = new double[hidden_len][input_len];		
		w_hi_b_ = new double[hidden_len][hidden_len];
		w_hf_b_ = new double[hidden_len][hidden_len];
		w_ho_b_ = new double[hidden_len][hidden_len];
		w_hz_b_ = new double[hidden_len][hidden_len];
		w_out_b_ = new double[output_len][hidden_len];
		
		b_i_b = new double[hidden_len];
		b_f_b = new double[hidden_len];
		b_o_b = new double[hidden_len];
		b_z_b = new double[hidden_len];
		
		b_i_b_ = new double[hidden_len];
		b_f_b_ = new double[hidden_len];
		b_o_b_ = new double[hidden_len];
		b_z_b_ = new double[hidden_len];
		
		input_i_b  = new double[unfold_count][hidden_len];
		forget_f_b = new double[unfold_count][hidden_len];
		block_z_b  = new double[unfold_count][hidden_len];
		cell_c_b   = new double[unfold_count][hidden_len];
		cell_c_ts_b= new double[unfold_count][hidden_len];
		output_o_b = new double[unfold_count][hidden_len];
		output_y_b = new double[unfold_count][hidden_len];
		
		hidden_pre_b = new double[hidden_len];
		cell_pre_b   = new double[hidden_len];
		
		error_y_b = new double[unfold_count][hidden_len];
		error_o_b = new double[unfold_count][hidden_len];
		error_c_b = new double[unfold_count][hidden_len];
		error_z_b = new double[unfold_count][hidden_len];
		error_f_b = new double[unfold_count][hidden_len];
		error_i_b = new double[unfold_count][hidden_len];
		
		cache_w_ni_m_b = new double[hidden_len][input_len];
		cache_w_ni_v_b = new double[hidden_len][input_len];
		cache_w_nf_m_b = new double[hidden_len][input_len];
		cache_w_nf_v_b = new double[hidden_len][input_len];
		cache_w_no_m_b = new double[hidden_len][input_len];
		cache_w_no_v_b = new double[hidden_len][input_len];
		cache_w_nz_m_b = new double[hidden_len][input_len];
		cache_w_nz_v_b = new double[hidden_len][input_len];
		
		cache_w_hi_m_b = new double[hidden_len][hidden_len];
		cache_w_hi_v_b = new double[hidden_len][hidden_len];
		cache_w_hf_m_b = new double[hidden_len][hidden_len];
		cache_w_hf_v_b = new double[hidden_len][hidden_len];
		cache_w_ho_m_b = new double[hidden_len][hidden_len];
		cache_w_ho_v_b = new double[hidden_len][hidden_len];
		cache_w_hz_m_b = new double[hidden_len][hidden_len];
		cache_w_hz_v_b = new double[hidden_len][hidden_len];
		
		cache_b_i_m_b = new double[hidden_len];
		cache_b_i_v_b = new double[hidden_len];
		cache_b_f_m_b = new double[hidden_len];
		cache_b_f_v_b = new double[hidden_len];
		cache_b_o_m_b = new double[hidden_len];
		cache_b_o_v_b = new double[hidden_len];
		cache_b_z_m_b = new double[hidden_len];
		cache_b_z_v_b = new double[hidden_len];
		
		cache_w_out_m_b = new double[output_len][hidden_len];
		cache_w_out_v_b = new double[output_len][hidden_len];
		if(!isadam){
			Arrays.fill(cache_b_i_m_b, 1);
			Arrays.fill(cache_b_i_m_f, 1);
			Arrays.fill(cache_b_f_m_b, 1);
			Arrays.fill(cache_b_f_m_f, 1);
			Arrays.fill(cache_b_o_m_b, 1);
			Arrays.fill(cache_b_o_m_f, 1);
			Arrays.fill(cache_b_z_m_b, 1);
			Arrays.fill(cache_b_z_m_f, 1);
			Arrays.fill(cache_b_out_m, 1);
			for(int i=0; i<hidden_len; i++){
				Arrays.fill(cache_w_ni_m_b[i], 1);
				Arrays.fill(cache_w_ni_m_f[i], 1);
				Arrays.fill(cache_w_hi_m_b[i], 1);
				Arrays.fill(cache_w_hi_m_f[i], 1);
				Arrays.fill(cache_w_nf_m_b[i], 1);
				Arrays.fill(cache_w_nf_m_f[i], 1);
				Arrays.fill(cache_w_hf_m_b[i], 1);
				Arrays.fill(cache_w_hf_m_f[i], 1);
				Arrays.fill(cache_w_no_m_b[i], 1);
				Arrays.fill(cache_w_no_m_f[i], 1);
				Arrays.fill(cache_w_ho_m_b[i], 1);
				Arrays.fill(cache_w_ho_m_f[i], 1);
				Arrays.fill(cache_w_nz_m_b[i], 1);
				Arrays.fill(cache_w_nz_m_f[i], 1);
				Arrays.fill(cache_w_hz_m_b[i], 1);
				Arrays.fill(cache_w_hz_m_f[i], 1);
			}
			for(int i=0; i<output_len; i++){
				Arrays.fill(cache_w_out_m_b[i], 1);
				Arrays.fill(cache_w_out_m_f[i], 1);
			}
		}
		dm = new DoubleMax();
	}
	public void init(boolean gass){
		Random rand = new Random();
		if(gass){
			for(int i=0; i<hidden_len; i++){
				for(int j=0; j<input_len; j++){
					w_ni_f[i][j] = rand.nextGaussian()/100;
					w_no_f[i][j] = rand.nextGaussian()/100;
					w_nf_f[i][j] = rand.nextGaussian()/100;
					w_nz_f[i][j] = rand.nextGaussian()/100;
					
					w_ni_b[i][j] = rand.nextGaussian()/100;
					w_no_b[i][j] = rand.nextGaussian()/100;
					w_nf_b[i][j] = rand.nextGaussian()/100;
					w_nz_b[i][j] = rand.nextGaussian()/100;
					
				}
				for(int j=0; j<hidden_len; j++){
					w_hi_f[i][j] = rand.nextGaussian()/100;
					w_ho_f[i][j] = rand.nextGaussian()/100;
					w_hf_f[i][j] = rand.nextGaussian()/100;
					w_hz_f[i][j] = rand.nextGaussian()/100;
					
					w_hi_b[i][j] = rand.nextGaussian()/100;
					w_ho_b[i][j] = rand.nextGaussian()/100;
					w_hf_b[i][j] = rand.nextGaussian()/100;
					w_hz_b[i][j] = rand.nextGaussian()/100;
				}
			}
			for(int i=0; i<output_len; i++){
				for(int j=0; j<hidden_len; j++){
					w_out_f[i][j] = rand.nextGaussian()/100;
					w_out_b[i][j] = rand.nextGaussian()/100;
				}
			}
		}else{
			for(int i=0; i<hidden_len; i++){
				for(int j=0; j<input_len; j++){
					w_ni_f[i][j] = rand.nextDouble()/100;
					w_no_f[i][j] = rand.nextDouble()/100;
					w_nf_f[i][j] = rand.nextDouble()/100;
					w_nz_f[i][j] = rand.nextDouble()/100;
					
					w_ni_b[i][j] = rand.nextDouble()/100;
					w_no_b[i][j] = rand.nextDouble()/100;
					w_nf_b[i][j] = rand.nextDouble()/100;
					w_nz_b[i][j] = rand.nextDouble()/100;
					
				}
				for(int j=0; j<hidden_len; j++){
					w_hi_f[i][j] = rand.nextDouble()/100;
					w_ho_f[i][j] = rand.nextDouble()/100;
					w_hf_f[i][j] = rand.nextDouble()/100;
					w_hz_f[i][j] = rand.nextDouble()/100;
					
					w_hi_b[i][j] = rand.nextDouble()/100;
					w_ho_b[i][j] = rand.nextDouble()/100;
					w_hf_b[i][j] = rand.nextDouble()/100;
					w_hz_b[i][j] = rand.nextDouble()/100;
				}
			}
			for(int i=0; i<output_len; i++){
				for(int j=0; j<hidden_len; j++){
					w_out_f[i][j] = rand.nextDouble()/100;
					w_out_b[i][j] = rand.nextDouble()/100;
				}
			}
		}
		Arrays.fill(b_i_f, 0);
		Arrays.fill(b_i_b, 0);
		Arrays.fill(b_f_f, 1);
		Arrays.fill(b_f_b, 1);
		Arrays.fill(b_o_f, 0);
		Arrays.fill(b_o_b, 0);
		Arrays.fill(b_z_f, 0);
		Arrays.fill(b_z_b, 0);
		Arrays.fill(b_out, 1);
	}
	public void forward(double train_x[][], double train_y[][], int n){
		forward_f(train_x, train_y, n);
		forward_b(train_x, train_y, n);
		for(int i=0; i<unfold_size ; i++){
			dm.addi(dm.mul_nm_m1(w_out_f, output_y_f[i]), b_out, output_out[i]);
			dm.addi(output_out[i], dm.mul_nm_m1(w_out_b, output_y_b[i]));
			System.arraycopy(output_out[i], 0, output_[i], 0, output_len);
			dm.softmaxi(output_out[i]);
		}
	}
	private void forward_f(double train_x[][], double train_y[][], int n){
		for(int i=0; i<unfold_size ; i++){
			if(i == 0){
				dm.addi(dm.mul_nm_m1(w_ni_f, dm.dot_(dropout_i[0], train_x[i+n])), dm.mul_nm_m1(w_hi_f, dm.dot_(dropout_h[0], hidden_pre_f)), input_i_f[i]);
				dm.addi(input_i_f[i], b_i_f);
				dm.sigmoidi(input_i_f[i]);
				dm.addi(dm.mul_nm_m1(w_nf_f, dm.dot_(dropout_i[1], train_x[i+n])), dm.mul_nm_m1(w_hf_f, dm.dot_(dropout_h[1], hidden_pre_f)), forget_f_f[i]);
				dm.addi(forget_f_f[i], b_f_f);
				dm.sigmoidi(forget_f_f[i]);
				dm.addi(dm.mul_nm_m1(w_no_f, dm.dot_(dropout_i[2], train_x[i+n])), dm.mul_nm_m1(w_ho_f, dm.dot_(dropout_h[2], hidden_pre_f)), output_o_f[i]);
				dm.addi(output_o_f[i], b_o_f);
				dm.sigmoidi(output_o_f[i]);
				dm.addi(dm.mul_nm_m1(w_nz_f, dm.dot_(dropout_i[3], train_x[i+n])), dm.mul_nm_m1(w_hz_f, dm.dot_(dropout_h[3], hidden_pre_f)), block_z_f[i]);
				dm.addi(block_z_f[i], b_z_f);
				dm.tanhi(block_z_f[i]);
				dm.addi(dm.dot_(input_i_f[i], block_z_f[i]), dm.dot_(forget_f_f[i], cell_pre_f), cell_c_f[i]);

			}else{
				dm.addi(dm.mul_nm_m1(w_ni_f, dm.dot_(dropout_i[0], train_x[i+n])), dm.mul_nm_m1(w_hi_f, dm.dot_(dropout_h[0], output_y_f[i-1])), input_i_f[i]);
				dm.addi(input_i_f[i], b_i_f);
				dm.sigmoidi(input_i_f[i]);
				dm.addi(dm.mul_nm_m1(w_nf_f, dm.dot_(dropout_i[1], train_x[i+n])), dm.mul_nm_m1(w_hf_f, dm.dot_(dropout_h[1], output_y_f[i-1])), forget_f_f[i]);
				dm.addi(forget_f_f[i], b_f_f);
				dm.sigmoidi(forget_f_f[i]);
				dm.addi(dm.mul_nm_m1(w_no_f, dm.dot_(dropout_i[2], train_x[i+n])), dm.mul_nm_m1(w_ho_f, dm.dot_(dropout_h[2], output_y_f[i-1])), output_o_f[i]);
				dm.addi(output_o_f[i], b_o_f);
				dm.sigmoidi(output_o_f[i]);
				dm.addi(dm.mul_nm_m1(w_nz_f, dm.dot_(dropout_i[3], train_x[i+n])), dm.mul_nm_m1(w_hz_f, dm.dot_(dropout_h[3], output_y_f[i-1])), block_z_f[i]);
				dm.addi(block_z_f[i], b_z_f);
				dm.tanhi(block_z_f[i]);
				dm.addi(dm.dot_(input_i_f[i], block_z_f[i]), dm.dot_(forget_f_f[i], cell_c_f[i-1]), cell_c_f[i]);
			}
			cell_c_ts_f[i] = dm.tanh_(cell_c_f[i]);
			System.arraycopy(dm.dot_(output_o_f[i], cell_c_ts_f[i]), 0, output_y_f[i], 0, output_y_f[i].length);
			
		}
	}
	private void forward_b(double train_x[][], double train_y[][], int n){
		int j = unfold_size-1;
		for(int i=j; i>=0; i--){
			if(i == j && j != 0){
				dm.addi(dm.mul_nm_m1(w_ni_b, dm.dot_(dropout_i[0], train_x[i+n])), dm.mul_nm_m1(w_hi_b, dm.dot_(dropout_h[0], hidden_pre_b)), input_i_b[i]);
				dm.addi(input_i_b[i], b_i_b);
				dm.sigmoidi(input_i_b[i]);
				dm.addi(dm.mul_nm_m1(w_nf_b, dm.dot_(dropout_i[1], train_x[i+n])), dm.mul_nm_m1(w_hf_b, dm.dot_(dropout_h[1], hidden_pre_b)), forget_f_b[i]);
				dm.addi(forget_f_b[i], b_f_b);
				dm.sigmoidi(forget_f_b[i]);
				dm.addi(dm.mul_nm_m1(w_no_b, dm.dot_(dropout_i[2], train_x[i+n])), dm.mul_nm_m1(w_ho_b, dm.dot_(dropout_h[2], hidden_pre_b)), output_o_b[i]);
				dm.addi(output_o_b[i], b_o_b);
				dm.sigmoidi(output_o_b[i]);
				dm.addi(dm.mul_nm_m1(w_nz_b, dm.dot_(dropout_i[3], train_x[i+n])), dm.mul_nm_m1(w_hz_b, dm.dot_(dropout_h[3], hidden_pre_b)), block_z_b[i]);
				dm.addi(block_z_b[i], b_z_b);
				dm.tanhi(block_z_b[i]);
				dm.addi(dm.dot_(input_i_b[i], block_z_b[i]), dm.dot_(forget_f_b[i], cell_pre_b), cell_c_b[i]);
			}else{
				dm.addi(dm.mul_nm_m1(w_ni_b, dm.dot_(dropout_i[0], train_x[i+n])), dm.mul_nm_m1(w_hi_b, dm.dot_(dropout_h[0], output_y_b[i+1])), input_i_b[i]);
				dm.addi(input_i_b[i], b_i_b);
				dm.sigmoidi(input_i_b[i]);
				dm.addi(dm.mul_nm_m1(w_nf_b, dm.dot_(dropout_i[1], train_x[i+n])), dm.mul_nm_m1(w_hf_b, dm.dot_(dropout_h[1], output_y_b[i+1])), forget_f_b[i]);
				dm.addi(forget_f_b[i], b_f_b);
				dm.sigmoidi(forget_f_b[i]);
				dm.addi(dm.mul_nm_m1(w_no_b, dm.dot_(dropout_i[2], train_x[i+n])), dm.mul_nm_m1(w_ho_b, dm.dot_(dropout_h[2], output_y_b[i+1])), output_o_b[i]);
				dm.addi(output_o_b[i], b_o_b);
				dm.sigmoidi(output_o_b[i]);
				dm.addi(dm.mul_nm_m1(w_nz_b, dm.dot_(dropout_i[3], train_x[i+n])), dm.mul_nm_m1(w_hz_b, dm.dot_(dropout_h[3], output_y_b[i+1])), block_z_b[i]);
				dm.addi(block_z_b[i], b_z_b);
				dm.tanhi(block_z_b[i]);
				dm.addi(dm.dot_(input_i_b[i], block_z_b[i]), dm.dot_(forget_f_b[i], cell_c_b[i+1]), cell_c_b[i]);
			}
			cell_c_ts_b[i] = dm.tanh_(cell_c_b[i]);
			System.arraycopy(dm.dot_(output_o_b[i], cell_c_ts_b[i]), 0, output_y_b[i], 0, output_y_b[i].length);
		}
	}
	public double logsumexp(double x, double y, boolean flg){
		if(flg) return y;
		double vmin = Math.min(x, y);
		double vmax = Math.max(x, y);
		if(vmax > vmin + 13) return vmax;
		else return vmax + Math.log(Math.exp(vmin - vmax) + 1.0);
	}
	public void crf(double train_y[][], int n){
		double[][] alphaSet = new double[unfold_size][];
		double[][] betaSet  = new double[unfold_size][];
		for(int i=0; i<unfold_size; i++){
			alphaSet[i] = new double[output_len];
			for(int j=0; j<output_len; j++){
				double dscore0 = 0;
				if( i > 0) {
					for(int k=0; k<output_len; k++){
						double fbgm = trs[j][k];
						double finit = alphaSet[i-1][k];
						double ftmp = fbgm + finit;
						dscore0 = logsumexp(dscore0, ftmp, k == 0);
					}
				}
				alphaSet[i][j] = dscore0 + output_[i][j];
			}
		}
		for(int i=unfold_size-1; i>=0; i--){
			betaSet[i] = new double[output_len];
			for(int j=0; j<output_len; j++){
				double dscore0 = 0;
				if(i < unfold_size - 1){
					for(int k=0; k<output_len; k++){
						double fbgm = trs[k][j];
						double finit = betaSet[i+1][k];
						double ftmp = fbgm + finit;
						dscore0 = logsumexp(dscore0, ftmp, k == 0);
					}
				}
				betaSet[i][j] = dscore0 + output_[i][j];
			}
		}
		double Z_ = 0.0;
		double[] betaSet_0 = betaSet[0];
		for(int i=0; i<output_len; i++){
			Z_ = logsumexp(Z_, betaSet_0[i], i == 0);
		}
		dm.clear(crf_output);
		for(int i=0; i<unfold_size; i++){
			for(int j=0; j<output_len; j++){
				crf_output[i][j] = Math.exp(alphaSet[i][j] + betaSet[i][j] - output_[i][j] - Z_);
			}
			System.arraycopy(crf_output[i], 0, output_out[i], 0, output_len);
			dm.softmaxi(output_out[i]);
		}
	}
	public void update_bigram(double train_y[][], int n){
		double trs_delta[][] = new double[output_len][output_len];
		for(int timeat = 1; timeat < unfold_size; timeat++){
			double[] crf_output_t = crf_output[timeat];
			double[] crf_output_pt = crf_output[timeat-1];
			for(int i=0; i<output_len; i++){
				double crf_output_ti = crf_output_t[i];
				double[] trs_i = trs[i];
				double[] trs_delta_i = trs_delta[i];
				int j = 0;
				while(j < output_len) {
					trs_delta_i[j] -= trs_i[j] * crf_output_ti * crf_output_pt[j];
					j++;
				}
			}
			int id = dm.max_i(train_y[timeat + n]);
			int last_id = dm.max_i(train_y[timeat + n - 1]);
			trs_delta[id][last_id] += 1;
		}
		dm.clip(trs_delta, -50, 50);
		for(int b = 0; b < output_len; b++){
			double[] vb = trs[b];
			double[] vdeltab = trs_delta[b];
			int a = 0;
			while(a < output_len){
				vb[a] += learning_rate * vdeltab[a];
				a++;
			}
		}
	}

	public void backward(double train_x[][], double train_y[][], int n){
		for(int i=0; i<unfold_size ; i++){
			dm.subi(crf_output[i], train_y[i+n], error_out[i]);
			dm.addi(b_out_, error_out[i]);
		}
		backward_f(train_x, train_y, n);
		backward_b(train_x, train_y, n);
	}
	private void backward_f(double train_x[][], double train_y[][], int n){
		int j =  unfold_size-1;
		for(int i=j; i>=0; i--){
			dm.addi(w_out_f_, dm.mul_n1_1m(error_out[i], output_y_f[i]));
			if(i == j && j != 0){
				error_y_f[i] = dm.mul_nm_m1(dm.T(w_out_f), error_out[i]);
				dm.doti(dm.dot(dm.dev_sigmoid(output_o_f[i]), error_y_f[i]), cell_c_ts_f[i], error_o_f[i]);
				dm.addi(b_o_f_, error_o_f[i]);
				dm.addi(w_no_f_, dm.mul_n1_1m(error_o_f[i], dm.dot_(dropout_i[2], train_x[i+n])));
				dm.doti(dm.dot(dm.dev_tanh(cell_c_ts_f[i]), error_y_f[i]), output_o_f[i], error_c_f[i]);
				dm.doti(dm.dot(dm.dev_sigmoid(forget_f_f[i]), error_c_f[i]), cell_c_f[i-1], error_f_f[i]);
				dm.addi(b_f_f_, error_f_f[i]);
				dm.addi(w_nf_f_, dm.mul_n1_1m(error_f_f[i], dm.dot_(dropout_i[1], train_x[i+n])));
			}else{
				dm.addi(dm.add(dm.mul_nm_m1(dm.T(w_hi_f), error_i_f[i+1]), dm.mul_nm_m1(dm.T(w_hf_f), error_f_f[i+1])), 
						dm.add(dm.mul_nm_m1(dm.T(w_ho_f), error_o_f[i+1]), dm.mul_nm_m1(dm.T(w_hz_f), error_z_f[i+1])),error_y_f[i]);
				dm.addi(error_y_f[i], dm.mul_nm_m1(dm.T(w_out_f), error_out[i]));
				dm.doti(dm.dot(dm.dev_sigmoid(output_o_f[i]), error_y_f[i]), cell_c_ts_f[i], error_o_f[i]);
				dm.addi(b_o_f_, error_o_f[i]);
				dm.addi(w_no_f_, dm.mul_n1_1m(error_o_f[i], dm.dot_(dropout_i[2], train_x[i+n])));
				dm.addi(w_ho_f_, dm.mul_n1_1m(error_o_f[i+1], dm.dot_(dropout_h[2], output_y_f[i])));
				dm.doti(dm.dot(dm.dev_tanh(cell_c_ts_f[i]), error_y_f[i]), output_o_f[i], error_c_f[i]);
				dm.addi(error_c_f[i], dm.dot_(error_c_f[i+1], forget_f_f[i]));
				if(i>0)
					dm.doti(dm.dot(dm.dev_sigmoid(forget_f_f[i]), error_c_f[i]), cell_c_f[i-1], error_f_f[i]);
				dm.addi(b_f_f_, error_f_f[i]);
				dm.addi(w_nf_f_, dm.mul_n1_1m(error_f_f[i], dm.dot_(dropout_i[1], train_x[i+n])));
				dm.addi(w_hf_f_, dm.mul_n1_1m(error_f_f[i+1], dm.dot_(dropout_h[1], output_y_f[i])));
			}
			dm.doti(dm.dot(dm.dev_sigmoid(input_i_f[i]), error_c_f[i]), block_z_f[i], error_i_f[i]);
			dm.addi(b_i_f_, error_i_f[i]);
			dm.addi(w_ni_f_, dm.mul_n1_1m(error_i_f[i], dm.dot_(dropout_i[0], train_x[i+n])));
			if(i<j)
				dm.addi(w_hi_f_, dm.mul_n1_1m(error_i_f[i+1], dm.dot_(dropout_h[0], output_y_f[i])));
			dm.doti(dm.dot(dm.dev_tanh(block_z_f[i]), error_c_f[i]), input_i_f[i], error_z_f[i]);
			dm.addi(b_z_f_, error_z_f[i]);
			dm.addi(w_nz_f_, dm.mul_n1_1m(error_z_f[i], dm.dot_(dropout_i[3], train_x[i+n])));
			if(i<j)
				dm.addi(w_hz_f_, dm.mul_n1_1m(error_z_f[i+1], dm.dot_(dropout_h[3], output_y_f[i])));
		}
	}
	private void backward_b(double train_x[][], double train_y[][], int n){
		int j = unfold_size-1;
		for(int i=0; i<unfold_size; i++){
			if(i == 0){
				error_y_b[i] = dm.mul_nm_m1(dm.T(w_out_b), error_out[i]);
				dm.doti(dm.dot(dm.dev_sigmoid(output_o_b[i]), error_y_b[i]), cell_c_ts_b[i], error_o_b[i]);
				dm.addi(b_o_b_, error_o_b[i]);
				dm.addi(w_no_b_, dm.mul_n1_1m(error_o_b[i], dm.dot_(dropout_i[2], train_x[i+n])));
				dm.doti(dm.dot(dm.dev_tanh(cell_c_ts_b[i]), error_y_b[i]), output_o_b[i], error_c_b[i]);
				dm.doti(dm.dot(dm.dev_sigmoid(forget_f_b[i]), error_c_b[i]), cell_c_b[i+1], error_f_b[i]);
				dm.addi(b_f_b_, error_f_b[i]);
				dm.addi(w_nf_b_, dm.mul_n1_1m(error_f_b[i], dm.dot_(dropout_i[1], train_x[i+n])));
			}else{
				dm.addi(dm.add(dm.mul_nm_m1(dm.T(w_hi_b), error_i_b[i-1]), dm.mul_nm_m1(dm.T(w_hf_b), error_f_b[i-1])), 
						dm.add(dm.mul_nm_m1(dm.T(w_ho_b), error_o_b[i-1]), dm.mul_nm_m1(dm.T(w_hz_b), error_z_b[i-1])),error_y_b[i]);
				dm.addi(error_y_b[i], dm.mul_nm_m1(dm.T(w_out_b), error_out[i]));
				dm.doti(dm.dot(dm.dev_sigmoid(output_o_b[i]), error_y_b[i]), cell_c_ts_b[i], error_o_b[i]);
				dm.addi(b_o_b_, error_o_b[i]);
				dm.addi(w_no_b_, dm.mul_n1_1m(error_o_b[i], dm.dot_(dropout_i[2], train_x[i+n])));
				dm.addi(w_ho_b_, dm.mul_n1_1m(error_o_b[i-1], dm.dot_(dropout_h[2], output_y_b[i])));
				dm.doti(dm.dot(dm.dev_tanh(cell_c_ts_b[i]), error_y_b[i]), output_o_b[i], error_c_b[i]);
				dm.addi(error_c_b[i], dm.dot_(error_c_b[i-1], forget_f_b[i]));
				if(i<j)
					dm.doti(dm.dot(dm.dev_sigmoid(forget_f_b[i]), error_c_b[i]), cell_c_b[i+1], error_f_b[i]);
				dm.addi(b_f_b_, error_f_b[i]);
				dm.addi(w_nf_b_, dm.mul_n1_1m(error_f_b[i], dm.dot_(dropout_i[1], train_x[i+n])));
				dm.addi(w_hf_b_, dm.mul_n1_1m(error_f_b[i-1], dm.dot_(dropout_h[1], output_y_b[i])));
			}
			dm.doti(dm.dot(dm.dev_sigmoid(input_i_b[i]), error_c_b[i]), block_z_b[i], error_i_b[i]);
			dm.addi(b_i_b_, error_i_b[i]);
			dm.addi(w_ni_b_, dm.mul_n1_1m(error_i_b[i], dm.dot_(dropout_i[0], train_x[i+n])));
			if(i>0)
				dm.addi(w_hi_b_, dm.mul_n1_1m(error_i_b[i-1], dm.dot_(dropout_h[0], output_y_b[i])));
			dm.doti(dm.dot(dm.dev_tanh(block_z_b[i]), error_c_b[i]), input_i_b[i], error_z_b[i]);
			dm.addi(b_z_b_, error_z_b[i]);
			dm.addi(w_nz_b_, dm.mul_n1_1m(error_z_b[i], dm.dot_(dropout_i[3], train_x[i+n])));
			if(i>0)
				dm.addi(w_hz_b_, dm.mul_n1_1m(error_z_b[i-1], dm.dot_(dropout_h[3], output_y_b[i])));
		}
	}
	public void update_rmsprop(){
		update_rmsprop_f();
		update_rmsprop_b();
	}
	private void update_rmsprop_f(){
		dm.dot__(w_ni_f_, 1.0/mini_batch);
		dm.dot__(w_hi_f_, 1.0/mini_batch);
		dm.dot__(w_nf_f_, 1.0/mini_batch);
		dm.dot__(w_hf_f_, 1.0/mini_batch);
		dm.dot__(w_no_f_, 1.0/mini_batch);
		dm.dot__(w_ho_f_, 1.0/mini_batch);
		dm.dot__(w_nz_f_, 1.0/mini_batch);
		dm.dot__(w_hz_f_, 1.0/mini_batch);
		dm.dot__(w_out_f_, 1.0/mini_batch);
		dm.dot_(b_i_f_, 1.0/mini_batch);
		dm.dot_(b_f_f_, 1.0/mini_batch);
		dm.dot_(b_o_f_, 1.0/mini_batch);
		dm.dot_(b_z_f_, 1.0/mini_batch);
		dm.dot_(b_out_, 1.0/mini_batch);

		dm.clip(w_ni_f_, -15, 15);
		dm.clip(w_hi_f_, -15, 15);
		dm.clip(w_nf_f_, -15, 15);
		dm.clip(w_hf_f_, -15, 15);
		dm.clip(w_no_f_, -15, 15);
		dm.clip(w_ho_f_, -15, 15);
		dm.clip(w_nz_f_, -15, 15);
		dm.clip(w_hz_f_, -15, 15);
		dm.clip(w_out_f_, -15, 15);
		dm.clip(b_i_f_, -15, 15);
		dm.clip(b_f_f_, -15, 15);
		dm.clip(b_o_f_, -15, 15);
		dm.clip(b_z_f_, -15, 15);
		dm.clip(b_out_, -15, 15);

		dm.dot_(cache_b_out_m, decay_rate);
		dm.addi(cache_b_out_m, dm.dot_o(dm.pow(b_out_), 1-decay_rate));
		dm.subi(b_out, dm.dot_o(dm.dot(dm.dev_sqrt_(dm.add_A_(cache_b_out_m, eps)), b_out_), learning_rate));
		dm.dot_(cache_b_i_m_f, decay_rate);
		dm.addi(cache_b_i_m_f, dm.dot_o(dm.pow(b_i_f_), 1-decay_rate));
		dm.subi(b_i_f, dm.dot_o(dm.dot(dm.dev_sqrt_(dm.add_A_(cache_b_i_m_f, eps)), b_i_f_), learning_rate));
		dm.dot_(cache_b_f_m_f, decay_rate);
		dm.addi(cache_b_f_m_f, dm.dot_o(dm.pow(b_f_f_), 1-decay_rate));
		dm.subi(b_f_f, dm.dot_o(dm.dot(dm.dev_sqrt_(dm.add_A_(cache_b_f_m_f, eps)), b_f_f_), learning_rate));
		dm.dot_(cache_b_o_m_f, decay_rate);
		dm.addi(cache_b_o_m_f, dm.dot_o(dm.pow(b_o_f_), 1-decay_rate));
		dm.subi(b_o_f, dm.dot_o(dm.dot(dm.dev_sqrt_(dm.add_A_(cache_b_o_m_f, eps)), b_o_f_), learning_rate));
		dm.dot_(cache_b_z_m_f, decay_rate);
		dm.addi(cache_b_z_m_f, dm.dot_o(dm.pow(b_z_f_), 1-decay_rate));
		dm.subi(b_z_f, dm.dot_o(dm.dot(dm.dev_sqrt_(dm.add_A_(cache_b_z_m_f, eps)), b_z_f_), learning_rate));
		
		dm.dot__(cache_w_out_m_f, decay_rate);
		dm.addi(cache_w_out_m_f, dm.dot__o(dm.pow(w_out_f_), 1-decay_rate));
		dm.subi(w_out_f, dm.dot__o(dm.dot__(dm.dev_sqrt_(dm.add_A_(cache_w_out_m_f, eps)), w_out_f_), learning_rate));
		dm.dot__(cache_w_ni_m_f, decay_rate);
		dm.addi(cache_w_ni_m_f, dm.dot__o(dm.pow(w_ni_f_), 1-decay_rate));
		dm.subi(w_ni_f, dm.dot__o(dm.dot__(dm.dev_sqrt_(dm.add_A_(cache_w_ni_m_f, eps)), w_ni_f_), learning_rate));
		dm.dot__(cache_w_hi_m_f, decay_rate);
		dm.addi(cache_w_hi_m_f, dm.dot__o(dm.pow(w_hi_f_), 1-decay_rate));
		dm.subi(w_hi_f, dm.dot__o(dm.dot__(dm.dev_sqrt_(dm.add_A_(cache_w_hi_m_f, eps)), w_hi_f_), learning_rate));
		dm.dot__(cache_w_nf_m_f, decay_rate);
		dm.addi(cache_w_nf_m_f, dm.dot__o(dm.pow(w_nf_f_), 1-decay_rate));
		dm.subi(w_nf_f, dm.dot__o(dm.dot__(dm.dev_sqrt_(dm.add_A_(cache_w_nf_m_f, eps)), w_nf_f_), learning_rate));
		dm.dot__(cache_w_hf_m_f, decay_rate);
		dm.addi(cache_w_hf_m_f, dm.dot__o(dm.pow(w_hf_f_), 1-decay_rate));
		dm.subi(w_hf_f, dm.dot__o(dm.dot__(dm.dev_sqrt_(dm.add_A_(cache_w_hf_m_f, eps)), w_hf_f_), learning_rate));
		dm.dot__(cache_w_no_m_f, decay_rate);
		dm.addi(cache_w_no_m_f, dm.dot__o(dm.pow(w_no_f_), 1-decay_rate));
		dm.subi(w_no_f, dm.dot__o(dm.dot__(dm.dev_sqrt_(dm.add_A_(cache_w_no_m_f, eps)), w_no_f_), learning_rate));
		dm.dot__(cache_w_ho_m_f, decay_rate);
		dm.addi(cache_w_ho_m_f, dm.dot__o(dm.pow(w_ho_f_), 1-decay_rate));
		dm.subi(w_ho_f, dm.dot__o(dm.dot__(dm.dev_sqrt_(dm.add_A_(cache_w_ho_m_f, eps)), w_ho_f_), learning_rate));
		
		dm.clear(w_ni_f_);
		dm.clear(w_hi_f_);
		dm.clear(w_nf_f_);
		dm.clear(w_hf_f_);
		dm.clear(w_no_f_);
		dm.clear(w_ho_f_);
		dm.clear(w_nz_f_);
		dm.clear(w_hz_f_);
		dm.clear(w_out_f_);
		Arrays.fill(b_i_f_, 0);
		Arrays.fill(b_f_f_, 0);
		Arrays.fill(b_o_f_, 0);
		Arrays.fill(b_z_f_, 0);
		Arrays.fill(b_out_, 0);
	}
	private void update_rmsprop_b(){
		dm.dot__(w_ni_b_, 1.0/mini_batch);
		dm.dot__(w_hi_b_, 1.0/mini_batch);
		dm.dot__(w_nf_b_, 1.0/mini_batch);
		dm.dot__(w_hf_b_, 1.0/mini_batch);
		dm.dot__(w_no_b_, 1.0/mini_batch);
		dm.dot__(w_ho_b_, 1.0/mini_batch);
		dm.dot__(w_nz_b_, 1.0/mini_batch);
		dm.dot__(w_hz_b_, 1.0/mini_batch);
		dm.dot__(w_out_b_, 1.0/mini_batch);
		dm.dot_(b_i_b_, 1.0/mini_batch);
		dm.dot_(b_f_b_, 1.0/mini_batch);
		dm.dot_(b_o_b_, 1.0/mini_batch);
		dm.dot_(b_z_b_, 1.0/mini_batch);

		dm.clip(w_ni_b_, -15, 15);
		dm.clip(w_hi_b_, -15, 15);
		dm.clip(w_nf_b_, -15, 15);
		dm.clip(w_hf_b_, -15, 15);
		dm.clip(w_no_b_, -15, 15);
		dm.clip(w_ho_b_, -15, 15);
		dm.clip(w_nz_b_, -15, 15);
		dm.clip(w_hz_b_, -15, 15);
		dm.clip(w_out_b_, -15, 15);
		dm.clip(b_i_b_, -15, 15);
		dm.clip(b_f_b_, -15, 15);
		dm.clip(b_o_b_, -15, 15);
		dm.clip(b_z_b_, -15, 15);

		dm.dot_(cache_b_i_m_b, decay_rate);
		dm.addi(cache_b_i_m_b, dm.dot_o(dm.pow(b_i_b_), 1-decay_rate));
		dm.subi(b_i_b, dm.dot_o(dm.dot(dm.dev_sqrt_(dm.add_A_(cache_b_i_m_b, eps)), b_i_b_), learning_rate));
		dm.dot_(cache_b_f_m_b, decay_rate);
		dm.addi(cache_b_f_m_b, dm.dot_o(dm.pow(b_f_b_), 1-decay_rate));
		dm.subi(b_f_b, dm.dot_o(dm.dot(dm.dev_sqrt_(dm.add_A_(cache_b_f_m_b, eps)), b_f_b_), learning_rate));
		dm.dot_(cache_b_o_m_b, decay_rate);
		dm.addi(cache_b_o_m_b, dm.dot_o(dm.pow(b_o_b_), 1-decay_rate));
		dm.subi(b_o_b, dm.dot_o(dm.dot(dm.dev_sqrt_(dm.add_A_(cache_b_o_m_b, eps)), b_o_b_), learning_rate));
		dm.dot_(cache_b_z_m_b, decay_rate);
		dm.addi(cache_b_z_m_b, dm.dot_o(dm.pow(b_z_b_), 1-decay_rate));
		dm.subi(b_z_b, dm.dot_o(dm.dot(dm.dev_sqrt_(dm.add_A_(cache_b_z_m_b, eps)), b_z_b_), learning_rate));
		
		dm.dot__(cache_w_out_m_b, decay_rate);
		dm.addi(cache_w_out_m_b, dm.dot__o(dm.pow(w_out_b_), 1-decay_rate));
		dm.subi(w_out_b, dm.dot__o(dm.dot__(dm.dev_sqrt_(dm.add_A_(cache_w_out_m_b, eps)), w_out_b_), learning_rate));
		dm.dot__(cache_w_ni_m_b, decay_rate);
		dm.addi(cache_w_ni_m_b, dm.dot__o(dm.pow(w_ni_b_), 1-decay_rate));
		dm.subi(w_ni_b, dm.dot__o(dm.dot__(dm.dev_sqrt_(dm.add_A_(cache_w_ni_m_b, eps)), w_ni_b_), learning_rate));
		dm.dot__(cache_w_hi_m_b, decay_rate);
		dm.addi(cache_w_hi_m_b, dm.dot__o(dm.pow(w_hi_b_), 1-decay_rate));
		dm.subi(w_hi_b, dm.dot__o(dm.dot__(dm.dev_sqrt_(dm.add_A_(cache_w_hi_m_b, eps)), w_hi_b_), learning_rate));
		dm.dot__(cache_w_nf_m_b, decay_rate);
		dm.addi(cache_w_nf_m_b, dm.dot__o(dm.pow(w_nf_b_), 1-decay_rate));
		dm.subi(w_nf_b, dm.dot__o(dm.dot__(dm.dev_sqrt_(dm.add_A_(cache_w_nf_m_b, eps)), w_nf_b_), learning_rate));
		dm.dot__(cache_w_hf_m_b, decay_rate);
		dm.addi(cache_w_hf_m_b, dm.dot__o(dm.pow(w_hf_b_), 1-decay_rate));
		dm.subi(w_hf_b, dm.dot__o(dm.dot__(dm.dev_sqrt_(dm.add_A_(cache_w_hf_m_b, eps)), w_hf_b_), learning_rate));
		dm.dot__(cache_w_no_m_b, decay_rate);
		dm.addi(cache_w_no_m_b, dm.dot__o(dm.pow(w_no_b_), 1-decay_rate));
		dm.subi(w_no_b, dm.dot__o(dm.dot__(dm.dev_sqrt_(dm.add_A_(cache_w_no_m_b, eps)), w_no_b_), learning_rate));
		dm.dot__(cache_w_ho_m_b, decay_rate);
		dm.addi(cache_w_ho_m_b, dm.dot__o(dm.pow(w_ho_b_), 1-decay_rate));
		dm.subi(w_ho_b, dm.dot__o(dm.dot__(dm.dev_sqrt_(dm.add_A_(cache_w_ho_m_b, eps)), w_ho_b_), learning_rate));
		
		dm.clear(w_ni_b_);
		dm.clear(w_hi_b_);
		dm.clear(w_nf_b_);
		dm.clear(w_hf_b_);
		dm.clear(w_no_b_);
		dm.clear(w_ho_b_);
		dm.clear(w_nz_b_);
		dm.clear(w_hz_b_);
		dm.clear(w_out_b_);
		Arrays.fill(b_i_b_, 0);
		Arrays.fill(b_f_b_, 0);
		Arrays.fill(b_o_b_, 0);
		Arrays.fill(b_z_b_, 0);
		//Arrays.fill(b_out_, 0);
	}
	public void update_adam(){
		update_adam_f();
		update_adam_b();
	}
	private void update_adam_f(){
		dm.dot__(w_ni_f_, 1.0/mini_batch);
		dm.dot__(w_hi_f_, 1.0/mini_batch);
		dm.dot__(w_nf_f_, 1.0/mini_batch);
		dm.dot__(w_hf_f_, 1.0/mini_batch);
		dm.dot__(w_no_f_, 1.0/mini_batch);
		dm.dot__(w_ho_f_, 1.0/mini_batch);
		dm.dot__(w_nz_f_, 1.0/mini_batch);
		dm.dot__(w_hz_f_, 1.0/mini_batch);
		dm.dot__(w_out_f_, 1.0/mini_batch);
		dm.dot_(b_i_f_, 1.0/mini_batch);
		dm.dot_(b_f_f_, 1.0/mini_batch);
		dm.dot_(b_o_f_, 1.0/mini_batch);
		dm.dot_(b_z_f_, 1.0/mini_batch);
		dm.dot_(b_out_, 1.0/mini_batch);
		
		dm.clip(w_ni_f_, -5, 5);
		dm.clip(w_hi_f_, -5, 5);
		dm.clip(w_nf_f_, -5, 5);
		dm.clip(w_hf_f_, -5, 5);
		dm.clip(w_no_f_, -5, 5);
		dm.clip(w_ho_f_, -5, 5);
		dm.clip(w_nz_f_, -5, 5);
		dm.clip(w_hz_f_, -5, 5);
		dm.clip(w_out_f_, -5, 5);
		dm.clip(b_i_f_, -5, 5);
		dm.clip(b_f_f_, -5, 5);
		dm.clip(b_o_f_, -5, 5);
		dm.clip(b_z_f_, -5, 5);
		dm.clip(b_out_, -5, 5);		
		
		dm.addi(dm.dot_o_(b_out_, 1-beta_adam1), dm.dot_o(cache_b_out_m, beta_adam1), cache_b_out_m);
		dm.addi(dm.dot_o(dm.dot_(b_out_, b_out_), 1-beta_adam2), dm.dot_o(cache_b_out_v, beta_adam2), cache_b_out_v);		
		dm.subi(b_out, dm.dot_o(dm.dot(dm.div_m(dm.add_A(dm.sqrt(cache_b_out_v), eps)), cache_b_out_m), 
				learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.addi(dm.dot_o_(b_i_f_, 1-beta_adam1), dm.dot_o(cache_b_i_m_f, beta_adam1), cache_b_i_m_f);
		dm.addi(dm.dot_o(dm.dot_(b_i_f_, b_i_f_), 1-beta_adam2), dm.dot_o(cache_b_i_v_f, beta_adam2), cache_b_i_v_f);		
		dm.subi(b_i_f, dm.dot_o(dm.dot(dm.div_m(dm.add_A(dm.sqrt(cache_b_i_v_f), eps)), cache_b_i_m_f), 
				learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.addi(dm.dot_o_(b_f_f_, 1-beta_adam1), dm.dot_o(cache_b_f_m_f, beta_adam1), cache_b_f_m_f);
		dm.addi(dm.dot_o(dm.dot_(b_f_f_, b_f_f_), 1-beta_adam2), dm.dot_o(cache_b_f_v_f, beta_adam2), cache_b_f_v_f);		
		dm.subi(b_f_f, dm.dot_o(dm.dot(dm.div_m(dm.add_A(dm.sqrt(cache_b_f_v_f), eps)), cache_b_f_m_f), 
				learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.addi(dm.dot_o_(b_o_f_, 1-beta_adam1), dm.dot_o(cache_b_o_m_f, beta_adam1), cache_b_o_m_f);
		dm.addi(dm.dot_o(dm.dot_(b_o_f_, b_o_f_), 1-beta_adam2), dm.dot_o(cache_b_o_v_f, beta_adam2), cache_b_o_v_f);		
		dm.subi(b_o_f, dm.dot_o(dm.dot(dm.div_m(dm.add_A(dm.sqrt(cache_b_o_v_f), eps)), cache_b_o_m_f), 
				learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.addi(dm.dot_o_(b_z_f_, 1-beta_adam1), dm.dot_o(cache_b_z_m_f, beta_adam1), cache_b_z_m_f);
		dm.addi(dm.dot_o(dm.dot_(b_z_f_, b_z_f_), 1-beta_adam2), dm.dot_o(cache_b_z_v_f, beta_adam2), cache_b_z_v_f);		
		dm.subi(b_z_f, dm.dot_o(dm.dot(dm.div_m(dm.add_A(dm.sqrt(cache_b_z_v_f), eps)), cache_b_z_m_f), 
				learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		
		dm.add_(dm.dot__o_(w_out_f_, 1-beta_adam1), dm.dot__o(cache_w_out_m_f, beta_adam1), cache_w_out_m_f);
		dm.add_(dm.dot__o(dm.dot__(w_out_f_, w_out_f_), 1-beta_adam2), dm.dot__o(cache_w_out_v_f, beta_adam2), cache_w_out_v_f);
		dm.subi(w_out_f, dm.dot__o(dm.dot__(dm.div_m(dm.add_A(dm.sqrt(cache_w_out_v_f), eps)), cache_w_out_m_f), 
				learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.add_(dm.dot__o_(w_ni_f_, 1-beta_adam1), dm.dot__o(cache_w_ni_m_f, beta_adam1), cache_w_ni_m_f);
		dm.add_(dm.dot__o(dm.dot__(w_ni_f_, w_ni_f_), 1-beta_adam2), dm.dot__o(cache_w_ni_v_f, beta_adam2), cache_w_ni_v_f);
		dm.subi(w_ni_f, dm.dot__o(dm.dot__(dm.div_m(dm.add_A(dm.sqrt(cache_w_ni_v_f), eps)), cache_w_ni_m_f), 
				learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.add_(dm.dot__o_(w_hi_f_, 1-beta_adam1), dm.dot__o(cache_w_hi_m_f, beta_adam1), cache_w_hi_m_f);
		dm.add_(dm.dot__o(dm.dot__(w_hi_f_, w_hi_f_), 1-beta_adam2), dm.dot__o(cache_w_hi_v_f, beta_adam2), cache_w_hi_v_f);
		dm.subi(w_hi_f, dm.dot__o(dm.dot__(dm.div_m(dm.add_A(dm.sqrt(cache_w_hi_v_f), eps)), cache_w_hi_m_f), 
				learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.add_(dm.dot__o_(w_nf_f_, 1-beta_adam1), dm.dot__o(cache_w_nf_m_f, beta_adam1), cache_w_nf_m_f);
		dm.add_(dm.dot__o(dm.dot__(w_nf_f_, w_nf_f_), 1-beta_adam2), dm.dot__o(cache_w_nf_v_f, beta_adam2), cache_w_nf_v_f);
		dm.subi(w_nf_f, dm.dot__o(dm.dot__(dm.div_m(dm.add_A(dm.sqrt(cache_w_nf_v_f), eps)), cache_w_nf_m_f), 
				learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.add_(dm.dot__o_(w_hf_f_, 1-beta_adam1), dm.dot__o(cache_w_hf_m_f, beta_adam1), cache_w_hf_m_f);
		dm.add_(dm.dot__o(dm.dot__(w_hf_f_, w_hf_f_), 1-beta_adam2), dm.dot__o(cache_w_hf_v_f, beta_adam2), cache_w_hf_v_f);
		dm.subi(w_hf_f, dm.dot__o(dm.dot__(dm.div_m(dm.add_A(dm.sqrt(cache_w_hf_v_f), eps)), cache_w_hf_m_f), 
				learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.add_(dm.dot__o_(w_no_f_, 1-beta_adam1), dm.dot__o(cache_w_no_m_f, beta_adam1), cache_w_no_m_f);
		dm.add_(dm.dot__o(dm.dot__(w_no_f_, w_no_f_), 1-beta_adam2), dm.dot__o(cache_w_no_v_f, beta_adam2), cache_w_no_v_f);
		dm.subi(w_no_f, dm.dot__o(dm.dot__(dm.div_m(dm.add_A(dm.sqrt(cache_w_no_v_f), eps)), cache_w_no_m_f), 
				learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.add_(dm.dot__o_(w_ho_f_, 1-beta_adam1), dm.dot__o(cache_w_ho_m_f, beta_adam1), cache_w_ho_m_f);
		dm.add_(dm.dot__o(dm.dot__(w_ho_f_, w_ho_f_), 1-beta_adam2), dm.dot__o(cache_w_ho_v_f, beta_adam2), cache_w_ho_v_f);
		dm.subi(w_ho_f, dm.dot__o(dm.dot__(dm.div_m(dm.add_A(dm.sqrt(cache_w_ho_v_f), eps)), cache_w_ho_m_f), 
				learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.add_(dm.dot__o_(w_nz_f_, 1-beta_adam1), dm.dot__o(cache_w_nz_m_f, beta_adam1), cache_w_nz_m_f);
		dm.add_(dm.dot__o(dm.dot__(w_nz_f_, w_nz_f_), 1-beta_adam2), dm.dot__o(cache_w_nz_v_f, beta_adam2), cache_w_nz_v_f);
		dm.subi(w_nz_f, dm.dot__o(dm.dot__(dm.div_m(dm.add_A(dm.sqrt(cache_w_nz_v_f), eps)), cache_w_nz_m_f), 
				learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.add_(dm.dot__o_(w_hz_f_, 1-beta_adam1), dm.dot__o(cache_w_hz_m_f, beta_adam1), cache_w_hz_m_f);
		dm.add_(dm.dot__o(dm.dot__(w_hz_f_, w_hz_f_), 1-beta_adam2), dm.dot__o(cache_w_hz_v_f, beta_adam2), cache_w_hz_v_f);
		dm.subi(w_hz_f, dm.dot__o(dm.dot__(dm.div_m(dm.add_A(dm.sqrt(cache_w_hz_v_f), eps)), cache_w_hz_m_f), 
				learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		
		dm.clear(w_ni_f_);
		dm.clear(w_hi_f_);
		dm.clear(w_nf_f_);
		dm.clear(w_hf_f_);
		dm.clear(w_no_f_);
		dm.clear(w_ho_f_);
		dm.clear(w_nz_f_);
		dm.clear(w_hz_f_);
		dm.clear(w_out_f_);
		Arrays.fill(b_i_f_, 0);
		Arrays.fill(b_f_f_, 0);
		Arrays.fill(b_o_f_, 0);
		Arrays.fill(b_z_f_, 0);
		Arrays.fill(b_out_, 0);
	}
	private void update_adam_b(){
		dm.dot__(w_ni_b_, 1.0/mini_batch);
		dm.dot__(w_hi_b_, 1.0/mini_batch);
		dm.dot__(w_nf_b_, 1.0/mini_batch);
		dm.dot__(w_hf_b_, 1.0/mini_batch);
		dm.dot__(w_no_b_, 1.0/mini_batch);
		dm.dot__(w_ho_b_, 1.0/mini_batch);
		dm.dot__(w_nz_b_, 1.0/mini_batch);
		dm.dot__(w_hz_b_, 1.0/mini_batch);
		dm.dot__(w_out_b_, 1.0/mini_batch);
		dm.dot_(b_i_b_, 1.0/mini_batch);
		dm.dot_(b_f_b_, 1.0/mini_batch);
		dm.dot_(b_o_b_, 1.0/mini_batch);
		dm.dot_(b_z_b_, 1.0/mini_batch);

		dm.clip(w_ni_b_, -5, 5);
		dm.clip(w_hi_b_, -5, 5);
		dm.clip(w_nf_b_, -5, 5);
		dm.clip(w_hf_b_, -5, 5);
		dm.clip(w_no_b_, -5, 5);
		dm.clip(w_ho_b_, -5, 5);
		dm.clip(w_nz_b_, -5, 5);
		dm.clip(w_hz_b_, -5, 5);
		dm.clip(w_out_b_, -5, 5);
		dm.clip(b_i_b_, -5, 5);
		dm.clip(b_f_b_, -5, 5);
		dm.clip(b_o_b_, -5, 5);
		dm.clip(b_z_b_, -5, 5);

		dm.addi(dm.dot_o_(b_i_b_, 1-beta_adam1), dm.dot_o(cache_b_i_m_b, beta_adam1), cache_b_i_m_b);
		dm.addi(dm.dot_o(dm.dot_(b_i_b_, b_i_b_), 1-beta_adam2), dm.dot_o(cache_b_i_v_b, beta_adam2), cache_b_i_v_b);		
		dm.subi(b_i_b, dm.dot_o(dm.dot(dm.div_m(dm.add_A(dm.sqrt(cache_b_i_v_b), eps)), cache_b_i_m_b), 
				learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.addi(dm.dot_o_(b_f_b_, 1-beta_adam1), dm.dot_o(cache_b_f_m_b, beta_adam1), cache_b_f_m_b);
		dm.addi(dm.dot_o(dm.dot_(b_f_b_, b_f_b_), 1-beta_adam2), dm.dot_o(cache_b_f_v_b, beta_adam2), cache_b_f_v_b);		
		dm.subi(b_f_b, dm.dot_o(dm.dot(dm.div_m(dm.add_A(dm.sqrt(cache_b_f_v_b), eps)), cache_b_f_m_b), 
				learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.addi(dm.dot_o_(b_o_b_, 1-beta_adam1), dm.dot_o(cache_b_o_m_b, beta_adam1), cache_b_o_m_b);
		dm.addi(dm.dot_o(dm.dot_(b_o_b_, b_o_b_), 1-beta_adam2), dm.dot_o(cache_b_o_v_b, beta_adam2), cache_b_o_v_b);		
		dm.subi(b_o_b, dm.dot_o(dm.dot(dm.div_m(dm.add_A(dm.sqrt(cache_b_o_v_b), eps)), cache_b_o_m_b), 
				learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.addi(dm.dot_o_(b_z_b_, 1-beta_adam1), dm.dot_o(cache_b_z_m_b, beta_adam1), cache_b_z_m_b);
		dm.addi(dm.dot_o(dm.dot_(b_z_b_, b_z_b_), 1-beta_adam2), dm.dot_o(cache_b_z_v_b, beta_adam2), cache_b_z_v_b);		
		dm.subi(b_z_b, dm.dot_o(dm.dot(dm.div_m(dm.add_A(dm.sqrt(cache_b_z_v_b), eps)), cache_b_z_m_b), 
				learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		
		dm.add_(dm.dot__o_(w_out_b_, 1-beta_adam1), dm.dot__o(cache_w_out_m_b, beta_adam1), cache_w_out_m_b);
		dm.add_(dm.dot__o(dm.dot__(w_out_b_, w_out_b_), 1-beta_adam2), dm.dot__o(cache_w_out_v_b, beta_adam2), cache_w_out_v_b);
		dm.subi(w_out_b, dm.dot__o(dm.dot__(dm.div_m(dm.add_A(dm.sqrt(cache_w_out_v_b), eps)), cache_w_out_m_b), 
				learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.add_(dm.dot__o_(w_ni_b_, 1-beta_adam1), dm.dot__o(cache_w_ni_m_b, beta_adam1), cache_w_ni_m_b);
		dm.add_(dm.dot__o(dm.dot__(w_ni_b_, w_ni_b_), 1-beta_adam2), dm.dot__o(cache_w_ni_v_b, beta_adam2), cache_w_ni_v_b);
		dm.subi(w_ni_b, dm.dot__o(dm.dot__(dm.div_m(dm.add_A(dm.sqrt(cache_w_ni_v_b), eps)), cache_w_ni_m_b), 
				learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.add_(dm.dot__o_(w_hi_b_, 1-beta_adam1), dm.dot__o(cache_w_hi_m_b, beta_adam1), cache_w_hi_m_b);
		dm.add_(dm.dot__o(dm.dot__(w_hi_b_, w_hi_b_), 1-beta_adam2), dm.dot__o(cache_w_hi_v_b, beta_adam2), cache_w_hi_v_b);
		dm.subi(w_hi_b, dm.dot__o(dm.dot__(dm.div_m(dm.add_A(dm.sqrt(cache_w_hi_v_b), eps)), cache_w_hi_m_b), 
				learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.add_(dm.dot__o_(w_nf_b_, 1-beta_adam1), dm.dot__o(cache_w_nf_m_b, beta_adam1), cache_w_nf_m_b);
		dm.add_(dm.dot__o(dm.dot__(w_nf_b_, w_nf_b_), 1-beta_adam2), dm.dot__o(cache_w_nf_v_b, beta_adam2), cache_w_nf_v_b);
		dm.subi(w_nf_b, dm.dot__o(dm.dot__(dm.div_m(dm.add_A(dm.sqrt(cache_w_nf_v_b), eps)), cache_w_nf_m_b), 
				learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.add_(dm.dot__o_(w_hf_b_, 1-beta_adam1), dm.dot__o(cache_w_hf_m_b, beta_adam1), cache_w_hf_m_b);
		dm.add_(dm.dot__o(dm.dot__(w_hf_b_, w_hf_b_), 1-beta_adam2), dm.dot__o(cache_w_hf_v_b, beta_adam2), cache_w_hf_v_b);
		dm.subi(w_hf_b, dm.dot__o(dm.dot__(dm.div_m(dm.add_A(dm.sqrt(cache_w_hf_v_b), eps)), cache_w_hf_m_b), 
				learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.add_(dm.dot__o_(w_no_b_, 1-beta_adam1), dm.dot__o(cache_w_no_m_b, beta_adam1), cache_w_no_m_b);
		dm.add_(dm.dot__o(dm.dot__(w_no_b_, w_no_b_), 1-beta_adam2), dm.dot__o(cache_w_no_v_b, beta_adam2), cache_w_no_v_b);
		dm.subi(w_no_b, dm.dot__o(dm.dot__(dm.div_m(dm.add_A(dm.sqrt(cache_w_no_v_b), eps)), cache_w_no_m_b), 
				learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.add_(dm.dot__o_(w_ho_b_, 1-beta_adam1), dm.dot__o(cache_w_ho_m_b, beta_adam1), cache_w_ho_m_b);
		dm.add_(dm.dot__o(dm.dot__(w_ho_b_, w_ho_b_), 1-beta_adam2), dm.dot__o(cache_w_ho_v_b, beta_adam2), cache_w_ho_v_b);
		dm.subi(w_ho_b, dm.dot__o(dm.dot__(dm.div_m(dm.add_A(dm.sqrt(cache_w_ho_v_b), eps)), cache_w_ho_m_b), 
				learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.add_(dm.dot__o_(w_nz_b_, 1-beta_adam1), dm.dot__o(cache_w_nz_m_b, beta_adam1), cache_w_nz_m_b);
		dm.add_(dm.dot__o(dm.dot__(w_nz_b_, w_nz_b_), 1-beta_adam2), dm.dot__o(cache_w_nz_v_b, beta_adam2), cache_w_nz_v_b);
		dm.subi(w_nz_b, dm.dot__o(dm.dot__(dm.div_m(dm.add_A(dm.sqrt(cache_w_nz_v_b), eps)), cache_w_nz_m_b), 
				learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.add_(dm.dot__o_(w_hz_b_, 1-beta_adam1), dm.dot__o(cache_w_hz_m_b, beta_adam1), cache_w_hz_m_b);
		dm.add_(dm.dot__o(dm.dot__(w_hz_b_, w_hz_b_), 1-beta_adam2), dm.dot__o(cache_w_hz_v_b, beta_adam2), cache_w_hz_v_b);
		dm.subi(w_hz_b, dm.dot__o(dm.dot__(dm.div_m(dm.add_A(dm.sqrt(cache_w_hz_v_b), eps)), cache_w_hz_m_b), 
				learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		
		dm.clear(w_ni_b_);
		dm.clear(w_hi_b_);
		dm.clear(w_nf_b_);
		dm.clear(w_hf_b_);
		dm.clear(w_no_b_);
		dm.clear(w_ho_b_);
		dm.clear(w_nz_b_);
		dm.clear(w_hz_b_);
		dm.clear(w_out_b_);
		Arrays.fill(b_i_b_, 0);
		Arrays.fill(b_f_b_, 0);
		Arrays.fill(b_o_b_, 0);
		Arrays.fill(b_z_b_, 0);
		//Arrays.fill(b_out_, 0);
	}
	public void train(double train_x[][], double train_y[][], double trs[][], int A[]){
		this.trs = trs;
		N = train_x.length;
		int k=epochs;
		while(epochs-- >= 0){
			System.out.println("epochs: "+(epochs+1));
			int a = 0;
			unfold_size = A[a];
			for(int i=0; i<N;){
				dropout_i = dm.dropout(4, input_len, drop);
				dropout_h = dm.dropout(4, hidden_len, drop);
				for(int j=0; j<mini_batch; j++){
					forward(train_x, train_y, i);
					crf(train_y, i);
					update_bigram(train_y, i);
					backward(train_x, train_y, i);
					i += A[a];
					if(a >= A.length - 1) break;
					unfold_size = A[++a];
				}
				if(!isadam)
					update_rmsprop();
				else
					update_adam();
			}
			if ((k-epochs) % 20 == 0)
				learning_rate /= 10;
		}
	}
	public void predict(double test_x[][], double test_y[][], double label_[][], int label[], int B[]){
		all = 0;
		int right = 0;
		N = test_x.length;
		int a = 0;
		unfold_size = B[a];
		dm.fill(dropout_i, 1.0);
		dm.fill(dropout_h, 1.0);
		for(int i=0; i<N; ){
			for(int j=0; j<mini_batch; j++){
				forward(test_x, test_y, i);
				crf(test_y, i);
				right += fit(test_y, i, label, label_);
				i += B[a];
				if(a >= B.length - 1) break;
				unfold_size = B[++a];
			}
		}
		System.out.println("accuracy num: "+right+" all num: "+test_x.length);
		System.out.println("accuracy rate: "+1.0*right/test_x.length);
		System.out.println("viterbi: "+all);
	}double all = 0;
	public int fit(double test_y[][], int j, int label[], double label_[][]){
		int a,b,c=0;
		int A[] = viterbi(output_);
		System.arraycopy(A, 0, label, j, unfold_size);
		for(int i=0; i<unfold_size && i+j<N; i++){
			a = dm.max_i(test_y[i+j]);
			b = dm.max_i(output_out[i]);
			if(a == b) c++;
			if(a == A[i]) all ++;
			label_[i+j] = output_out[i].clone();
		}
		return c;
	}
	public int[] viterbi(double A[][]){
		int vpath[][] = new int[unfold_size][output_len];
		double vpalpha[] = new double[output_len];
		double valpha[]  = new double[output_len];
		int start = 0;
		for(int i=0; i<output_len; i++){
			vpalpha[i] = A[0][i];
			if( i != start)
				vpalpha[i] += Double.MIN_VALUE;
			vpath[0][i] = start;
		}
		for(int t=0; t<unfold_size; t++){
			for(int j=0; j<output_len; j++){
				vpath[t][j] = 0;
				double trs_j[] = trs[j];
				double A_t[] = A[t];
				double maxScore = Double.MIN_VALUE;
				for(int i=0; i<output_len; i++){
					double score = vpalpha[i] + trs_j[i] + A_t[j];
					if(score > maxScore){
						maxScore = score;
						vpath[t][j] = i;
					}
				}
				valpha[j] = maxScore;
			}
			vpalpha = valpha.clone();
			valpha = new double[output_len];
		}
		int tag[] = new int[unfold_size];
		tag[unfold_size - 1] = dm.max_i(vpalpha);
		int next = tag[unfold_size - 1];
		for(int t = unfold_size - 2; t>=0; t--){
			tag[t] = vpath[t+1][next];
			next = tag[t];
		}
		return tag;
	}

}
