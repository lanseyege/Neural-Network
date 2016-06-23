
import java.util.Arrays;
import java.util.Random;


public class LSTMFould {

	double w_ni[][] = null;//now input
	double w_hi[][] = null;//hidden input (prev)
	double w_nf[][] = null;
	double w_hf[][] = null;
	double w_nz[][] = null;
	double w_hz[][] = null;
	double w_no[][] = null;
	double w_ho[][] = null;
	double w_out[][] = null;
	
	double w_ni_[][] = null;
	double w_hi_[][] = null;
	double w_nf_[][] = null;
	double w_hf_[][] = null;
	double w_nz_[][] = null;
	double w_hz_[][] = null;
	double w_no_[][] = null;
	double w_ho_[][] = null;
	double w_out_[][] = null;
	
	double b_i[] = null;
	double b_z[] = null;
	double b_f[] = null;
	double b_o[] = null;
	double b_out[] = null;
	
	double b_i_[] = null;
	double b_z_[] = null;
	double b_f_[] = null;
	double b_o_[] = null;
	double b_out_[] = null;
	
	double input_i[][] = null;
	double forget_f[][] = null; 
	double block_z[][] = null;
	double cell_c[][] = null;
	double cell_c_ts[][] = null;
	double output_o[][] = null;
	double output_y[][] = null;
	double output_out[][] = null;
	double output_[][] = null;
	double hidden_pre[] = null;
	double cell_pre[] = null;
	
	double error_out[][] = null;
	double error_y[][] = null;
	double error_o[][] = null;
	double error_c[][] = null;
	double error_z[][] = null;
	double error_f[][] = null;
	double error_i[][] = null;
	
	double cache_w_ni_m[][] = null;
	double cache_w_ni_v[][] = null;
	double cache_w_hi_m[][] = null;
	double cache_w_hi_v[][] = null;
	double cache_w_nf_m[][] = null;
	double cache_w_nf_v[][] = null;
	double cache_w_hf_m[][] = null;
	double cache_w_hf_v[][] = null;
	double cache_w_no_m[][] = null;
	double cache_w_no_v[][] = null;
	double cache_w_ho_m[][] = null;
	double cache_w_ho_v[][] = null;
	double cache_w_nz_m[][] = null;
	double cache_w_nz_v[][] = null;
	double cache_w_hz_m[][] = null;
	double cache_w_hz_v[][] = null;
	double cache_w_out_m[][] = null;
	double cache_w_out_v[][] = null;
	
	double cache_b_i_m[] = null;
	double cache_b_i_v[] = null;
	double cache_b_f_m[] = null;
	double cache_b_f_v[] = null;
	double cache_b_o_m[] = null;
	double cache_b_o_v[] = null;
	double cache_b_z_m[] = null;
	double cache_b_z_v[] = null;
	double cache_b_out_m[] = null;
	double cache_b_out_v[] = null;
	
	int input_len;
	int hidden_len;
	int output_len;
	
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
	
	int N;
	boolean isadam;
	DoubleMax dm = null;
	public LSTMFould(int input_len, int hidden_len, int output_len,
			int epochs, int mini_batch, int unfold_size,
			double decay_rate, double learning_rate, double eps,
			double beta_adam1, double beta_adam2, double drop){
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
		w_ni = new double[hidden_len][input_len];
		w_nf = new double[hidden_len][input_len];
		w_no = new double[hidden_len][input_len];
		w_nz = new double[hidden_len][input_len];
		w_out = new double[output_len][hidden_len];
		
		w_ni_ = new double[hidden_len][input_len];
		w_nf_ = new double[hidden_len][input_len];
		w_no_ = new double[hidden_len][input_len];
		w_nz_ = new double[hidden_len][input_len];
		w_out_ = new double[output_len][hidden_len];
		
		w_hi = new double[hidden_len][hidden_len];
		w_hf = new double[hidden_len][hidden_len];
		w_ho = new double[hidden_len][hidden_len];
		w_hz = new double[hidden_len][hidden_len];
		
		w_hi_ = new double[hidden_len][hidden_len];
		w_hf_ = new double[hidden_len][hidden_len];
		w_ho_ = new double[hidden_len][hidden_len];
		w_hz_ = new double[hidden_len][hidden_len];
		
		b_i = new double[hidden_len];
		b_f = new double[hidden_len];
		b_o = new double[hidden_len];
		b_z = new double[hidden_len];
		b_out = new double[output_len];
		
		b_i_ = new double[hidden_len];
		b_f_ = new double[hidden_len];
		b_o_ = new double[hidden_len];
		b_z_ = new double[hidden_len];
		b_out_ = new double[output_len];
		
		input_i    = new double[unfold_count][hidden_len];
		forget_f   = new double[unfold_count][hidden_len];
		block_z    = new double[unfold_count][hidden_len];
		cell_c     = new double[unfold_count][hidden_len];
		cell_c_ts  = new double[unfold_count][hidden_len];
		output_o   = new double[unfold_count][hidden_len];
		output_y   = new double[unfold_count][hidden_len];
		output_out = new double[unfold_count][output_len];
		output_ = new double[unfold_count][output_len];
		hidden_pre = new double[hidden_len];
		cell_pre   = new double[hidden_len];
		
		error_y = new double[unfold_count][hidden_len];
		error_o = new double[unfold_count][hidden_len];
		error_c = new double[unfold_count][hidden_len];
		error_z = new double[unfold_count][hidden_len];
		error_f = new double[unfold_count][hidden_len];
		error_i = new double[unfold_count][hidden_len];
		error_out = new double[unfold_count][output_len];
		
		cache_w_ni_m = new double[hidden_len][input_len];
		cache_w_nf_m = new double[hidden_len][input_len];
		cache_w_no_m = new double[hidden_len][input_len];
		cache_w_nz_m = new double[hidden_len][input_len];

		cache_w_hi_m = new double[hidden_len][hidden_len];
		cache_w_hf_m = new double[hidden_len][hidden_len];
		cache_w_ho_m = new double[hidden_len][hidden_len];
		cache_w_hz_m = new double[hidden_len][hidden_len];
		cache_w_out_m = new double[output_len][hidden_len];
		
		cache_b_i_m = new double[hidden_len];
		cache_b_f_m = new double[hidden_len];
		cache_b_o_m = new double[hidden_len];
		cache_b_z_m = new double[hidden_len];
		cache_b_out_m = new double[output_len];
		
		if(!isadam){
			Arrays.fill(cache_b_i_m, 1);
			Arrays.fill(cache_b_f_m, 1);
			Arrays.fill(cache_b_o_m, 1);
			Arrays.fill(cache_b_z_m, 1);
			Arrays.fill(cache_b_out_m, 1);
			for(int i=0; i<hidden_len; i++){
				Arrays.fill(cache_w_ni_m[i], 1);
				Arrays.fill(cache_w_hi_m[i], 1);
				Arrays.fill(cache_w_nf_m[i], 1);
				Arrays.fill(cache_w_hf_m[i], 1);
				Arrays.fill(cache_w_no_m[i], 1);
				Arrays.fill(cache_w_ho_m[i], 1);
				Arrays.fill(cache_w_nz_m[i], 1);
				Arrays.fill(cache_w_hz_m[i], 1);			
			}
			for(int i=0; i<output_len; i++)
				Arrays.fill(cache_w_out_m[i], 1);
		}else{
			cache_w_ni_v = new double[hidden_len][input_len];
			cache_w_nf_v = new double[hidden_len][input_len];
			cache_w_no_v = new double[hidden_len][input_len];
			cache_w_nz_v = new double[hidden_len][input_len];

			cache_w_hi_v = new double[hidden_len][hidden_len];
			cache_w_hf_v = new double[hidden_len][hidden_len];
			cache_w_ho_v = new double[hidden_len][hidden_len];
			cache_w_hz_v = new double[hidden_len][hidden_len];
			cache_w_out_v = new double[output_len][hidden_len];
			
			cache_b_i_v = new double[hidden_len];
			cache_b_f_v = new double[hidden_len];
			cache_b_o_v = new double[hidden_len];
			cache_b_z_v = new double[hidden_len];
			cache_b_out_v = new double[output_len];
		}
		dm = new DoubleMax();
	}
	public void init(boolean gass){
		Random rand = new Random();
		if(!gass){
			for(int i=0; i<hidden_len; i++){
				for(int j=0; j<input_len; j++){
					w_ni[i][j] = (rand.nextDouble()-0.5)/5;
					w_nf[i][j] = (rand.nextDouble()-0.5)/5;
					w_no[i][j] = (rand.nextDouble()-0.5)/5;
					w_nz[i][j] = (rand.nextDouble()-0.5)/5;
				}
				for(int j=0; j<hidden_len; j++){
					w_hi[i][j] = (rand.nextDouble()-0.5)/5;
					w_hf[i][j] = (rand.nextDouble()-0.5)/5;
					w_ho[i][j] = (rand.nextDouble()-0.5)/5;
					w_hz[i][j] = (rand.nextDouble()-0.5)/5;					
				}
			}
			for(int i=0; i<output_len; i++)
				for(int j=0; j<hidden_len; j++)
					w_out[i][j] = (rand.nextDouble()-0.5)/5;
			
		}else{
			for(int i=0; i<hidden_len; i++){
				for(int j=0; j<input_len; j++){
					w_ni[i][j] = rand.nextGaussian()/100;
					w_nf[i][j] = rand.nextGaussian()/100;
					w_no[i][j] = rand.nextGaussian()/100;
					w_nz[i][j] = rand.nextGaussian()/100;
				}
				for(int j=0; j<hidden_len; j++){
					w_hi[i][j] = rand.nextGaussian()/100;
					w_hf[i][j] = rand.nextGaussian()/100;
					w_ho[i][j] = rand.nextGaussian()/100;
					w_hz[i][j] = rand.nextGaussian()/100;					
				}
			}
			for(int i=0; i<output_len; i++)
				for(int j=0; j<hidden_len; j++)
					w_out[i][j] = rand.nextGaussian()/100;	
		}
		Arrays.fill(b_i, 0);
		Arrays.fill(b_f, 1);
		Arrays.fill(b_o, 0);
		Arrays.fill(b_z, 0);
		Arrays.fill(b_out, 1);
	}
	public void forward(double train_x[][], double train_y[][], int n){
		for(int i=0; i<unfold_size ; i++){
			if(i==0){
				dm.addi(dm.mul_nm_m1(w_ni, dm.dot_(dropout_i[0], train_x[i+n])), dm.mul_nm_m1(w_hi, dm.dot_(dropout_h[0], hidden_pre)), input_i[i]);
				dm.addi(input_i[i], b_i);
				dm.sigmoidi(input_i[i]);
				dm.addi(dm.mul_nm_m1(w_nf, dm.dot_(dropout_i[1], train_x[i+n])), dm.mul_nm_m1(w_hf, dm.dot_(dropout_h[1], hidden_pre)), forget_f[i]);
				dm.addi(forget_f[i], b_f);
				dm.sigmoidi(forget_f[i]);
				dm.addi(dm.mul_nm_m1(w_no, dm.dot_(dropout_i[2], train_x[i+n])), dm.mul_nm_m1(w_ho, dm.dot_(dropout_h[2], hidden_pre)), output_o[i]);
				dm.addi(output_o[i], b_o);
				dm.sigmoidi(output_o[i]);
				dm.addi(dm.mul_nm_m1(w_nz, dm.dot_(dropout_i[3], train_x[i+n])), dm.mul_nm_m1(w_hz, dm.dot_(dropout_h[3], hidden_pre)), block_z[i]);
				dm.addi(block_z[i], b_z);
				dm.tanhi(block_z[i]);
				dm.addi(dm.dot_(input_i[i], block_z[i]), dm.dot_(forget_f[i], cell_pre), cell_c[i]);
			}else{
				dm.addi(dm.mul_nm_m1(w_ni, dm.dot_(dropout_i[0], train_x[i+n])), dm.mul_nm_m1(w_hi, dm.dot_(dropout_h[0], output_y[i-1])), input_i[i]);
				dm.addi(input_i[i], b_i);
				dm.sigmoidi(input_i[i]);
				dm.addi(dm.mul_nm_m1(w_nf, dm.dot_(dropout_i[1], train_x[i+n])), dm.mul_nm_m1(w_hf, dm.dot_(dropout_h[1], output_y[i-1])), forget_f[i]);
				dm.addi(forget_f[i], b_f);
				dm.sigmoidi(forget_f[i]);
				dm.addi(dm.mul_nm_m1(w_no, dm.dot_(dropout_i[2], train_x[i+n])), dm.mul_nm_m1(w_ho, dm.dot_(dropout_h[2], output_y[i-1])), output_o[i]);
				dm.addi(output_o[i], b_o);
				dm.sigmoidi(output_o[i]);
				dm.addi(dm.mul_nm_m1(w_nz, dm.dot_(dropout_i[3], train_x[i+n])), dm.mul_nm_m1(w_hz, dm.dot_(dropout_h[3], output_y[i-1])), block_z[i]);
				dm.addi(block_z[i], b_z);
				dm.tanhi(block_z[i]);
				dm.addi(dm.dot_(input_i[i], block_z[i]), dm.dot_(forget_f[i], cell_c[i-1]), cell_c[i]);
			}
			cell_c_ts[i] = dm.tanh_(cell_c[i]);
			System.arraycopy(dm.dot_(output_o[i], cell_c_ts[i]), 0, output_y[i], 0, output_y[i].length);
			dm.addi(dm.mul_nm_m1(w_out, output_y[i]), b_out, output_out[i]);
			System.arraycopy(output_out[i], 0, output_[i], 0, output_len);
			dm.softmaxi(output_out[i]);
		}
	//	dm.print(input_i);
	//	System.arraycopy(output_y[unfold_size-1], 0, hidden_pre, 0, hidden_pre.length);
	//	System.arraycopy(cell_c[unfold_size-1], 0, cell_pre, 0, cell_pre.length);
	/*	for(double a : hidden_pre){
			System.out.print(a+" h ");
		}System.out.println();
		for(double a : cell_pre){
			System.out.print(a+" c ");
		}System.out.println();*/

	}
	public void backward(double train_x[][], double train_y[][], int n){
		int j = unfold_size-1;
		for(int i=j; i>=0; i--){
			 dm.subi(output_out[i], train_y[i+n], error_out[i]);
			 dm.addi(b_out_, error_out[i]);// bp b
			 dm.addi(w_out_, dm.mul_n1_1m(error_out[i], output_y[i]));// bp w
			if(i == j && j != 0){
				error_y[i] = dm.mul_nm_m1(dm.T(w_out), error_out[i]);
				dm.doti(dm.dot(dm.dev_sigmoid(output_o[i]), error_y[i]), cell_c_ts[i], error_o[i]);
				dm.addi(b_o_, error_o[i]);//  bp b
				dm.addi(w_no_, dm.mul_n1_1m(error_o[i], dm.dot_(dropout_i[2], train_x[i+n])));// bp w
				dm.doti(dm.dot(dm.dev_tanh(cell_c_ts[i]), error_y[i]), output_o[i], error_c[i]);
				dm.doti(dm.dot(dm.dev_sigmoid(forget_f[i]), error_c[i]), cell_c[i-1], error_f[i]);
				dm.addi(b_f_, error_f[i]);// bp b
				dm.addi(w_nf_, dm.mul_n1_1m(error_f[i], dm.dot_(dropout_i[1], train_x[i+n])));// bp w
			}else{
				dm.addi(dm.add(dm.mul_nm_m1(dm.T(w_hi), error_i[i+1]), dm.mul_nm_m1(dm.T(w_hf), error_f[i+1])),
						dm.add(dm.mul_nm_m1(dm.T(w_ho), error_o[i+1]), dm.mul_nm_m1(dm.T(w_hz), error_z[i+1])), error_y[i]);
				dm.add(error_y[i], dm.mul_nm_m1(dm.T(w_out), error_out[i]));
				dm.doti(dm.dot(dm.dev_sigmoid(output_o[i]), error_y[i]), cell_c_ts[i], error_o[i]);
				dm.addi(b_o_, error_o[i]);//  bp b
				dm.addi(w_no_, dm.mul_n1_1m(error_o[i], dm.dot_(dropout_i[2], train_x[i+n])));// bp w
				dm.addi(w_ho_, dm.mul_n1_1m(error_o[i+1], dm.dot_(dropout_h[2], output_y[i])));// bp w hidden
				dm.doti(dm.dot(dm.dev_tanh(cell_c_ts[i]), error_y[i]), output_o[i], error_c[i]);
				dm.add(error_c[i], dm.dot_(error_c[i+1], forget_f[i]));
				if(i>0)
					dm.doti(dm.dot(dm.dev_sigmoid(forget_f[i]), error_c[i]), cell_c[i-1], error_f[i]);		
				dm.addi(b_f_, error_f[i]);// bp b
				dm.addi(w_nf_, dm.mul_n1_1m(error_f[i], dm.dot_(dropout_i[1], train_x[i+n])));// bp w
				dm.addi(w_hf_, dm.mul_n1_1m(error_f[i+1], dm.dot_(dropout_h[1], output_y[i])));// bp w hidden

			}
			dm.doti(dm.dot(dm.dev_sigmoid(input_i[i]), error_c[i]), block_z[i], error_i[i]);
			dm.addi(b_i_, error_i[i]);// bp b
			dm.addi(w_ni_, dm.mul_n1_1m(error_i[i], dm.dot_(dropout_i[0], train_x[i+n])));// bp w
			if(i < j)
				dm.addi(w_hi_, dm.mul_n1_1m(error_i[i+1], dm.dot_(dropout_h[0], output_y[i])));// bp w hidden
			dm.doti(dm.dot(dm.dev_tanh(block_z[i]), error_c[i]), input_i[i], error_z[i]);
			dm.addi(b_z_, error_z[i]);// bp b
			dm.addi(w_nz_, dm.mul_n1_1m(error_z[i], dm.dot_(dropout_i[3], train_x[i+n])));// bp w
			if(i < j)
				dm.addi(w_hz_, dm.mul_n1_1m(error_z[i+1], dm.dot_(dropout_h[3], output_y[i])));// bp w hidden
		}
	}
	public void update_rmsprop(){
		dm.dot__(w_ni_, 1.0/mini_batch);
		dm.dot__(w_hi_, 1.0/mini_batch);
		dm.dot__(w_nf_, 1.0/mini_batch);
		dm.dot__(w_hf_, 1.0/mini_batch);
		dm.dot__(w_no_, 1.0/mini_batch);
		dm.dot__(w_ho_, 1.0/mini_batch);
		dm.dot__(w_nz_, 1.0/mini_batch);
		dm.dot__(w_hz_, 1.0/mini_batch);
		dm.dot__(w_out_, 1.0/mini_batch);
		dm.dot_(b_i_, 1.0/mini_batch);
		dm.dot_(b_f_, 1.0/mini_batch);
		dm.dot_(b_o_, 1.0/mini_batch);
		dm.dot_(b_z_, 1.0/mini_batch);
		dm.dot_(b_out_, 1.0/mini_batch);
		dm.clip(b_f_, -15, 15);
		dm.clip(b_i_, -15, 15);
		dm.clip(b_o_, -15, 15);
		dm.clip(b_z_, -15, 15);
		dm.clip(b_out_, -15, 15);
		dm.clip(w_ni_, -15, 15);
		dm.clip(w_hi_, -15, 15);
		dm.clip(w_nf_, -15, 15);
		dm.clip(w_hf_, -15, 15);
		dm.clip(w_no_, -15, 15);
		dm.clip(w_ho_, -15, 15);
		dm.clip(w_nz_, -15, 15);
		dm.clip(w_hz_, -15, 15);
		dm.clip(w_out_, -15, 15);

		dm.dot_(cache_b_out_m, decay_rate);
		dm.addi(cache_b_out_m, dm.dot_o(dm.pow(b_out_), 1-decay_rate));
		dm.subi(b_out, dm.dot_o(dm.dot(dm.dev_sqrt_(dm.add_A_(cache_b_out_m, eps)), b_out_), learning_rate));
		dm.dot_(cache_b_i_m, decay_rate);
		dm.addi(cache_b_i_m, dm.dot_o(dm.pow(b_i_), 1-decay_rate));
		dm.subi(b_i, dm.dot_o(dm.dot(dm.dev_sqrt_(dm.add_A_(cache_b_i_m, eps)), b_i_), learning_rate));
		dm.dot_(cache_b_f_m, decay_rate);
		dm.addi(cache_b_f_m, dm.dot_o(dm.pow(b_f_), 1-decay_rate));
		dm.subi(b_f, dm.dot_o(dm.dot(dm.dev_sqrt_(dm.add_A_(cache_b_f_m, eps)), b_f_), learning_rate));
		dm.dot_(cache_b_o_m, decay_rate);
		dm.addi(cache_b_o_m, dm.dot_o(dm.pow(b_o_), 1-decay_rate));
		dm.subi(b_o, dm.dot_o(dm.dot(dm.dev_sqrt_(dm.add_A_(cache_b_o_m, eps)), b_o_), learning_rate));
		dm.dot_(cache_b_z_m, decay_rate);
		dm.addi(cache_b_z_m, dm.dot_o(dm.pow(b_z_), 1-decay_rate));
		dm.subi(b_z, dm.dot_o(dm.dot(dm.dev_sqrt_(dm.add_A_(cache_b_z_m, eps)), b_z_), learning_rate));

		dm.dot__(cache_w_out_m, decay_rate);
		dm.addi(cache_w_out_m, dm.dot__o(dm.pow(w_out_), 1-decay_rate));
		dm.subi(w_out, dm.dot__o(dm.dot__(dm.dev_sqrt_(dm.add_A_(cache_w_out_m, eps)), w_out_), learning_rate));
		dm.dot__(cache_w_ni_m, decay_rate);
		dm.addi(cache_w_ni_m, dm.dot__o(dm.pow(w_ni_), 1-decay_rate));
		dm.subi(w_ni, dm.dot__o(dm.dot__(dm.dev_sqrt_(dm.add_A_(cache_w_ni_m, eps)), w_ni_), learning_rate));
		dm.dot__(cache_w_hi_m, decay_rate);
		dm.addi(cache_w_hi_m, dm.dot__o(dm.pow(w_hi_), 1-decay_rate));
		dm.subi(w_hi, dm.dot__o(dm.dot__(dm.dev_sqrt_(dm.add_A_(cache_w_hi_m, eps)), w_hi_), learning_rate));
		dm.dot__(cache_w_nf_m, decay_rate);
		dm.addi(cache_w_nf_m, dm.dot__o(dm.pow(w_nf_), 1-decay_rate));
		dm.subi(w_nf, dm.dot__o(dm.dot__(dm.dev_sqrt_(dm.add_A_(cache_w_nf_m, eps)), w_nf_), learning_rate));
		dm.dot__(cache_w_hf_m, decay_rate);
		dm.addi(cache_w_hf_m, dm.dot__o(dm.pow(w_hf_), 1-decay_rate));
		dm.subi(w_hf, dm.dot__o(dm.dot__(dm.dev_sqrt_(dm.add_A_(cache_w_hf_m, eps)), w_hf_), learning_rate));
		dm.dot__(cache_w_no_m, decay_rate);
		dm.addi(cache_w_no_m, dm.dot__o(dm.pow(w_no_), 1-decay_rate));
		dm.subi(w_no, dm.dot__o(dm.dot__(dm.dev_sqrt_(dm.add_A_(cache_w_no_m, eps)), w_no_), learning_rate));
		dm.dot__(cache_w_ho_m, decay_rate);
		dm.addi(cache_w_ho_m, dm.dot__o(dm.pow(w_ho_), 1-decay_rate));
		dm.subi(w_ho, dm.dot__o(dm.dot__(dm.dev_sqrt_(dm.add_A_(cache_w_ho_m, eps)), w_ho_), learning_rate));

		dm.clear(w_ni_);
		dm.clear(w_hi_);
		dm.clear(w_nf_);
		dm.clear(w_hf_);
		dm.clear(w_no_);
		dm.clear(w_ho_);
		dm.clear(w_nz_);
		dm.clear(w_hz_);
		dm.clear(w_out_);
		Arrays.fill(b_i_, 0);
		Arrays.fill(b_f_, 0);
		Arrays.fill(b_o_, 0);
		Arrays.fill(b_z_, 0);
		Arrays.fill(b_out_, 0);
	}
	public void update_adam(){	
		
		dm.dot__(w_ni_, 1.0/mini_batch);
		dm.dot__(w_hi_, 1.0/mini_batch);
		dm.dot__(w_nf_, 1.0/mini_batch);
		dm.dot__(w_hf_, 1.0/mini_batch);
		dm.dot__(w_no_, 1.0/mini_batch);
		dm.dot__(w_ho_, 1.0/mini_batch);
		dm.dot__(w_nz_, 1.0/mini_batch);
		dm.dot__(w_hz_, 1.0/mini_batch);
		dm.dot__(w_out_, 1.0/mini_batch);
		dm.dot_(b_i_, 1.0/mini_batch);
		dm.dot_(b_f_, 1.0/mini_batch);
		dm.dot_(b_o_, 1.0/mini_batch);
		dm.dot_(b_z_, 1.0/mini_batch);
		dm.dot_(b_out_, 1.0/mini_batch);

		dm.clip(b_f_, -15, 15);
		dm.clip(b_i_, -15, 15);
		dm.clip(b_o_, -15, 15);
		dm.clip(b_z_, -15, 15);
		dm.clip(b_out_, -15, 15);
		dm.clip(w_ni_, -15, 15);
		dm.clip(w_hi_, -15, 15);
		dm.clip(w_nf_, -15, 15);
		dm.clip(w_hf_, -15, 15);
		dm.clip(w_no_, -15, 15);
		dm.clip(w_ho_, -15, 15);
		dm.clip(w_nz_, -15, 15);
		dm.clip(w_hz_, -15, 15);
		dm.clip(w_out_, -15, 15);

		dm.addi(dm.dot_o_(b_out_, 1-beta_adam1), dm.dot_o(cache_b_out_m, beta_adam1), cache_b_out_m);
		dm.addi(dm.dot_o(dm.dot_(b_out_, b_out_), 1-beta_adam2), dm.dot_o(cache_b_out_v, beta_adam2), cache_b_out_v);		
		dm.subi(b_out, dm.dot_o(dm.dot(dm.div_m(dm.add_A(dm.sqrt(cache_b_out_v), eps)), cache_b_out_m), 
					learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.addi(dm.dot_o_(b_i_, 1-beta_adam1), dm.dot_o(cache_b_i_m, beta_adam1), cache_b_i_m);
		dm.addi(dm.dot_o(dm.dot_(b_i_, b_i_), 1-beta_adam2), dm.dot_o(cache_b_i_v, beta_adam2), cache_b_i_v);		
		dm.subi(b_i, dm.dot_o(dm.dot(dm.div_m(dm.add_A(dm.sqrt(cache_b_i_v), eps)), cache_b_i_m), 
					learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.addi(dm.dot_o_(b_f_, 1-beta_adam1), dm.dot_o(cache_b_f_m, beta_adam1), cache_b_f_m);
		dm.addi(dm.dot_o(dm.dot_(b_f_, b_f_), 1-beta_adam2), dm.dot_o(cache_b_f_v, beta_adam2), cache_b_f_v);		
		dm.subi(b_f, dm.dot_o(dm.dot(dm.div_m(dm.add_A(dm.sqrt(cache_b_f_v), eps)), cache_b_f_m), 
					learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.addi(dm.dot_o_(b_o_, 1-beta_adam1), dm.dot_o(cache_b_o_m, beta_adam1), cache_b_o_m);
		dm.addi(dm.dot_o(dm.dot_(b_o_, b_o_), 1-beta_adam2), dm.dot_o(cache_b_o_v, beta_adam2), cache_b_o_v);		
		dm.subi(b_o, dm.dot_o(dm.dot(dm.div_m(dm.add_A(dm.sqrt(cache_b_o_v), eps)), cache_b_o_m), 
					learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.addi(dm.dot_o_(b_z_, 1-beta_adam1), dm.dot_o(cache_b_z_m, beta_adam1), cache_b_z_m);
		dm.addi(dm.dot_o(dm.dot_(b_z_, b_z_), 1-beta_adam2), dm.dot_o(cache_b_z_v, beta_adam2), cache_b_z_v);		
		dm.subi(b_z, dm.dot_o(dm.dot(dm.div_m(dm.add_A(dm.sqrt(cache_b_z_v), eps)), cache_b_z_m), 
					learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));

		dm.add_(dm.dot__o_(w_out_, 1-beta_adam1), dm.dot__o(cache_w_out_m, beta_adam1), cache_w_out_m);
		dm.add_(dm.dot__o(dm.dot__(w_out_, w_out_), 1-beta_adam2), dm.dot__o(cache_w_out_v, beta_adam2), cache_w_out_v);
		dm.subi(w_out, dm.dot__o(dm.dot__(dm.div_m(dm.add_A(dm.sqrt(cache_w_out_v), eps)), cache_w_out_m), 
					learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.add_(dm.dot__o_(w_ni_, 1-beta_adam1), dm.dot__o(cache_w_ni_m, beta_adam1), cache_w_ni_m);
		dm.add_(dm.dot__o(dm.dot__(w_ni_, w_ni_), 1-beta_adam2), dm.dot__o(cache_w_ni_v, beta_adam2), cache_w_ni_v);
		dm.subi(w_ni, dm.dot__o(dm.dot__(dm.div_m(dm.add_A(dm.sqrt(cache_w_ni_v), eps)), cache_w_ni_m), 
					learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.add_(dm.dot__o_(w_hi_, 1-beta_adam1), dm.dot__o(cache_w_hi_m, beta_adam1), cache_w_hi_m);
		dm.add_(dm.dot__o(dm.dot__(w_hi_, w_hi_), 1-beta_adam2), dm.dot__o(cache_w_hi_v, beta_adam2), cache_w_hi_v);
		dm.subi(w_hi, dm.dot__o(dm.dot__(dm.div_m(dm.add_A(dm.sqrt(cache_w_hi_v), eps)), cache_w_hi_m), 
					learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.add_(dm.dot__o_(w_nf_, 1-beta_adam1), dm.dot__o(cache_w_nf_m, beta_adam1), cache_w_nf_m);
		dm.add_(dm.dot__o(dm.dot__(w_nf_, w_nf_), 1-beta_adam2), dm.dot__o(cache_w_nf_v, beta_adam2), cache_w_nf_v);
		dm.subi(w_nf, dm.dot__o(dm.dot__(dm.div_m(dm.add_A(dm.sqrt(cache_w_nf_v), eps)), cache_w_nf_m), 
					learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.add_(dm.dot__o_(w_hf_, 1-beta_adam1), dm.dot__o(cache_w_hf_m, beta_adam1), cache_w_hf_m);
		dm.add_(dm.dot__o(dm.dot__(w_hf_, w_hf_), 1-beta_adam2), dm.dot__o(cache_w_hf_v, beta_adam2), cache_w_hf_v);
		dm.subi(w_hf, dm.dot__o(dm.dot__(dm.div_m(dm.add_A(dm.sqrt(cache_w_hf_v), eps)), cache_w_hf_m), 
					learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.add_(dm.dot__o_(w_no_, 1-beta_adam1), dm.dot__o(cache_w_no_m, beta_adam1), cache_w_no_m);
		dm.add_(dm.dot__o(dm.dot__(w_no_, w_no_), 1-beta_adam2), dm.dot__o(cache_w_no_v, beta_adam2), cache_w_no_v);
		dm.subi(w_no, dm.dot__o(dm.dot__(dm.div_m(dm.add_A(dm.sqrt(cache_w_no_v), eps)), cache_w_no_m), 
					learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.add_(dm.dot__o_(w_ho_, 1-beta_adam1), dm.dot__o(cache_w_ho_m, beta_adam1), cache_w_ho_m);
		dm.add_(dm.dot__o(dm.dot__(w_ho_, w_ho_), 1-beta_adam2), dm.dot__o(cache_w_ho_v, beta_adam2), cache_w_ho_v);
		dm.subi(w_ho, dm.dot__o(dm.dot__(dm.div_m(dm.add_A(dm.sqrt(cache_w_ho_v), eps)), cache_w_ho_m), 
					learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.add_(dm.dot__o_(w_nz_, 1-beta_adam1), dm.dot__o(cache_w_nz_m, beta_adam1), cache_w_nz_m);
		dm.add_(dm.dot__o(dm.dot__(w_nz_, w_nz_), 1-beta_adam2), dm.dot__o(cache_w_nz_v, beta_adam2), cache_w_nz_v);
		dm.subi(w_nz, dm.dot__o(dm.dot__(dm.div_m(dm.add_A(dm.sqrt(cache_w_nz_v), eps)), cache_w_nz_m), 
					learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.add_(dm.dot__o_(w_hz_, 1-beta_adam1), dm.dot__o(cache_w_hz_m, beta_adam1), cache_w_hz_m);
		dm.add_(dm.dot__o(dm.dot__(w_hz_, w_hz_), 1-beta_adam2), dm.dot__o(cache_w_hz_v, beta_adam2), cache_w_hz_v);
		dm.subi(w_hz, dm.dot__o(dm.dot__(dm.div_m(dm.add_A(dm.sqrt(cache_w_hz_v), eps)), cache_w_hz_m), 
					learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
/*
		for(double A[]:w_hz){
			for(double a:A){
				System.out.print(a+" ");
			}System.out.println();
		}*/
	//	dm.print(w_hz);
		dm.clear(w_ni_);
		dm.clear(w_hi_);
		dm.clear(w_nf_);
		dm.clear(w_hf_);
		dm.clear(w_no_);
		dm.clear(w_ho_);
		dm.clear(w_nz_);
		dm.clear(w_hz_);
		dm.clear(w_out_);
		Arrays.fill(b_i_, 0);
		Arrays.fill(b_f_, 0);
		Arrays.fill(b_o_, 0);
		Arrays.fill(b_z_, 0);
		Arrays.fill(b_out_, 0);
	}
	public void train(double train_x[][], double train_y[][], int A[]){
		N = train_x.length;
		while(epochs-- >= 0){
			System.out.println("epochs: "+(epochs+1));
			int a = 0;
			unfold_size = A[a];
			for(int i=0; i<N; ){
				dropout_i = dm.dropout(4, input_len, drop);
				dropout_h = dm.dropout(4, hidden_len, drop);
				for(int j=0; j<mini_batch; j++){
					forward(train_x, train_y, i);
					backward(train_x, train_y, i);
					i += A[a];
					if(a >=  A.length - 1) break;
					unfold_size = A[++a];
				}
				if(!isadam)
					update_rmsprop();
				else
					update_adam();
			}
		}
	}
	public void predict(double test_x[][], double test_y[][], double label_[][], int B[]){
		int right = 0;
		N = test_x.length;
		int a = 0;
		unfold_size = B[a];
		dm.fill(dropout_i, 1.0);
		dm.fill(dropout_h, 1.0);
		for(int i=0; i<N; ){				
			for(int j=0; j<mini_batch; j++){
				forward(test_x, test_y, i);
				right += fit(test_y, i, label_);
				i += B[a];
				if(a >= B.length - 1) break;
				unfold_size = B[++a];
			}
		}
		System.out.println("accuracy num: "+right+" all num: "+test_x.length);
		System.out.println("accuracy rate: "+1.0*right/test_x.length);
	}
	public int fit(double test_y[][], int j, double label_[][]){
		int a,b,c=0;
		for(int i=0; i<unfold_size && i+j<N; i++){
			a = dm.max_i(test_y[i+j]);
			b = dm.max_i(output_out[i]);
			if(a == b) c++;
			label_[i+j] = output_out[i].clone();
		}
		return c;
	}
}
