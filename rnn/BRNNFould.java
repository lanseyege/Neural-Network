
import java.util.Arrays;
import java.util.Random;

public class BRNNFould {
	double w_ih_f[][] = null;
	double w_hh_f[][] = null;
	double w_ho_f[][] = null;
	double w_ih_b[][] = null;
	double w_hh_b[][] = null;
	double w_ho_b[][] = null;
	double bias1_f[] = null;
	double bias1_b[] = null;
	double bias2[] = null;
	
	double w_ih_f_[][] = null;
	double w_hh_f_[][] = null;
	double w_ho_f_[][] = null;
	double w_ih_b_[][] = null;
	double w_hh_b_[][] = null;
	double w_ho_b_[][] = null;
	double bias1_f_[] = null;
	double bias1_b_[] = null;
	double bias2_[] = null;
	
	double hidden_f[][] = null;
	double hidden_b[][] = null;
	double output[][] = null;
	
	double error_o[][] = null;
	double error_h_f[][] = null;
	double error_h_b[][] = null;
	double error_h_[] = null;
	
	double cache_b2[] = null;
	double cache_b1_f[] = null;
	double cache_b1_b[] = null;
	double cache_ih_f[][] = null;
	double cache_ih_b[][] = null;
	double cache_hh_f[][] = null;
	double cache_hh_b[][] = null;
	double cache_ho_f[][] = null;
	double cache_ho_b[][] = null;	
	double cache_b2_v[] = null;
	double cache_b1_f_v[] = null;
	double cache_b1_b_v[] = null;
	double cache_ih_f_v[][] = null;
	double cache_ih_b_v[][] = null;
	double cache_hh_f_v[][] = null;
	double cache_hh_b_v[][] = null;
	double cache_ho_f_v[][] = null;
	double cache_ho_b_v[][] = null;
	
	double dropout[] = null;
	int input_len;
	int hidden_len;
	int output_len;
	int N;
	
	int epochs;
	int mini_batch;
	int unfold_size;
	int unfold_count = 120;
	double learning_rate;
	double decay_rate;
	double eps;
	double beta_adam1;
	double beta_adam2;
	double drop;
	boolean isadam ;
	
	DoubleMax dm = null;
	public BRNNFould(int input_len, int hidden_len, int output_len, 
			int epochs, int mini_batch, int unfold_size,
			double decay_rate, double learning_rate, double eps, 
			double beta_adam1, double beta_adam2, double drop){
		this.input_len = input_len;
		this.hidden_len = hidden_len;
		this.output_len = output_len;
		this.epochs = epochs;
		this.mini_batch = mini_batch;
		this.unfold_size = unfold_size;
		this.learning_rate = learning_rate;
		this.decay_rate = decay_rate;
		this.eps = eps;
		this.beta_adam1 = beta_adam1;
		this.beta_adam2 = beta_adam2;
		this.drop = drop;
	}
	public void net(boolean isadam){
		this.isadam = isadam;
		w_ih_f = new double[hidden_len][input_len];
		w_ih_b = new double[hidden_len][input_len];
		w_hh_f = new double[hidden_len][hidden_len];
		w_hh_b = new double[hidden_len][hidden_len];
		w_ho_f = new double[output_len][hidden_len];
		w_ho_b = new double[output_len][hidden_len];
		bias1_f = new double[hidden_len];
		bias1_b = new double[hidden_len];
		bias2 = new double[output_len];
		
		w_ih_f_ = new double[hidden_len][input_len];
		w_ih_b_ = new double[hidden_len][input_len];
		w_hh_f_ = new double[hidden_len][hidden_len];
		w_hh_b_ = new double[hidden_len][hidden_len];
		w_ho_f_ = new double[output_len][hidden_len];
		w_ho_b_ = new double[output_len][hidden_len];
		bias1_f_ = new double[hidden_len];
		bias1_b_ = new double[hidden_len];
		bias2_ = new double[output_len];
		
		hidden_f = new double[unfold_count][hidden_len];
		hidden_b = new double[unfold_count][hidden_len];
		output = new double[unfold_count][output_len];
		
		error_o = new double[unfold_count][output_len];
		error_h_f = new double[unfold_count][hidden_len];
		error_h_b = new double[unfold_count][hidden_len];
		error_h_ = new double[hidden_len];
		
		cache_b2 = new double[output_len];
		cache_b1_f = new double[hidden_len];
		cache_b1_b = new double[hidden_len];
		cache_ih_f = new double[hidden_len][input_len];
		cache_ih_b = new double[hidden_len][input_len];
		cache_hh_f = new double[hidden_len][hidden_len];
		cache_hh_b = new double[hidden_len][hidden_len];
		cache_ho_f = new double[output_len][hidden_len];
		cache_ho_b = new double[output_len][hidden_len];
		if(!isadam){
			Arrays.fill(cache_b2, 1);
			Arrays.fill(cache_b1_f,1);
			Arrays.fill(cache_b1_b,1);
			for(int i=0; i<hidden_len; i++){
				Arrays.fill(cache_ih_f[i], 1);
				Arrays.fill(cache_ih_b[i], 1);
				Arrays.fill(cache_hh_f[i], 1);
				Arrays.fill(cache_hh_b[i], 1);
			}
			for(int i=0; i<output_len; i++){
				Arrays.fill(cache_ho_f[i], 1);
				Arrays.fill(cache_ho_b[i], 1);
			}
		}else{
			cache_b2_v = new double[output_len];
			cache_b1_f_v = new double[hidden_len];
			cache_b1_b_v = new double[hidden_len];
			cache_ih_f_v = new double[hidden_len][input_len];
			cache_ih_b_v = new double[hidden_len][input_len];
			cache_hh_f_v = new double[hidden_len][hidden_len];
			cache_hh_b_v = new double[hidden_len][hidden_len];
			cache_ho_f_v = new double[output_len][hidden_len];
			cache_ho_b_v = new double[output_len][hidden_len];
		}
	}
	public void init(boolean isGass){
		if(isGass){
			Random random = new Random();
			for(int i=0; i<hidden_len; i++){
				for(int j=0; j<input_len; j++){
					w_ih_f[i][j] = random.nextGaussian()/100;
					w_ih_b[i][j] = random.nextGaussian()/100;
				}
				for(int j=0; j<hidden_len; j++){
					w_hh_f[i][j] = random.nextGaussian()/100;
					w_hh_b[i][j] = random.nextGaussian()/100;
				}
			}
			for(int i=0; i<output_len; i++){
				for(int j=0; j<hidden_len; j++){
					w_ho_f[i][j] = random.nextGaussian()/100;
					w_ho_b[i][j] = random.nextGaussian()/100;
				}
			}			
		}else{
			Random random = new Random();
			for(int i=0; i<hidden_len; i++){
				for(int j=0; j<input_len; j++){
					w_ih_f[i][j] = random.nextDouble()/100;
					w_ih_b[i][j] = random.nextDouble()/100;
				}
				for(int j=0; j<hidden_len; j++){
					w_hh_f[i][j] = random.nextDouble()/100;
					w_hh_b[i][j] = random.nextDouble()/100;
				}
			}
			for(int i=0; i<output_len; i++){
				for(int j=0; j<hidden_len; j++){
					w_ho_f[i][j] = random.nextDouble()/100;
					w_ho_b[i][j] = random.nextDouble()/100;
				}
			}
		}
		Arrays.fill(bias1_f, 1);
		Arrays.fill(bias1_b, 1);
		Arrays.fill(bias2, 1);
		
		dm = new DoubleMax();
	}
	public void forward(double train_x[][], double train_y[][], int n){
		forward_f(train_x, train_y, n);
		forward_b(train_x, train_y, n);
		for(int i=0; i<unfold_size ; i++){
			dm.addi(dm.add(dm.mul_nm_m1(w_ho_f, hidden_f[i]), 
					dm.mul_nm_m1(w_ho_b, hidden_b[i])), bias2, output[i]);
			dm.softmaxi(output[i]);
		}
	}
	private void forward_f(double train_x[][], double train_y[][], int n){
		dm.addi(dm.mul_nm_m1(w_ih_f, dm.dot_(dropout, train_x[0+n])), bias1_f, hidden_f[0]);
		dm.sigmoidi(hidden_f[0]);
		for(int i=1; i<unfold_size ; i++){
			dm.addi(dm.add(dm.mul_nm_m1(w_ih_f, dm.dot_(dropout, train_x[i+n])), 
						dm.mul_nm_m1(w_hh_f, hidden_f[i-1])), bias1_f, hidden_f[i]);
			dm.sigmoidi(hidden_f[i]);
		}
	}
	private void forward_b(double train_x[][], double train_y[][], int n){
		dm.addi(dm.mul_nm_m1(w_ih_b, dm.dot_(dropout, train_x[unfold_size-1+n])), bias1_b, hidden_b[unfold_size-1]);
		dm.sigmoidi(hidden_b[unfold_size-1]);
		for(int i=unfold_size-2; i>=0; i--){
			dm.addi(dm.add(dm.mul_nm_m1(w_ih_b, dm.dot_(dropout, train_x[i+n])), 
						dm.mul_nm_m1(w_hh_b, hidden_b[i+1])), bias1_b, hidden_b[i]);
			dm.sigmoidi(hidden_b[i]);
		}
	}
	public void backward(double train_x[][], double train_y[][], int n){
		for(int i=0; i<unfold_size; i++){			
			dm.subi(output[i], train_y[i+n], error_o[i]);
			dm.addi(bias2_, error_o[i]);
		}
		
		backward_f(train_x, train_y, n);
		backward_b(train_x, train_y, n);
	}
	private void backward_f(double[][] train_x, double[][] train_y, int n) {
		double A[] = null;
		for(int i=unfold_size-1; i>=0; i--){
			dm.addi(w_ho_f_, dm.mul_n1_1m(error_o[i], hidden_f[i]));
			A = dm.add(dm.mul_nm_m1(dm.T(w_ho_f), error_o[i]), error_h_);
			error_h_f[i] = dm.dot(A, dm.dev_sigmoid_(hidden_f[i]));
			dm.addi(bias1_f_, error_h_f[i]);
			dm.addi(w_ih_f_, dm.mul_n1_1m(error_h_f[i], dm.dot_(dropout, train_x[i+n])));
			if(i>0)
				dm.addi(w_hh_f_, dm.mul_n1_1m(error_h_f[i], hidden_f[i-1]));
			error_h_ = dm.mul_nm_m1(dm.T(w_hh_f), error_h_f[i]);
		}
		Arrays.fill(error_h_, 0.0);
	}
	private void backward_b(double[][] train_x, double[][] train_y, int n) {
		double A[] = null;
		int j = unfold_size-1;
		for(int i=0; i<=j; i++){
			dm.addi(w_ho_b_, dm.mul_n1_1m(error_o[i], hidden_b[i]));
			A = dm.add(dm.mul_nm_m1(dm.T(w_ho_b), error_o[i]), error_h_);
			error_h_b[i] = dm.dot(A, dm.dev_sigmoid_(hidden_b[i]));
			dm.addi(bias1_b_, error_h_b[i]);
			dm.addi(w_ih_b_, dm.mul_n1_1m(error_h_b[i], dm.dot_(dropout, train_x[i+n])));
			if(i<j)
				dm.addi(w_hh_b_, dm.mul_n1_1m(error_h_b[i], hidden_b[i+1]));
			error_h_ = dm.mul_nm_m1(dm.T(w_hh_b), error_h_b[i]);
		}
		Arrays.fill(error_h_, 0.0);
	}
	public void update_rmsprop(){
		dm.dot__(w_ih_f_, 1.0/mini_batch);
		dm.dot__(w_ih_b_, 1.0/mini_batch);
		dm.dot__(w_hh_f_, 1.0/mini_batch);
		dm.dot__(w_hh_b_, 1.0/mini_batch);
		dm.dot__(w_ho_f_, 1.0/mini_batch);
		dm.dot__(w_ho_b_, 1.0/mini_batch);
		dm.dot_(bias2_,   1.0/mini_batch);
		dm.dot_(bias1_f_, 1.0/mini_batch);
		dm.dot_(bias1_b_, 1.0/mini_batch);
		
		dm.dot_(cache_b2, decay_rate);
		dm.addi(cache_b2, dm.dot_o(dm.pow(bias2_), 1-decay_rate));
		dm.subi(bias2, dm.dot_o(dm.dot(dm.dev_sqrt_(dm.add_A_(cache_b2, eps)), bias2_), learning_rate));
		dm.dot_(cache_b1_f, decay_rate);
		dm.addi(cache_b1_f, dm.dot_o(dm.pow(bias1_f_), 1-decay_rate));
		dm.subi(bias1_f, dm.dot_o(dm.dot(dm.dev_sqrt_(dm.add_A_(cache_b1_f, eps)), bias1_f_), learning_rate));
		dm.dot_(cache_b1_b, decay_rate);
		dm.addi(cache_b1_b, dm.dot_o(dm.pow(bias1_b_), 1-decay_rate));
		dm.subi(bias1_b, dm.dot_o(dm.dot(dm.dev_sqrt_(dm.add_A_(cache_b1_b, eps)), bias1_b_), learning_rate));
		dm.dot__(cache_ho_f, decay_rate);
		dm.addi(cache_ho_f, dm.dot__o(dm.pow(w_ho_f_), 1-decay_rate));
		dm.subi(w_ho_f, dm.dot__o(dm.dot__(dm.dev_sqrt_(dm.add_A_(cache_ho_f, eps)), w_ho_f_), learning_rate));		
		dm.dot__(cache_ho_b, decay_rate);
		dm.addi(cache_ho_b, dm.dot__o(dm.pow(w_ho_b_), 1-decay_rate));
		dm.subi(w_ho_b, dm.dot__o(dm.dot__(dm.dev_sqrt_(dm.add_A_(cache_ho_b, eps)), w_ho_b_), learning_rate));
		dm.dot__(cache_hh_f, decay_rate);
		dm.addi(cache_hh_f, dm.dot__o(dm.pow(w_hh_f_), 1-decay_rate));
		dm.subi(w_hh_f, dm.dot__o(dm.dot__(dm.dev_sqrt_(dm.add_A_(cache_hh_f, eps)), w_hh_f_), learning_rate));
		dm.dot__(cache_hh_b, decay_rate);
		dm.addi(cache_hh_b, dm.dot__o(dm.pow(w_hh_b_), 1-decay_rate));
		dm.subi(w_hh_b, dm.dot__o(dm.dot__(dm.dev_sqrt_(dm.add_A_(cache_hh_b, eps)), w_hh_b_), learning_rate));
		dm.dot__(cache_ih_f, decay_rate);
		dm.addi(cache_ih_f, dm.dot__o(dm.pow(w_ih_f_), 1-decay_rate));
		dm.subi(w_ih_f, dm.dot__o(dm.dot__(dm.dev_sqrt_(dm.add_A_(cache_ih_f, eps)), w_ih_f_), learning_rate));
		dm.dot__(cache_ih_b, decay_rate);
		dm.addi(cache_ih_b, dm.dot__o(dm.pow(w_ih_b_), 1-decay_rate));
		dm.subi(w_ih_b, dm.dot__o(dm.dot__(dm.dev_sqrt_(dm.add_A_(cache_ih_b, eps)), w_ih_b_), learning_rate));
		
		dm.clear(w_ih_f_);
		dm.clear(w_ih_b_);
		dm.clear(w_hh_f_);
		dm.clear(w_hh_b_);
		dm.clear(w_ho_f_);
		dm.clear(w_hh_b_);
		Arrays.fill(bias2_, 0);
		Arrays.fill(bias1_f_, 0);
		Arrays.fill(bias1_b_, 0);
	}
	public void update_adam(){
		dm.dot__(w_ih_f_, 1.0/mini_batch);
		dm.dot__(w_ih_b_, 1.0/mini_batch);
		dm.dot__(w_hh_f_, 1.0/mini_batch);
		dm.dot__(w_hh_b_, 1.0/mini_batch);
		dm.dot__(w_ho_f_, 1.0/mini_batch);
		dm.dot__(w_ho_b_, 1.0/mini_batch);
		dm.dot_(bias2_,   1.0/mini_batch);
		dm.dot_(bias1_f_, 1.0/mini_batch);
		dm.dot_(bias1_b_, 1.0/mini_batch);

		dm.clip(w_ih_f_, -15, 15);
		dm.clip(w_ih_b_, -15, 15);
		dm.clip(w_hh_f_, -15, 15);
		dm.clip(w_hh_b_, -15, 15);
		dm.clip(w_ho_f_, -15, 15);
		dm.clip(w_ho_b_, -15, 15);
		dm.clip(bias2_, -15, 15);
		dm.clip(bias1_f_, -15, 15);
		dm.clip(bias1_b_, -15, 15);
		
		dm.addi(dm.dot_o_(bias1_f_, 1-beta_adam1), dm.dot_o(cache_b1_f, beta_adam1), cache_b1_f);
		dm.addi(dm.dot_o(dm.dot_(bias1_f_, bias1_f_), 1-beta_adam2), dm.dot_o(cache_b1_f_v, beta_adam2), cache_b1_f_v);		
		dm.subi(bias1_f, dm.dot_o(dm.dot(dm.div_m(dm.add_A(dm.sqrt(cache_b1_f_v), eps)), cache_b1_f), 
					learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.addi(dm.dot_o_(bias1_b_, 1-beta_adam1), dm.dot_o(cache_b1_b, beta_adam1), cache_b1_b);
		dm.addi(dm.dot_o(dm.dot_(bias1_b_, bias1_b_), 1-beta_adam2), dm.dot_o(cache_b1_b_v, beta_adam2), cache_b1_b_v);		
		dm.subi(bias1_b, dm.dot_o(dm.dot(dm.div_m(dm.add_A(dm.sqrt(cache_b1_b_v), eps)), cache_b1_b), 
					learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.addi(dm.dot_o_(bias2_, 1-beta_adam1), dm.dot_o(cache_b2, beta_adam1), cache_b2);
		dm.addi(dm.dot_o(dm.dot_(bias2_, bias2_), 1-beta_adam2), dm.dot_o(cache_b2_v, beta_adam2), cache_b2_v);		
		dm.subi(bias2, dm.dot_o(dm.dot(dm.div_m(dm.add_A(dm.sqrt(cache_b2_v), eps)), cache_b2), 
					learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.add_(dm.dot__o_(w_ho_f_, 1-beta_adam1), dm.dot__o(cache_ho_f, beta_adam1), cache_ho_f);
		dm.add_(dm.dot__o(dm.dot__(w_ho_f_, w_ho_f_), 1-beta_adam2), dm.dot__o(cache_ho_f_v, beta_adam2), cache_ho_f_v);
		dm.subi(w_ho_f, dm.dot__o(dm.dot__(dm.div_m(dm.add_A(dm.sqrt(cache_ho_f_v), eps)), cache_ho_f), 
					learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));		
		dm.add_(dm.dot__o_(w_ho_b_, 1-beta_adam1), dm.dot__o(cache_ho_b, beta_adam1), cache_ho_b);
		dm.add_(dm.dot__o(dm.dot__(w_ho_b_, w_ho_b_), 1-beta_adam2), dm.dot__o(cache_ho_b_v, beta_adam2), cache_ho_b_v);
		dm.subi(w_ho_b, dm.dot__o(dm.dot__(dm.div_m(dm.add_A(dm.sqrt(cache_ho_b_v), eps)), cache_ho_b), 
					learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.add_(dm.dot__o_(w_hh_f_, 1-beta_adam1), dm.dot__o(cache_hh_f, beta_adam1), cache_hh_f);
		dm.add_(dm.dot__o(dm.dot__(w_hh_f_, w_hh_f_), 1-beta_adam2), dm.dot__o(cache_hh_f_v, beta_adam2), cache_hh_f_v);
		dm.subi(w_hh_f, dm.dot__o(dm.dot__(dm.div_m(dm.add_A(dm.sqrt(cache_hh_f_v), eps)), cache_hh_f), 
					learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.add_(dm.dot__o_(w_hh_b_, 1-beta_adam1), dm.dot__o(cache_hh_b, beta_adam1), cache_hh_b);
		dm.add_(dm.dot__o(dm.dot__(w_hh_b_, w_hh_b_), 1-beta_adam2), dm.dot__o(cache_hh_b_v, beta_adam2), cache_hh_b_v);
		dm.subi(w_hh_b, dm.dot__o(dm.dot__(dm.div_m(dm.add_A(dm.sqrt(cache_hh_b_v), eps)), cache_hh_b), 
					learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.add_(dm.dot__o_(w_ih_f_, 1-beta_adam1), dm.dot__o(cache_ih_f, beta_adam1), cache_ih_f);
		dm.add_(dm.dot__o(dm.dot__(w_ih_f_, w_ih_f_), 1-beta_adam2), dm.dot__o(cache_ih_f_v, beta_adam2), cache_ih_f_v);
		dm.subi(w_ih_f, dm.dot__o(dm.dot__(dm.div_m(dm.add_A(dm.sqrt(cache_ih_f_v), eps)), cache_ih_f), 
					learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.add_(dm.dot__o_(w_ih_b_, 1-beta_adam1), dm.dot__o(cache_ih_b, beta_adam1), cache_ih_b);
		dm.add_(dm.dot__o(dm.dot__(w_ih_b_, w_ih_b_), 1-beta_adam2), dm.dot__o(cache_ih_b_v, beta_adam2), cache_ih_b_v);
		dm.subi(w_ih_b, dm.dot__o(dm.dot__(dm.div_m(dm.add_A(dm.sqrt(cache_ih_b_v), eps)), cache_ih_b), 
					learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));


		dm.clear(w_ih_f_);
		dm.clear(w_ih_b_);
		dm.clear(w_hh_f_);
		dm.clear(w_hh_b_);
		dm.clear(w_ho_f_);
		dm.clear(w_ho_b_);
		Arrays.fill(bias2_, 0);
		Arrays.fill(bias1_f_, 0);
		Arrays.fill(bias1_b_, 0);

	}
	public void train(double train_x[][], double train_y[][], int A[]){
		N = train_x.length;
		while(epochs-- >= 0){
			System.out.println("epochs: "+(epochs + 1));
			int a = 0;
			unfold_size = A[a];
			for(int i=0; i<N; ){
				dropout = dm.dropout(input_len, drop);
				for(int j=0; j<mini_batch; j++){
					forward(train_x, train_y, i);
					backward(train_x, train_y, i);
					i += A[a];
					if(a >= A.length - 1) break;
					unfold_size = A[++a];
				}
				if(! isadam) update_rmsprop();
				else	update_adam();
			}
		}
	}
	public void predict(double test_x[][], double test_y[][], double label_[][], int B[]){
		int right = 0;
		N = test_x.length;
		int a = 0;
		unfold_size = B[a];
		Arrays.fill(dropout, 1);
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
			b = dm.max_i(output[i]);
			if(a == b) c++;
			label_[i+j] = output[i].clone();
		}
		return c;
	}
}
