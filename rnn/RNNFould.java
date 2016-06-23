
import java.util.Arrays;
import java.util.Random;


public class RNNFould {

	double w_ih[][] = null;
	double w_ih_[][] = null;
	double w_hh[][] = null;
	double w_hh_[][] = null;
	double w_ho[][] = null;
	double w_ho_[][] = null;
	double input[] = null;
	double hidden[][] = null;
	double output[][] = null;
	double hidden_pre[] = null;
	double bias1[] = null;
	double bias1_[] = null;
	double bias2[] = null;
	double bias2_[] = null;
	double error_o[][] = null;
	double error_h[][] = null;
	double error_h_[] = null;
	
	double cache_b1[] = null;
	double cache_b2[] = null;
	double cache_ih[][] = null;
	double cache_hh[][] = null;
	double cache_ho[][] = null;
	double dropout[] = null;
	double decay_rate;
	double learning_rate;
	double eps;
	double beta;//rmsprop with momentum (Nesterov accelerated gradiet)
	double drop;	

	double beta_adam1;
	double beta_adam2;
	double cache_b1_v[] = null;
	double cache_b2_v[] = null;
	double cache_ih_v[][] = null;
	double cache_hh_v[][] = null;
	double cache_ho_v[][] = null;
	
	int N ;
	int input_len;
	int hidden_len;
	int output_len;
	int epochs;
	int unfold_size;
	int mini_batch;
	int unfold_count = 120;
	boolean isadam;
	DoubleMax dm = null;
	public RNNFould(int input_len, int hidden_len, int output_len, 
			int epochs, int mini_batch, int unfold_size, 
			double decay_rate, double learning_rate, double eps,
			double beta_adam1, double beta_adam2, double drop){
		this.input_len = input_len;
		this.hidden_len = hidden_len;
		this.output_len = output_len;
		
		this.epochs = epochs;
		this.unfold_size = unfold_size;
		this.mini_batch = mini_batch;
		this.decay_rate = decay_rate;
		this.learning_rate = learning_rate;
		this.eps = eps;
		this.beta_adam1 = beta_adam1;
		this.beta_adam2 = beta_adam2;
		this.drop = drop;
	}
	public void net(boolean isadam){
		this.isadam = isadam;
		w_ih = new double[hidden_len][input_len];
		w_ih_ = new double[hidden_len][input_len];
		w_hh = new double[hidden_len][hidden_len];
		w_hh_ = new double[hidden_len][hidden_len];
		w_ho = new double[output_len][hidden_len];
		w_ho_ = new double[output_len][hidden_len];
		bias1 = new double[hidden_len];
		bias1_ = new double[hidden_len];
		bias2 = new double[output_len];
		bias2_ = new double[output_len];
		error_o = new double[unfold_count][output_len];
		error_h = new double[unfold_count][hidden_len];
		error_h_ = new double[hidden_len];
		hidden = new double[unfold_count][hidden_len];
		output = new double[unfold_count][output_len];
		hidden_pre = new double[hidden_len];
		dm = new DoubleMax();
		
		cache_b1 = new double[hidden_len];
		cache_b2 = new double[output_len];
		cache_ih = new double[hidden_len][input_len];
		cache_hh = new double[hidden_len][hidden_len];
		cache_ho = new double[output_len][hidden_len];
		if(!isadam){
			Arrays.fill(cache_b1, 1);
			Arrays.fill(cache_b2, 1);		
			for(int i=0; i<hidden_len; i++){
				Arrays.fill(cache_ih[i], 1);
				Arrays.fill(cache_hh[i], 1);
			}
			for(int i=0; i<output_len; i++)
				Arrays.fill(cache_ho[i], 1);
		}else{
			cache_b1_v = new double[hidden_len];
			cache_b2_v = new double[output_len];
			cache_ih_v = new double[hidden_len][input_len];
			cache_hh_v = new double[hidden_len][hidden_len];
			cache_ho_v = new double[output_len][hidden_len];
			}
	}
	public void init(boolean gass){
		if(gass){
			Random random = new Random();
			for(int i=0; i<hidden_len; i++){
				for(int j=0; j<input_len; j++){
					w_ih[i][j] = random.nextGaussian()/100;					
				}
				for(int j=0; j<hidden_len; j++){
					w_hh[i][j] = random.nextGaussian()/100;
				}
			}
			for(int i=0; i<output_len; i++)
				for(int j=0; j<hidden_len; j++)
					w_ho[i][j] = random.nextGaussian()/100;
			Arrays.fill(bias1, 1);
			Arrays.fill(bias2, 1);
		}else{
			Random random = new Random();
			for(int i=0; i<hidden_len; i++){
				for(int j=0; j<input_len; j++){
					w_ih[i][j] = (random.nextDouble()-0.5)/50;
				}					
				for(int j=0; j<hidden_len; j++){
					w_hh[i][j] = (random.nextDouble()-0.5)/50;
				}
				bias1[i] = random.nextDouble();
			}
			for(int i=0; i<output_len; i++){
				for(int j=0; j<hidden_len; j++)
					w_ho[i][j] = (random.nextDouble()-0.5)/50;
				bias2[i] = random.nextDouble();
			}
			Arrays.fill(bias1, 1);
			Arrays.fill(bias2, 1);
		}
	}
	public void forward(double train_x[][], double train_y[][], int n, boolean istest){
		for(int i=0; i<unfold_size ; i++){
			if(i == 0 )
				dm.addi(dm.add(dm.mul_nm_m1(w_ih, dm.dot_(dropout, train_x[i+n])) ,
						dm.mul_nm_m1(w_hh, hidden_pre)), bias1, hidden[i]);								
			else
				dm.addi(dm.add(dm.mul_nm_m1(w_ih, dm.dot_(dropout, train_x[i+n])) ,
						dm.mul_nm_m1(w_hh, hidden[i-1])), bias1, hidden[i]);
			if(istest && drop > 0)
				dm.dot_(hidden[i], 1-drop);
		/*	for(double a : hidden[i]){
				System.out.print(a+" ");
			}System.out.println();
		*/	dm.clip(hidden[i], -50, 50);
			dm.sigmoidi(hidden[i]);			
			dm.addi(dm.mul_nm_m1(w_ho, hidden[i]), bias2, output[i]);
			dm.softmaxi(output[i]);				
		}
		
	}
	public void backward(double train_x[][], double train_y[][], int n){
		double A[] = null;
		double B[] = null;
		for(int i=unfold_size-1; i>=0; i--){
			dm.subi(output[i] , train_y[i+n], error_o[i]);
			dm.addi(bias2_, error_o[i]);
			dm.addi(w_ho_, dm.mul_n1_1m(error_o[i], hidden[i]));
			A = dm.mul_nm_m1(dm.T(w_ho), error_o[i]);
			B = dm.add(A, error_h_);
			error_h[i] = dm.dot(B, dm.dev_sigmoid_(hidden[i]));
			dm.addi(bias1_, error_h[i]);
			dm.addi(w_ih_, dm.mul_n1_1m(error_h[i], train_x[i+n]));
			if(i>0)
				dm.addi(w_hh_, dm.mul_n1_1m(error_h[i], hidden[i-1]));
			else
				dm.addi(w_hh_, dm.mul_n1_1m(error_h[i], hidden_pre));
			error_h_ = dm.mul_nm_m1(dm.T(w_hh), error_h[i]);
		}
	//	System.arraycopy(hidden[unfold_size-1], 0, hidden_pre, 0, hidden_len);
	}
	public void update_rmsprop(){
		dm.dot__(w_ih_, 1.0/mini_batch);
		dm.dot__(w_hh_, 1.0/mini_batch);
		dm.dot__(w_ho_, 1.0/mini_batch);
		dm.dot_(bias2_, 1.0/mini_batch);
		dm.dot_(bias1_, 1.0/mini_batch);

		dm.clip(bias2_, -15, 15);
		dm.clip(w_ho_, -15, 15);
		dm.clip(bias1_, -15, 15);
		dm.clip(w_ih_, -15, 15);
		dm.clip(w_hh_, -15, 15);



		dm.dot_(cache_b1, decay_rate);
		dm.addi(cache_b1, dm.dot_o(dm.pow(bias1_), 1-decay_rate));
		dm.subi(bias1, dm.dot_o(dm.dot(dm.dev_sqrt_(dm.add_A_(cache_b1, eps)), bias1_), learning_rate));
		dm.dot_(cache_b2, decay_rate);
		dm.addi(cache_b2, dm.dot_o(dm.pow(bias2_), 1-decay_rate));
		dm.subi(bias2, dm.dot_o(dm.dot(dm.dev_sqrt_(dm.add_A_(cache_b2, eps)), bias2_), learning_rate));
		dm.dot__(cache_ho, decay_rate);
		dm.addi(cache_ho, dm.dot__o(dm.pow(w_ho_), 1-decay_rate));
		dm.subi(w_ho, dm.dot__o(dm.dot__(dm.dev_sqrt_(dm.add_A_(cache_ho, eps)), w_ho_), learning_rate));
		dm.dot__(cache_hh, decay_rate);
		dm.addi(cache_hh, dm.dot__o(dm.pow(w_hh_), 1-decay_rate));
		dm.subi(w_hh, dm.dot__o(dm.dot__(dm.dev_sqrt_(dm.add_A_(cache_hh, eps)), w_hh_), learning_rate));
		dm.dot__(cache_ih, decay_rate);
		dm.addi(cache_ih, dm.dot__o(dm.pow(w_ih_), 1-decay_rate));
		dm.subi(w_ih, dm.dot__o(dm.dot__(dm.dev_sqrt_(dm.add_A_(cache_ih, eps)), w_ih_), learning_rate));

		dm.clear(w_hh_);
		dm.clear(w_ho_);
		dm.clear(w_ih_);
		Arrays.fill(bias1_, 0);
		Arrays.fill(bias2_, 0);
	}
	public void update_rmsprop_beta(){

	}
	public void update_adam(){
		dm.dot__(w_ih_, 1.0/mini_batch);
		dm.dot__(w_hh_, 1.0/mini_batch);
		dm.dot__(w_ho_, 1.0/mini_batch);
		dm.dot_(bias2_, 1.0/mini_batch);
		dm.dot_(bias1_, 1.0/mini_batch);

		dm.clip(bias2_, -15, 15);
		dm.clip(w_ho_, -15, 15);
		dm.clip(bias1_, -15, 15);
		dm.clip(w_ih_, -15, 15);
		dm.clip(w_hh_, -15, 15);

		dm.addi(dm.dot_o_(bias1_, 1-beta_adam1), dm.dot_o(cache_b1, beta_adam1), cache_b1);
		dm.addi(dm.dot_o(dm.dot_(bias1_, bias1_), 1-beta_adam2), dm.dot_o(cache_b1_v, beta_adam2), cache_b1_v);		
		dm.subi(bias1, dm.dot_o(dm.dot(dm.div_m(dm.add_A(dm.sqrt(cache_b1_v), eps)), cache_b1), 
					learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.addi(dm.dot_o_(bias2_, 1-beta_adam1), dm.dot_o(cache_b2, beta_adam1), cache_b2);
		dm.addi(dm.dot_o(dm.dot_(bias2_, bias2_), 1-beta_adam2), dm.dot_o(cache_b2_v, beta_adam2), cache_b2_v);		
		dm.subi(bias2, dm.dot_o(dm.dot(dm.div_m(dm.add_A(dm.sqrt(cache_b2_v), eps)), cache_b2), 
					learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.add_(dm.dot__o_(w_ho_, 1-beta_adam1), dm.dot__o(cache_ho, beta_adam1), cache_ho);
		dm.add_(dm.dot__o(dm.dot__(w_ho_, w_ho_), 1-beta_adam2), dm.dot__o(cache_ho_v, beta_adam2), cache_ho_v);
		dm.subi(w_ho, dm.dot__o(dm.dot__(dm.div_m(dm.add_A(dm.sqrt(cache_ho_v), eps)), cache_ho), 
					learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.add_(dm.dot__o_(w_hh_, 1-beta_adam1), dm.dot__o(cache_hh, beta_adam1), cache_hh);
		dm.add_(dm.dot__o(dm.dot__(w_hh_, w_hh_), 1-beta_adam2), dm.dot__o(cache_hh_v, beta_adam2), cache_hh_v);
		dm.subi(w_hh, dm.dot__o(dm.dot__(dm.div_m(dm.add_A(dm.sqrt(cache_hh_v), eps)), cache_hh), 
					learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));
		dm.add_(dm.dot__o_(w_ih_, 1-beta_adam1), dm.dot__o(cache_ih, beta_adam1), cache_ih);	
	
		dm.add_(dm.dot__o(dm.dot__(w_ih_, w_ih_), 1-beta_adam2), dm.dot__o(cache_ih_v, beta_adam2), cache_ih_v);	
	
		dm.subi(w_ih, dm.dot__o(dm.dot__(dm.div_m(dm.add_A(dm.sqrt(cache_ih_v), eps)), cache_ih), 
					learning_rate*Math.sqrt(1-beta_adam2)/(1-beta_adam1)));

		dm.clear(w_hh_);
		dm.clear(w_ho_);
		dm.clear(w_ih_);
		Arrays.fill(bias1_, 0);
		Arrays.fill(bias2_, 0);
	}
	public void train(double train_x[][], double train_y[][], int A[]){
		dm.dot__(train_x, 0.5);
		N = train_x.length;
		System.out.println("A length: "+A.length);
		while(epochs-- >= 0){
			System.out.println("epochs: "+(epochs+1));
			int a = 0;
			unfold_size = A[a];
			for(int i=0; i<N; ){
				dropout = dm.dropout(input_len, drop);
				for(int j=0; j<mini_batch; j++){
					forward( train_x, train_y, i, false);
					backward(train_x, train_y, i);
					i += A[a];
					if(a >= A.length - 1) break;
					unfold_size = A[++a];
				}
				if(!isadam) update_rmsprop();
				else update_adam();
			}
		}
	}
	public void predict(double test_x[][], double test_y[][], double label_[][], int B[]){
		if(N!=test_x.length)
			dm.dot__(test_x, 0.5);
		int right = 0;
		N = test_x.length;	
		int a = 0;
		unfold_size = B[a];
		Arrays.fill(dropout, 1);
		for(int i=0; i<N;){		
			for(int j=0; j<mini_batch; j++){
				forward(test_x, test_y, i, true);
				right += fit(test_y, i, label_);
				i += B[a];
				if(a>=B.length - 1) break;
				unfold_size = B[++a];
			}
		}
		System.out.println("accuracy num: "+right+" all num: "+test_x.length);
		System.out.println("accuracy rate: "+1.0*right/test_x.length);
	}
	public int fit(double test_y[][], int j, double label_[][]){
		int a,b,c=0;
		for(int i=0; i<unfold_size; i++){
			a = dm.max_i(test_y[i+j]);
			b = dm.max_i(output[i]);
			if(a == b) {
				c++;
			}
			label_[i+j] = output[i].clone();
		}
		return c;
	}
}
