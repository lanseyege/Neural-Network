
import java.util.Arrays;
import java.util.Random;

public class RNNCRFould {
	double w_ih[][] = null;
	double w_ih_[][] = null;
	double w_hh[][] = null;
	double w_hh_[][] = null;
	double w_ho[][] = null;
	double w_ho_[][] = null;
	double input[] = null;
	double hidden[][] = null;
	double output[][] = null;
	double output_[][] = null;
	double crf_output[][] = null;
	double hidden_pre[] = null;
	double bias1[] = null;
	double bias1_[] = null;
	double bias2[] = null;
	double bias2_[] = null;
	double error_o[][] = null;
	double error_h[][] = null;
	double error_h_[] = null;
	double error_trs[][] = null;//new double[output_len][output_len];
	double error_o_[][] = null;//new double[error_o.length][error_o[0].length];

	double cache_b1[] = null;
	double cache_b2[] = null;
	double cache_ih[][] = null;
	double cache_hh[][] = null;
	double cache_ho[][] = null;
	double cache_trs[][] = null;
	double decay_rate;
	double learning_rate;
	double eps;
	double beta;//rmsprop with momentum (Nesterov accelerated gradiet)
	double drop;// drop out

	double beta_adam1;
	double beta_adam2;
	double cache_b1_v[] = null;
	double cache_b2_v[] = null;
	double cache_ih_v[][] = null;
	double cache_hh_v[][] = null;
	double cache_ho_v[][] = null;
	double cache_trs_v[][] = null;
	double dropout[] = null;
	double trs[][] = null;

	int N ;
	int input_len;
	int hidden_len;
	int output_len;
	int epochs;
	int unfold_size;
	int unfold_count = 120;
	int mini_batch;
	boolean isadam;
	DoubleMax dm = null;
	public RNNCRFould(int input_len, int hidden_len, int output_len, 
			int epochs, int unfold_size, int mini_batch,
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
		this.drop = drop;
		this.beta_adam1 = beta_adam1;
		this.beta_adam2 = beta_adam2;
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
		error_o_ = new double[unfold_count][output_len];
		error_h = new double[unfold_count][hidden_len];
		error_h_ = new double[hidden_len];
		hidden = new double[unfold_count][hidden_len];
		output = new double[unfold_count][output_len];
		output_ = new double[unfold_count][output_len];
		crf_output = new double[unfold_count][output_len];
		hidden_pre = new double[hidden_len];
		error_trs = new double[output_len][output_len];
		dm = new DoubleMax();

		cache_b1 = new double[hidden_len];
		cache_b2 = new double[output_len];
		cache_ih = new double[hidden_len][input_len];
		cache_hh = new double[hidden_len][hidden_len];
		cache_ho = new double[output_len][hidden_len];
		cache_trs = new double[output_len][output_len];
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
			cache_trs_v = new double[output_len][output_len];
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
					w_ih[i][j] = random.nextDouble()/100;
				}					
				for(int j=0; j<hidden_len; j++){
					w_hh[i][j] = random.nextDouble()/100;
				}
			}
			for(int i=0; i<output_len; i++)
				for(int j=0; j<hidden_len; j++)
					w_ho[i][j] = random.nextDouble()/100;
			Arrays.fill(bias1, 1);
			Arrays.fill(bias2, 1);
		}
	}
	public void forward(double train_x[][], double train_y[][], int n){
		for(int i=0; i<unfold_size && i+n<N; i++){
			//dm.dot_(dropout, train_x[i+n]);//dropout

			if(i == 0 )
				dm.addi(dm.add(dm.mul_nm_m1(w_ih, dm.dot_(dropout, train_x[i+n])) ,
							dm.mul_nm_m1(w_hh, hidden_pre)), bias1, hidden[i]);								
			else
				dm.addi(dm.add(dm.mul_nm_m1(w_ih, dm.dot_(dropout, train_x[i+n])) ,
							dm.mul_nm_m1(w_hh, hidden[i-1])), bias1, hidden[i]);
			/*	for(double A[] :hidden){
				for(double a:A){
				System.out.print(a+" h");
				}System.out.println();
				}*/

			//dm.sigmoidi(hidden[i]);		
			dm.tanhi(hidden[i]);
			dm.addi(dm.mul_nm_m1(w_ho, hidden[i]), bias2, output_[i]);
			/*	for(double A[] :output_){
				for(double a:A){
				System.out.print(a+" oo ");
				}System.out.println();
				}*/
			System.arraycopy(output_[i], 0, output[i], 0, output_len);
			dm.softmaxi(output[i]);
			/*for(int k=0; k<output_len; k++){
			  System.out.print(output_[i][k]+" ");
			}
			System.out.println();*/
		}
		//System.out.println();
	}
	public double logsumexp(double x, double y, boolean flg){
		if(flg) return y;
		double vmin = Math.min(x, y);
		double vmax = Math.max(x, y);
		if(vmax > vmin + 13) return vmax;
		else return vmax + Math.log(Math.exp(vmin - vmax) + 1.0);
	}
	public void crf2(double train_y[][], int n){
		double[][] alphaSet = new double[unfold_size][];
		double[][] betaSet = new double[unfold_size][];
		for(int i=0; i<unfold_size; i++){
			alphaSet[i] = new double[output_len];
			for(int j=0; j<output_len; j++){
				double dscore0 = 0;
				if(i > 0){
					for(int k=0; k<output_len; k++){
						double fbgm = trs[j][k];
						double finit = alphaSet[i-1][k];
						double ftmp = fbgm + finit;
						dscore0 = logsumexp(dscore0, ftmp, k==0);							
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
			Z_ =  logsumexp(Z_, betaSet_0[i], i==0);
		}
		//	Z_ = Math.log(Z_);
		dm.clear(crf_output);
		for(int i=0; i<unfold_size; i++){
			for(int j=0; j<output_len; j++){
				crf_output[i][j]=Math.exp(alphaSet[i][j]+betaSet[i][j]-output_[i][j]-Z_);
			}	
			System.arraycopy(crf_output[i], 0, output[i], 0, output_len);
			dm.softmaxi(output[i]);		
		}

	}
	public void update_bigram(double train_y[][], int n){
		double trs_delta[][] = new double[output_len][output_len];
		for(int timeat=1; timeat<unfold_size; timeat++){
			double[] crf_output_t = crf_output[timeat];
			double[] crf_output_pt = crf_output[timeat-1];
			for(int i=0; i<output_len; i++){
				double crf_output_ti = crf_output_t[i];
				double[] trs_i = trs[i];
				double[] trs_delta_i = trs_delta[i];
				int j=0;
				while(j < output_len){
					trs_delta_i[j] -= trs_i[j]*crf_output_ti*crf_output_pt[j];
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
		double A[] = null;
		double B[] = null;
		int j = unfold_size-1<N-n-1?unfold_size-1:N-n-1;
		for(int i=j; i>=0; i--){
			dm.subi(crf_output[i] , train_y[i+n], error_o[i]);
			dm.addi(bias2_, error_o[i]);
			dm.addi(w_ho_, dm.mul_n1_1m(error_o[i], hidden[i]));
			A = dm.mul_nm_m1(dm.T(w_ho), error_o[i]);
			B = dm.add(A, error_h_);
			error_h[i] = dm.dot(B, dm.dev_tanh_(hidden[i]));//dm.dev_tanh(A)
			dm.addi(bias1_, error_h[i]);
			dm.addi(w_ih_, dm.mul_n1_1m(error_h[i], dm.dot_(dropout, train_x[i+n])));//dropout
			if(i>0)
				dm.addi(w_hh_, dm.mul_n1_1m(error_h[i], hidden[i-1]));
			else
				dm.addi(w_hh_, dm.mul_n1_1m(error_h[i], hidden_pre));
			error_h_ = dm.mul_nm_m1(dm.T(w_hh), error_h[i]);

		}
		Arrays.fill(error_h_, 0);
		System.arraycopy(hidden[unfold_size-1<N-n-1?unfold_size-1:N-n-1], 0, hidden_pre, 0, hidden_len);
	}
	public void update_rmsprop(){
		dm.dot__(w_ih_, 1.0/mini_batch);
		dm.dot__(w_hh_, 1.0/mini_batch);
		dm.dot__(w_ho_, 1.0/mini_batch);
		dm.dot__(error_trs,   1.0/mini_batch);
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
		/*
		   dm.dot__(cache_trs, decay_rate);
		   dm.addi(cache_trs, dm.dot__o(dm.pow(error_trs), 1-decay_rate));
		   dm.subi(trs, dm.dot__o(dm.dot__(dm.dev_sqrt_(dm.add_A_(cache_trs, eps)), error_trs), learning_rate));*/

		dm.clear(w_hh_);
		dm.clear(w_ho_);
		dm.clear(w_ih_);
		//dm.clear(error_trs);
		Arrays.fill(bias1_, 0);
		Arrays.fill(bias2_, 0);
	}
	public void update_rmsprop_beta(){

	}
	public void update_adam(){
		dm.dot__(w_ih_, 1.0/mini_batch);
		dm.dot__(w_hh_, 1.0/mini_batch);
		dm.dot__(w_ho_, 1.0/mini_batch);
		//dm.dot__(error_trs, 1.0/mini_batch);
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
		dm.clear(error_trs);
		Arrays.fill(bias1_, 0);
		Arrays.fill(bias2_, 0);
	}
	public void train(double train_x[][], double train_y[][], double trs[][], int A[]){
		dm.dot__(train_x, 0.5);
		this.trs = trs;
		N = train_x.length;
		while(epochs-- >= 0){
			System.out.println("epochs: "+(epochs+1));
			int a = 0;
			unfold_size = A[0];
			for(int i=0; i<N; ){		
				dropout = dm.dropout(input_len, drop);
				for(int j=0; j<mini_batch; j++){
					forward(train_x, train_y, i);
					crf2(train_y, i);
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
		}
	}
	public void predict(double test_x[][], double test_y[][], double label_[][], int label[], int B[]){
		if(N != test_x.length)
			dm.dot__(test_x, 0.5);
		//	if(istest)
		//		dm.norm(test_x);
		all = 0;
		int right = 0;
		N = test_x.length;
		int a = 0;
		unfold_size = B[a];
		Arrays.fill(dropout, 1);
		for(int i=0; i<N; ){				
			for(int j=0; j<mini_batch; j++){
				forward(test_x, test_y, i);
				crf2(test_y, i);
				right += fit(test_y, i, label, label_);
				i += B[a];
				if(a >= B.length - 1) break;
				unfold_size = B[++a];
			}
		}
		System.out.println("accuracy num: "+right+" all num: "+test_x.length);
		System.out.println("accuracy rate: "+1.0*right/test_x.length);
		System.out.println("all : "+all);
	}double all = 0;
	public int fit(double test_y[][], int j, int label[], double label_[][]){
		int a,b,c=0;
		int A[] = viterbi(output_);
		System.arraycopy(A, 0, label, j, unfold_size);
		
		for(int i=0; i<unfold_size && i+j<N; i++){
			a = dm.max_i(test_y[i+j]);
			b = dm.max_i(output[i]);
			if(a == b) c++;
			if(a == A[i]) all++;
			label_[i+j] = output[i].clone();
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
		//System.out.println(next);
		for(int t = unfold_size - 2; t>=0; t--){
			tag[t] = vpath[t+1][next];
			next = tag[t];
		}
		return tag;
	}
}
