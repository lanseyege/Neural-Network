
import java.io.FileInputStream;
import java.util.Properties;

public class PropReader {

	private int input_len;
	private int input_len_b;
	private int hidden_len;
	private int hidden_len_b;
	private int output_len;
	private int epochs;
	private int mini_batch;
	private int unfold_size;
	private double learning_rate;
	private double decay_rate;
	private double eps;
	private double beta1;
	private double beta2;
	private double dropout;
	private double momentem;
	private String train_x;
	private String train_x_b;
	private String train_y;
	private String test_x;
	private String test_x_b;
	private String test_y;
	private String trs;
	private String unfold_A;
	private String unfold_B;
	private String vocab;
	private String pos;
	private String train;
	private String test;
	private boolean crf;
	private boolean isadam;
	private boolean isgass;
	public void read_conf(String path){
		String input_ = "input";
		String input_b_ = "input_b";
		String hidden_ = "hidden";
		String hidden_b_ = "hidden_b";
		String output_ = "output";
		String epochs_ = "epochs";
		String mini_ = "mini_batch";
		String unfold_ = "unfold_size";
		String learning_ = "learning_rate";
		String decay_ = "decay_rate";
		String eps_ = "eps";
		String beta1_ = "beta1";
		String beta2_ = "beta2";
		String drop_ = "dropout";
		String train_x_ = "train_x";
		String train_x_b_ = "train_x_b";
		String train_y_ = "train_y";
		String test_x_ = "test_x";
		String test_x_b_ = "test_x_b";
		String test_y_ = "test_y";
		String hit = "hit";
		String adam = "adam";
		String gass = "gass";
		String trs_ = "trs";
		String crf_ = "crf";
		String unfold_A_ = "unfold_A";
		String unfold_B_ = "unfold_B";
		String momentem_ = "momentem";
		String train_ = "train";
		String test_ = "test";
		String vocab_ = "vocab";
		String pos_ = "pos";
		Properties prop = new Properties();

		try{
			FileInputStream fileInput = new FileInputStream(path);
			prop.load(fileInput);
			if(prop.containsKey(hit)){
				System.out.println(prop.getProperty(hit));
			}
			if(prop.containsKey(train_x_)){
				train_x = prop.getProperty(train_x_);
				System.out.println("train_x: "+train_x);
			}	
			if(prop.containsKey(train_x_b_)){
				train_x_b = prop.getProperty(train_x_b_);
				System.out.println("train_x_b: "+train_x_b);
			}
			if(prop.containsKey(train_y_)){
				train_y = prop.getProperty(train_y_);
				System.out.println("train_y: "+train_y);
			}
			if(prop.containsKey(test_x_)){
				test_x = prop.getProperty(test_x_);
				System.out.println("test_x: "+test_x);
			}	
			if(prop.containsKey(test_x_b_)){
				test_x_b = prop.getProperty(test_x_b_);
				System.out.println("test_x_b: "+test_x_b);
			}
			if(prop.containsKey(test_y_)){
				test_y = prop.getProperty(test_y_);
				System.out.println("test_y: "+test_y);
			}	
			if(prop.containsKey(trs_)){
				trs = prop.getProperty(trs_);
				System.out.println("trs: "+trs);
			}
			if(prop.containsKey(input_)){
				input_len = Integer.parseInt(prop.getProperty(input_));
				System.out.println("input_len: "+input_len);
			}	
			if(prop.containsKey(input_b_)){
				input_len_b = Integer.parseInt(prop.getProperty(input_b_));
				System.out.println("input_len_b: "+input_len_b);
			}
			if(prop.containsKey(hidden_)){
				hidden_len = Integer.parseInt(prop.getProperty(hidden_));
				System.out.println("hidden_len: "+hidden_len);
			}	
			if(prop.containsKey(hidden_b_)){
				hidden_len_b = Integer.parseInt(prop.getProperty(hidden_b_));
				System.out.println("hidden_len_b: "+hidden_len_b);
			}
			if(prop.containsKey(output_)){
				output_len = Integer.parseInt(prop.getProperty(output_));
				System.out.println("output_len: "+output_len);
			}
			if(prop.containsKey(epochs_)){
				epochs = Integer.parseInt(prop.getProperty(epochs_));
				System.out.println("epochs: "+epochs);
			}
			if(prop.containsKey(mini_)){
				mini_batch = Integer.parseInt(prop.getProperty(mini_));
				System.out.println("mini_batch: "+mini_batch);
			}
			if(prop.containsKey(unfold_)){
				unfold_size = Integer.parseInt(prop.getProperty(unfold_));
				System.out.println("unfold_: "+unfold_size);
			}
			if(prop.containsKey(learning_)){
				learning_rate = Double.parseDouble(prop.getProperty(learning_));
				System.out.println("learning_rate: "+learning_rate);
			}
			if(prop.containsKey(decay_)){
				decay_rate = Double.parseDouble(prop.getProperty(decay_));
				System.out.println("decay_rate: "+decay_rate);
			}
			if(prop.containsKey(eps_)){
				eps = Double.parseDouble(prop.getProperty(eps_));
				System.out.println("eps: "+eps);
			}
			if(prop.containsKey(beta1_)){
				beta1 = Double.parseDouble(prop.getProperty(beta1_));
				System.out.println("beta1: "+beta1);
			}
			if(prop.containsKey(beta2_)){
				beta2 = Double.parseDouble(prop.getProperty(beta2_));
				System.out.println("beta2: "+beta2);
			}
			if(prop.containsKey(drop_)){
				dropout = Double.parseDouble(prop.getProperty(drop_));
				System.out.println("dropout: "+dropout);
			}
			if(prop.containsKey(adam)){
				isadam = Boolean.parseBoolean(prop.getProperty(adam));
				System.out.println("train methods: (true is adam, false is rmsprop) "+isadam);
			}
			if(prop.containsKey(gass)){
				isgass = Boolean.parseBoolean(prop.getProperty(gass));
				System.out.println("init methods: (true is gass, false is uniform) " +isgass);
			}	
			if(prop.containsKey(crf_)){
				crf = Boolean.parseBoolean(prop.getProperty(crf_));
				System.out.println("crf methods: (true use crf, false dont use crf) " +crf);
			}	
			if(prop.containsKey(unfold_A_)){
				unfold_A = prop.getProperty(unfold_A_);
				System.out.println("unfold_A path: "+unfold_A);
			}	
			if(prop.containsKey(unfold_B_)){
				unfold_B = prop.getProperty(unfold_B_);
				System.out.println("unfold_B path: "+unfold_B);
			}	
			if(prop.containsKey(momentem_)){
				momentem = Double.parseDouble(prop.getProperty(momentem_));
				System.out.println("momentem: "+momentem);
			}	
			if(prop.containsKey(vocab_)){
				vocab = prop.getProperty(vocab_);
				System.out.println("vocab: "+vocab);
			}	
			if(prop.containsKey(pos_)){
				pos = prop.getProperty(pos_);
				System.out.println("pos: "+pos);
			}	
			if(prop.containsKey(train_)){
				train = prop.getProperty(train_);
				System.out.println("train: "+train);
			}	
			if(prop.containsKey(test_)){
				test = prop.getProperty(test_);
				System.out.println("test: "+test);
			}
		}catch(Exception e){
			e.printStackTrace();
		}
	}
	public String get_train(){
		return train;
	}
	public String get_test(){
		return test;
	}
	public String get_v(){
		return vocab;
	}
	public String get_p(){
		return pos;
	}
	public double get_momentem(){
		return momentem;
	}
	public String get_unfold_A(){
		return unfold_A;
	}	
	public String get_unfold_B(){
		return unfold_B;
	}
	public boolean get_crf(){
		return crf;
	}
	public boolean get_train_methods(){
		return isadam;
	}
	public boolean get_init_methods(){
		return isgass;
	}
	public String get_trainx(){
		return train_x;
	}
	public String get_trs(){
		return trs;
	}
	public String get_trainx_b(){
		return train_x_b;
	}
	public String get_trainy(){
		return train_y;
	}
	public String get_testx(){
		return test_x;
	}
	public String get_testx_b(){
		return test_x_b;
	}
	public String get_testy(){
		return test_y;
	}
	public int get_input(){
		return input_len;
	}
	public int get_input_b(){
		return input_len_b;
	}
	public int get_hidden(){
		return hidden_len;
	}
	public int get_hidden_b(){
		return hidden_len_b;
	}
	public int get_output(){
		return output_len;
	}
	public int get_epochs(){
		return epochs;
	}
	public int get_mini(){
		return mini_batch;
	}
	public int get_unfold(){
		return unfold_size;
	}
	public double get_lrate(){
		return learning_rate;
	}
	public double get_decay(){
		return decay_rate;
	}
	public double get_eps(){
		return eps;
	}
	public double get_beta1(){
		return beta1;
	}
	public double get_beta2(){
		return beta2;
	}
	public double get_drop(){
		return dropout;
	}
}

