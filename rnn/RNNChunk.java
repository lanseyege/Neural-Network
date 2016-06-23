
import java.util.ArrayList;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;

public class RNNChunk {

	double train_x[][] = null;//new double[435280][100];
	double train_y[][] = null;//new double[435280][100];
	double test_x[][] = null;//new double[53864][100];
	double test_y[][] = null;//new double[53864][100];
	double trs[][] = null; 
	int A[] = null;
	int B[] = null;
	public void read_data(String path_x, String path_y, boolean istest, int input, int output){
		if(!istest){
			trs = new double[output][output];
			train_x = new double[435280][input];
			train_y = new double[435280][output];
		}else{
			test_x = new double[53864][input];
			test_y = new double[53864][output];
		}
		BufferedReader buff = null;
		try{
			buff = new BufferedReader(new InputStreamReader(new FileInputStream(path_x), "UTF-8"));
			String line = null;
			String temp[] = null;
			int a=0;
			while((line = buff.readLine()) != null){
				temp = line.split(" ");
				if(!istest){
					for(int i=0; i<temp.length; i++){
						train_x[a][i] = Double.parseDouble(temp[i]);
					}
				}else{
					for(int i=0; i<temp.length; i++)
						test_x[a][i] = Double.parseDouble(temp[i]);
				}
				a++;
			}
			a = 0;
			buff.close();
			buff = new BufferedReader(new InputStreamReader(new FileInputStream(path_y), "UTF-8"));
			while((line = buff.readLine()) != null ){
				temp = line.split(" ");
				if(!istest){
					for(int i=0; i<temp.length; i++)
						train_y[a][i] = Double.parseDouble(temp[i]);
				}else
					for(int i=0; i<temp.length; i++)
						test_y[a][i] = Double.parseDouble(temp[i]);
				a++;
			}
			buff.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		
	}
	public void trs(String path){
		
		BufferedReader buff = null;
		try{
			buff = new BufferedReader(new InputStreamReader(new FileInputStream(path), "UTF-8"));
			String line = null;
			String temp[] = null;
			int a=0;
			while((line = buff.readLine()) != null){
				temp = line.split(" ");
				for(int i=0; i<temp.length; i++){
					trs[a][i] = Integer.parseInt(temp[i]);
				}
				a++;
			}
			buff.close();
		}catch(Exception e){
			e.printStackTrace();
		}

	}	
	public void unfold(String path, boolean flag){
		
		BufferedReader buff = null;
		ArrayList<Integer> list = new ArrayList<Integer>();
		try{
			buff = new BufferedReader(new InputStreamReader(new FileInputStream(path), "UTF-8"));
			String line = null;
			String temp[] = null;
			while((line = buff.readLine()) != null){
				list.add(Integer.parseInt(line));
				temp = line.split(" ");
			}
			buff.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		if(flag){
			A = new int[list.size()];
			for(int i=0; i<list.size(); i++) A[i] = list.get(i);
		}else{
			B = new int[list.size()];
			for(int i=0; i<list.size(); i++) B[i] = list.get(i);
		}
	}
	public double[][] get_trs2(double label[][], int output_len){
		double t[][] = new double[output_len][output_len];
		for(int i=0; i<output_len; i++)
			for(int j=0; j<output_len; j++)
				t[i][j] =  1;
		int ta[] = new int[label.length];
		int a=0,b=0;
		for(int i=0; i<label.length; i++){
			for(int j=0; j<label[0].length; j++){
				if(label[i][j]>0.5){
					a=j;
					break;
				}
			}
			ta[i] = a;
		}
		for(int i=1; i<label.length; i++){
			t[ta[i]][ta[i-1]] += 1;
		}
		for(int i=0; i<output_len; i++){
			double sum = 0;
			for(int j=0; j<output_len; j++){
				sum += t[i][j];
			}
			for(int j=0; j<output_len; j++){
				t[i][j] = Math.log(t[i][j]/sum);
			}
		}
		return t;
	}
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		PropReader prop = new PropReader();
		String conf = args[0];
		prop.read_conf(conf);
		String path_tr_x = prop.get_trainx();
		String path_tr_y = prop.get_trainy();
		String path_te_x = prop.get_testx();
		String path_te_y = prop.get_testy();
		RNNChunk test = new RNNChunk();
		test.read_data(path_tr_x, path_tr_y, false, prop.get_input(), prop.get_output());
		test.read_data(path_te_x, path_te_y, true, prop.get_input(), prop.get_output());
		test.trs(prop.get_trs());
		test.unfold(prop.get_unfold_A(), true);
		test.unfold(prop.get_unfold_B(),false);
		double trs2[][] = test.get_trs2(test.train_y, prop.get_output());
		if(prop.get_crf()){
			RNNCRFould rnn = new RNNCRFould(
					prop.get_input(),
					prop.get_hidden(),
					prop.get_output(),
					prop.get_epochs(),
					prop.get_mini(),
					prop.get_unfold(),
					prop.get_decay(),
					prop.get_lrate(),
					prop.get_eps(),
					prop.get_beta1(),
					prop.get_beta2(),
					prop.get_drop()
					);	
			rnn.net(prop.get_train_methods());
			rnn.init(prop.get_init_methods());
			rnn.train(test.train_x, test.train_y, trs2, test.A);
			System.out.println("pred on train:");
			double A[][] = new double[test.train_y.length][prop.get_output()];
			Valuate valuate = new Valuate();
			int lab1[] = new int[test.train_y.length];
			rnn.predict(test.train_x, test.train_y, A, lab1, test.A);
			System.out.println("val on train:");
			valuate.val(test.train_y, A);
			valuate.val(test.train_y, lab1);
			double B[][] = new double[test.test_y.length][prop.get_output()];
			System.out.println("pred on test:");
			int lab2[] = new int[test.test_y.length];
			rnn.predict(test.test_x, test.test_y, B, lab2, test.B);
			System.out.println("val on test:");
			valuate.val(test.test_y, B);
			valuate.val(test.test_y, lab2);
		}else{
			RNNFould rnn = new RNNFould(
					prop.get_input(),
					prop.get_hidden(),
					prop.get_output(),
					prop.get_epochs(),
					prop.get_mini(),
					prop.get_unfold(),
					prop.get_decay(),
					prop.get_lrate(),
					prop.get_eps(),
					prop.get_beta1(),
					prop.get_beta2(),
					prop.get_drop()
					);	
			rnn.net(prop.get_train_methods());
			rnn.init(prop.get_init_methods());
			rnn.train(test.train_x, test.train_y, test.A);
			System.out.println("pred on train:");
			double A[][] = new double[test.train_y.length][prop.get_output()];
			Valuate valuate = new Valuate();
			rnn.predict(test.train_x, test.train_y, A, test.A);
			System.out.println("val on train:");
			valuate.val(test.train_y, A);
			double B[][] = new double[test.test_y.length][prop.get_output()];
			System.out.println("pred on test:");
			rnn.predict(test.test_x, test.test_y, B, test.B);
			System.out.println("val on test:");
			valuate.val(test.test_y, B);
		}
	}
}
