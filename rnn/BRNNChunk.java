
public class BRNNChunk {

	public static void main(String[] args) {
		PropReader prop = new PropReader();
		String conf = args[0];
		prop.read_conf(conf);
		String path_tr_x = prop.get_trainx();
		String path_tr_y = prop.get_trainy();
		String path_te_x = prop.get_testx();
		String path_te_y = prop.get_testy();
		RNNChunkCRF test = new RNNChunkCRF();
		test.read_data(path_tr_x, path_tr_y, false, prop.get_input(), prop.get_output());
		test.read_data(path_te_x, path_te_y, true, prop.get_input(), prop.get_output());
		test.trs(prop.get_trs());
		test.unfold(prop.get_unfold_A(), true);
		test.unfold(prop.get_unfold_B(), false);
		double trs2[][] = test.get_trs2(test.train_y, prop.get_output());
		if(prop.get_crf()){
			BRNNCRF rnn = new BRNNCRF(
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
			double A[][] = new double[test.train_y.length][prop.get_output()];
			double B[][] = new double[test.test_y.length][prop.get_output()];
			int lab1[] = new int[test.train_y.length];
			int lab2[] = new int[test.test_y.length];
			rnn.net(prop.get_train_methods());
			rnn.init(prop.get_init_methods());
			rnn.train(test.train_x, test.train_y, trs2, test.A);
			System.out.println("pred on train:");
			Valuate valuate = new Valuate();
			rnn.predict(test.train_x, test.train_y, A, lab1, test.A);
			System.out.println("val on train:");
			valuate.val(test.train_y, A);
			valuate.val(test.train_y, lab1);
			System.out.println("pred on test:");
			rnn.predict(test.test_x, test.test_y, B, lab2, test.B);
			System.out.println("val on test:");
			valuate.val(test.test_y, B);
			valuate.val(test.test_y, lab2);

		}else{
			BRNNFould rnn = new BRNNFould(
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

			double A[][] = new double[test.train_y.length][prop.get_output()];
			double B[][] = new double[test.test_y.length][prop.get_output()];
			rnn.net(prop.get_train_methods());
			rnn.init(prop.get_init_methods());
			rnn.train(test.train_x, test.train_y, test.A);
			System.out.println("pred on train:");
			Valuate valuate = new Valuate();
			rnn.predict(test.train_x, test.train_y, A, test.A);
			System.out.println("val on train:");
			valuate.val(test.train_y, A);
			System.out.println("pred on test:");
			rnn.predict(test.test_x, test.test_y, B, test.B);
			System.out.println("val on test:");
			valuate.val(test.test_y, B);
		}
	}
}
