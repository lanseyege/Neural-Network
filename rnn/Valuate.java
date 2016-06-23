
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Arrays;

public class Valuate {

	String[] labs = {"B-NP","I-NP","E-NP",
            "B-VP","I-VP","E-VP",
            "B-QP","I-QP","E-QP",
            "B-PP","I-PP","E-PP",
            "B-ADVP","I-ADVP","E-ADVP",
            "B-DNP","I-DNP","E-DNP",
            "B-LCP","I-LCP","E-LCP",
            "B-DP","I-DP","E-DP",
            "B-ADJP","I-ADJP","E-ADJP",
            "B-DVP","I-DVP","E-DVP",
            "B-LST","I-LST","E-LST",
            "B-CLP","I-CLP","E-CLP",/*
            "S-NP","S-VP","S-QP",
            "S-PP","S-ADVP","S-DNP",
            "S-LCP","S-DP","S-ADJP",
            "S-DVP","S-LST","S-CLP",*/
            "O"};
	int classes = 13;
	int tlabel[] = new int[classes];
	int plabel[] = new int[classes];
	int rlabel[] = new int[classes];
	HashMap<String, String> truelist = new HashMap<String, String>();
	HashMap<String, String> predlist = new HashMap<String, String>();

	public void val(double C[][], double B[][], double Y[][], int unfold_size, int output_len){
		DoubleMax dm = new DoubleMax();
		int a,b;
		int label_[] = new int[B.length];
		for(int i=0; i<B.length; i++ ){
//			a = dm.max_i(C[i]);
			b = dm.max_i(B[i]);
			label_[i] = b;
		}
		val(C,label_);
		label_ = viterbi(B, Y, unfold_size, output_len);
		System.out.println("viterbi...");
		val(C, label_);
	}
	public void val(double C[][], double B[][]){
		DoubleMax dm = new DoubleMax();
		int a,b;
		int label_[] = new int[B.length];
		for(int i=0; i<B.length; i++ ){
			b = dm.max_i(B[i]);
			label_[i] = b;
		}
		val(C,label_);
	}
	public int[] viterbi(double B[][], double Y[][], int unfold_size, int output_len){
		int n = B.length;
		DoubleMax dm = new DoubleMax();
		double alpha[] = new double[output_len];
		double pre_alpha[] = new double[output_len];
		int path[][] = new int[n][output_len];
		int label[] = new int[n];
		for(int i=0; i<n;){
			for(int j=0; j<output_len; j++){
				pre_alpha[j] = B[i][j];
			}
			int a=0;
			for(int j=i; j<i+unfold_size && j<n; j++){
				for(int k=0; k<output_len; k++){
					double max_score = Double.MIN_VALUE;
					double score;
					int index;
					for(int t=0; t<output_len; t++){
						score=pre_alpha[t]+Y[k][t]+B[j][k]; 
						if(score > max_score){
							max_score = score;
							path[j][k] = t;
						}
					}
					alpha[k] = max_score;
				}
				pre_alpha = alpha.clone();
				Arrays.fill(alpha,0);
				a = j;
			}
			label[a] = dm.max_i(path[a]);
			for(int j=a-1; j>=i; j--){
				label[j] = path[j+1][label[j+1]];
			}
			i+=unfold_size;
		}
		return label;
	}
	public void val(int A[], int B[]){
		int n = A.length;
		int a=0,b=0,c=0;
		
		for(int i=0; i<n; ){
			a = A[i]/3;
			b = A[i]%3;
			if(b==0){
				tlabel[a]++;
				while(++i<n && A[i]%3 != 0);
			}
		}
		for(int i=0; i<n; ){
			a = B[i]/3;
			b = B[i]%3;
			c = B[i];
			if(b==0){
				while(++i<n){
					if(B[i]%3 == 0){
						if(B[i-1]%3 == 0)
							plabel[a]++;
						break;
					}
					if(B[i]/3 != a) break;
					if(B[i] == c+1){
						
					}
					if(B[i] == c+2){
						plabel[a]++;
						break;
					}					
				}
			}else{
				i++;
			}
		}
		boolean flag = true;
		for(int i=0; i<n; ){
			a = A[i]/3;
			b = A[i]%3;
			if(b==0){
				//tlabel[a]++;
				if(A[i] != B[i]){
					flag = false;
				}
				while(++i<n ){
					if(A[i]%3 != 0){
						if(A[i] != B[i]){
							flag = false;
						}
					}else{
						if(B[i]%3 !=0 )
							flag = false;
						break;
					}
				}
				if(flag)
					rlabel[a] ++;
			}
			flag = true;
		}
		pred();

	}
	public void val(double C[][], int B[]){
		int A[] = new int[C.length];
		for(int i=0; i<A.length; i++){
			for(int j=0; j<C[i].length; j++){
				if(C[i][j] > 0.5){
					A[i] = j;
					break;
				}
			}
		}
		val(A, B);
	}
	public void pred(){
		System.out.print("labels\ttlabel\tplabel\trlabel\trecall\t\tpresion\t\tF1\t\n");
		int a=0,b=0,c=0;
		for(int i=0; i<classes; i++){
			System.out.print(labs[i*3]+"\t");
			System.out.print(tlabel[i]+"\t");
			System.out.print(plabel[i]+"\t");
			System.out.print(rlabel[i]+"\t");
			System.out.print(1.0*rlabel[i]/plabel[i]+"\t");
			System.out.print(1.0*rlabel[i]/tlabel[i]+"\t");
			System.out.print(2.0*rlabel[i]/(plabel[i]+tlabel[i])+"\t\n");
			a+=tlabel[i];
			b+=plabel[i];
			c+=rlabel[i];
		}
		System.out.print("\t"+a+"\t"+b+"\t"+c+"\t"+(1.0*c/b)+"\t"+(1.0*c/a)+"\t"+(2.0*c/(a+b))+"\t\n");
		Arrays.fill(tlabel, 0);
		Arrays.fill(plabel, 0);
		Arrays.fill(rlabel, 0);
	}
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
	}

}
