
import java.util.Arrays;

public class DoubleMax {

	//add
	public void addi(double A[], double B[]){
		for(int i=0; i<A.length; i++)
			A[i] += B[i];
	}
	public double[] add(double A[], double B[]){
		for(int i=0; i<A.length; i++)
			A[i] += B[i];
		return A;
	}
	public void addi(double A[], double B[], double C[]){
		for(int i=0; i<A.length; i++)
			C[i] = A[i] + B[i];
	}
	public void addi(double A[][], double B[][]){
		for(int i=0; i<A.length; i++)
			addi(A[i],B[i]);
	}
	public void add_(double A[][], double B[][], double C[][]){
		for(int i=0; i<A.length; i++)
			addi(A[i], B[i], C[i]);
	}
	//vector or matrix add a num
	public void addi_A(double A[], double a){
		for(int i=0; i<A.length; i++)
			A[i] += a;
	}
	public double[] add_A(double A[], double a){
		for(int i=0; i<A.length; i++)
			A[i] += a;
		return A;
	}
	public double[] add_A_(double A[], double a){
		int n = A.length;
		double t[] = new double[n];
		for(int i=0; i<n; i++)
			t[i] = A[i] + a;
		return t;
	} 
	public void addi_A(double A[][], double a){
		for(int i=0; i<A.length; i++)
			addi_A(A[i], a);
	}
	public double[][] add_A(double A[][], double a){
		for(int i=0; i<A.length; i++)
			add_A(A[i], a);
		return A;		
	}
	public double[][] add_A_(double A[][], double a){
		int n = A.length;
		double t[][] = new double[n][];
		for(int i=0; i<n; i++){
			t[i] = add_A_(A[i], a);
		}
		return t;
	}
	//sub
	public void subi(double A[], double B[]){
		for(int i=0; i<A.length; i++)
			A[i] -= B[i];
	}
	public double[] sub(double A[], double B[]){
		for(int i=0; i<A.length; i++)
			A[i] -= B[i];
		return A;
	}
	public void subi(double A[], double B[], double C[]){
		for(int i=0; i<C.length; i++)
			C[i] = A[i] - B[i];
	}
	public void subi(double A[][], double B[][]){
		for(int i=0; i<A.length; i++)
			subi(A[i],B[i]);
	}
	// dot
	public void doti(double A[], double B[]){
		for(int i=0; i<A.length; i++)
			A[i] *= B[i];
	}
	public double[] dot(double A[], double B[]){
		for(int i=0; i<A.length; i++)
			A[i] *= B[i];
		return A;
	}
	public double[] dot_(double A[], double B[]){
		double t[] = new double[A.length];
		for(int i=0; i<A.length; i++)
			t[i] = A[i] * B[i];
		return t;
	}
	public void doti(double A[], double B[], double C[]){
		for(int i=0; i<A.length; i++)
			C[i] = A[i] * B[i];
	}
	public void doti(double A[][], double B[][]){
		for(int i=0; i<A.length; i++)
			doti(A[i],B[i]);
	}
	public double[][] dot__(double A[][], double B[][]){
		for(int i=0; i<A.length; i++)
			A[i] = dot(A[i], B[i]);
		return A;
	}
	// n * 1 , 1
	public void dot_(double A[], double a){
		for(int i=0; i<A.length; i++)
			A[i] = A[i] * a;
	}
	public double[] dot_o(double A[], double a){
		for(int i=0; i<A.length; i++)
			A[i] = A[i] * a;
		return A;
	}
	public double[] dot_o_(double A[], double a){
		double t[] = new double[A.length];
		for(int i=0; i<A.length; i++)
			t[i] = A[i] * a;
		return t;
	}
	// n * m , 1
	public void dot__(double A[][], double a){
		for(int i=0; i<A.length; i++)
			dot_(A[i], a);
	}
	public double[][] dot__o(double A[][], double a){
		for(int i=0; i<A.length; i++)
			dot_(A[i], a);
		return A;	
	}
	public double[][] dot__o_(double A[][], double a){
		double t[][] = new double[A.length][A[0].length];
		for(int i=0; i<A.length; i++)
			for(int j=0; j<A[0].length; j++)
				t[i][j] = A[i][j] * a;
		return t;
	}
	// mul
	// 1*n , n*1
	public double mul_1n_n1(double A[], double B[]){
		double temp = 0;
		for(int i=0; i<A.length; i++)
			temp +=A[i] * B[i];
		return temp;
	}
	// n * 1 , 1 * m
	public double[][] mul_n1_1m(double A[], double B[]){
		double t[][] = new double[A.length][B.length];
		for(int i=0; i<A.length; i++)
			for(int j=0; j<B.length; j++)
				t[i][j] = A[i] * B[j];
		return t;
	}
	// n * m , m * 1
	public double[] mul_nm_m1(double A[][], double B[]){
		double t[] = new double[A.length];
		for(int i=0; i<A.length; i++)
			t[i] = mul_1n_n1(A[i], B);
		return t;
	}
	// 1 * m , m * n
	public double[] mul_1m_mn(double A[], double B[][]){
		double t[] = new double[B[0].length];
		double tp = 0.0;
		for(int i=0; i<B[0].length; i++){
			for(int j=0; j<A.length; j++){
				tp += A[j] * B[j][i];
			}
			t[i] = tp; tp = 0.0;
		}				
		return t;
	}
	// n * m , m * n
	public double[][] muli_nn_nn(double A[][], double B[][]){
		double t[][] = new double[A.length][B[0].length];
		double a = 0;
		for(int i=0; i<A.length; i++){
			for(int j=0; j<B[0].length; j++){
				for(int k=0; k<B.length; k++){
					a = a + A[i][k] * B[k][j];
				}
				t[i][j] = a; a = 0;
			}			
		}
		return t;
	}
	
	// Transponse of a Matrix
	public void Ti(double A[][], double B[][]){
		for(int i=0; i<A.length; i++)
			for(int j=0; j<A[0].length; j++)
				B[j][i] = A[i][j];
	}
	public double[][] T(double A[][]){
		double t[][] = new double[A[0].length][A.length];
		for(int i=0; i<A.length; i++)
			for(int j=0; j<A[0].length; j++)
				t[j][i] = A[i][j];
		return t;
	}
	//functions
	//sigmoid
	public void sigmoidi(double A[]){
		for(int i=0; i<A.length; i++)
			A[i] = 1.0 / (1.0 + Math.exp(-A[i]));		
	}
	public double[] sigmoid(double A[]){
		for(int i=0; i<A.length; i++)
			A[i] = 1.0 / (1.0 + Math.exp(-A[i]));
		return A;
	}
	// dev sigmoid
	public void dev_sigmoidi(double A[]){
		for(int i=0; i<A.length; i++)
			A[i] = A[i] * (1 - A[i]);		
	}
	public double[] dev_sigmoid(double A[]){
		double t[] = A.clone();
		for(int i=0; i<t.length; i++)
			t[i] = t[i] * (1 - t[i]);
		return t;
	}
	public double[] dev_sigmoid_(double A[]){
		for(int i=0; i<A.length; i++)
			A[i] = A[i] * (1 - A[i]);
		return A;
	}
	//softmax
	public void softmaxi(double A[]){
		double t = 0.0;
		for(int i=0; i<A.length; i++)
			t += Math.exp(A[i]);
		for(int i=0; i<A.length; i++)
			A[i] = Math.exp(A[i]) / t;
	}
	public void softmaxi(double A[], int n){
		double t = 0;
		for(int i=0; i<n; i++)
			t += Math.exp(A[i]);
		for(int i=0; i<n; i++)
			A[i] = Math.exp(A[i]) / t;
	}
	public double[] softmax(double A[]){
		double t = 0.0;
		for(int i=0; i<A.length; i++)
			t += Math.exp(A[i]);
		for(int i=0; i<A.length; i++)
			A[i] = Math.exp(A[i]) / t;
		return A;
	}
	public void logistic(double A[], int n){
		double t=0;
		for(int i=0; i<n; i++){
			A[i] = 1.0/(1 + Math.exp(-A[i]));
			t += A[i];
		}
		for(int i=0; i<n; i++){
			A[i] /= t;
		}
	}
	//tanh
	public void tanhi(double A[]){
		for(int i=0; i<A.length; i++){
			A[i] = Math.tanh(A[i]);
		}
	}
	
	public double[] tanh(double A[]){
		for(int i=0; i<A.length; i++){
			A[i] = Math.tanh(A[i]);
		}
		return A;
	}
	public double[] tanh_(double A[]){
		double t[] = new double[A.length];
		for(int i=0; i<A.length; i++){
			t[i] = Math.tanh(A[i]);
		}
		return t;
	}
	public void dev_tanhi(double A[]){
		for(int i=0; i<A.length; i++){
			if(Double.isNaN(A[i]*A[i])){
				A[i] = 1;
			}else
				A[i] = 1 - A[i]*A[i];
		}
	}
	public double[] dev_tanh(double A[]){
		double t[] = new double[A.length];
		for(int i=0; i<A.length; i++)
			if(Double.isNaN(A[i]*A[i])){
				t[i] = 1;
			}else
				t[i] = 1 - A[i]*A[i];
		return t;
	}
	public double[] dev_tanh_(double A[]){
		for(int i=0; i<A.length; i++){
			if(Double.isNaN(A[i]*A[i])){
				A[i] = 1;
			}else
				A[i] = 1 - A[i]*A[i];
		}
		return A;
	}
	// max of matrix 
	public double max(double A[]){
		double t = Double.MIN_VALUE;
		for(double a : A){
			if(a>t) t = a;
		}
		return t;
	}
	// max 's index
	public int max_i(double A[]){
		double t = Double.MIN_VALUE;
		int j = 0;
		for(int i=0; i<A.length; i++){
			if(A[i] > t){
				t = A[i];
				j = i; 
			}
		}
		return j;
	}	
	public int max_i(int A[]){
		double t = Double.MIN_VALUE;
		int j = 0;
		for(int i=0; i<A.length; i++){
			if(A[i] > t){
				t = A[i];
				j = i; 
			}
		}
		return j;
	}
	// sign of a matrix
	public double[][] sgn(double A[][]){
		double t[][] = new double[A.length][A[0].length];
		for(int i=0; i<A.length; i++)
			for(int j=0; j<A[0].length; j++){
				if(A[i][j] > 0)
					t[i][j] = 1;
				else if(A[i][j] < 0)
					t[i][j] = -1;
				else
					t[i][j] = 0;
			}
		return t;
	}
	// clear, set the matrix 0
	public void clear(double A[][]){
		for(double a[] : A)
			Arrays.fill(a, 0);
	}
	public void fill(double A[][], double a){
		for(double b[] : A)
			Arrays.fill(b, a);
	}
	// matrix ^ 2
	public void powi(double A[]){
		for(int i=0; i<A.length; i++)
			A[i] *= A[i];
	}
	public double[] pow(double A[]){
		int n=A.length;
		double t[] = new double[n];
		for(int i=0; i<n; i++)
			t[i] = A[i] * A[i];
		return t;
	}
	public double[] pow_(double A[]){
		for(int i=0; i<A.length; i++)
			A[i] *= A[i];
		return A;
	}
	public void powi(double A[][]){
		for(int i=0; i<A.length; i++)
			powi(A[i]);
	}
	public double[][] pow(double A[][]){
		int n = A.length;
		double t[][] = new double[n][];
		for(int i=0; i<n; i++)
			t[i] = pow(A[i]);
		return t;
	}
	public double[][] pow_(double A[][]){
		for(int i=0; i<A.length; i++)
			powi(A[i]);
		return A;
	}
	// 1/(matrix dev sqrt)
	public void dev_sqrti(double A[]){
		for(int i=0; i<A.length; i++){
			A[i] = 1/Math.sqrt(A[i]);
		}
	}
	public double[] dev_sqrt(double A[]){
		double t[] = new double[A.length];
		for(int i=0; i<A.length; i++)
			t[i] = 1/Math.sqrt(A[i]);
		return t;
	}
	public double[] dev_sqrt_(double A[]){
		for(int i=0; i<A.length; i++){
			A[i] = 1/Math.sqrt(A[i]);
		}
		return A;
	}
	public double[][] dev_sqrt(double A[][]){
		double t[][] = new double[A.length][];
		for(int i=0; i<A.length; i++)
			t[i] = dev_sqrt(A[i]);
		return t;
	}
	public double[][] dev_sqrt_(double A[][]){
		for(int i=0; i<A.length; i++)
			A[i] = dev_sqrt_(A[i]);
		return A;
	}
	// matrix sqrt
	public void sqrti(double A[]){
		for(int i=0; i<A.length; i++)
			A[i] = Math.sqrt(A[i]);
	}
	public void sqrti(double A[][]){
		for(int i=0; i<A.length; i++)
			sqrti(A[i]);
	}
	public double[] sqrt(double A[]){
		double t[] = new double[A.length];
		for(int i=0; i<A.length; i++)
			t[i] = Math.sqrt(A[i]);
		return t;
	}
	public double[][] sqrt(double A[][]){
		double t[][] = new double[A.length][A[0].length];
		for(int i=0; i<A.length; i++)
			for(int j=0; j<A[0].length; j++)
				t[i][j] = Math.sqrt(A[i][j]);			
		return t;
	}
	public double[] sqrti_(double A[]){
		for(int i=0; i<A.length; i++)
			A[i] = Math.sqrt(A[i]);
		return A;
	}
	public double[][] sqrti_(double A[][]){
		for(int i=0; i<A.length; i++)
			sqrti(A[i]);
		return A;
	}
	// 1/matrix
	public void div_mi(double A[]){
		for(int i=0; i<A.length; i++)
			A[i] = 1/A[i];
	}
	public double[] div_m(double A[]){
		for(int i=0; i<A.length; i++){
			A[i] = 1/A[i];
		}
		return A;
	}
	public double[][] div_m(double A[][]){
		for(int i=0; i<A.length; i++)
			div_mi(A[i]);
		return A;
	}
	//dropout
	public double[] dropout(int len, double a){
		double A[] = new double[len];
		Arrays.fill(A, 1);
		int b = (int)(len*a);
		while(b-->0){
			A[(int)(len*Math.random())] = 0;
		}
		return A;
	}
	public double[][] dropout(int raw, int len, double a){
		double A[][] = new double[raw][];
		while(raw-->0){
			A[raw] = dropout(len, a);
		}
		return A;
	}
	public void clip(double A[], double a, double b){
		for(int i=0; i<A.length; i++){
			if(A[i] < a) A[i] = a;
			else if(A[i]>b) A[i] = b;
		}
	}
	public void clip(double A[][], double a, double b){
		for(int i=0; i<A.length; i++){
			clip(A[i], a, b);
		}
	}
	public void norm(double A[][]){
		int n = A.length;
		int m = A[0].length;
		double sum=0, sigma = 0;
		for(int i=0; i<m; i++){
			for(int j=0; j<n; j++){
				sum += A[j][i];
			}
			sum = sum / n;
			for(int j=0; j<n; j++){
				sigma += Math.pow(A[j][i] - sum, 2); 
			}
			sigma = Math.sqrt(sigma/n);
			for(int j=0; j<n; j++){
				A[j][i] = (A[j][i] - sum) / sigma;
			}
			sum = 0 ; sigma = 0;
		}
	}
	public void print(double A[]){
		for(double a : A){
			System.out.print(a+" ");
		}	System.out.println();
	}
	public void print(double A[], int n){
		n = n>A.length?A.length:n;
		for (int i=0; i<n; i++){
			System.out.print(A[i]+" ");
		}System.out.println();
	}
	public void print(double A[][]){
		for(double B[] : A){
			print(B);
		}
	}
	public double[] conc(double A[], double B[]){
		double t[] = new double[A.length+B.length];
		System.arraycopy(A, 0, t, 0, A.length);
		System.arraycopy(B, 0, t, A.length, B.length);
		return t;
	}
	public double cos(double A[], double B[], int n){
		double a = 0;
		double b = 0;
		double c = 0;
		for(int i=0; i<n; i++){
			a += A[i] * A[i];
			b += B[i] * B[i];
		}
		a = Math.sqrt(a);
		b = Math.sqrt(b);
		for(int i=0; i<n; i++){
			c += A[i] * B[i];
		}
		System.out.println("abc: "+a+" "+b+" "+c);
		return c/(a*b);
	}
}
