#include <iostream>
#include <cstdio>
#include <cstdlib>

int NT = 128;

__global__ void householder_kernel(double* A, double* v, const int k, const int n){
    extern __shared__ double smem[];
    const int start = k + 1;
    double* x = A + k * n;
    const double x0 = x[start];

    //norma al cuadrado de la columna
    double temp = 0.0;
    for(int t = start + threadIdx.x; t < n; t += blockDim.x){
        temp += x[t] * x[t];
    }
    smem[threadIdx.x] = temp;
    __syncthreads();

    for(int i = blockDim.x / 2; i > 0; i >>= 1){
        if(threadIdx.x < i){
            smem[threadIdx.x] += smem[threadIdx.x + i];
        }
        __syncthreads();
    }

    const double norma = sqrt(smem[0]);
    const double alpha = x0 < 0 ? norma : -norma;

    //||v||^2 = 2*norma*(norma + |x0|), no hace falta un segundo producto punto
    const double norma_v = sqrt(2.0 * norma * (norma + fabs(x0)));
    const double inv = norma_v > 0 ? 1.0 / norma_v : 0.0;

    for(int t = start + threadIdx.x; t < n; t += blockDim.x){
        double vt = x[t];
        if(t == start) vt -= alpha;
        v[t] = vt * inv;

        double a = t == start ? alpha : 0.0;
        x[t] = a;
        A[t*n+k] = a;
    }
}

// p[j] = sum_{t=k+1}^{n-1} A[j*n+t] * v[t], un bloque por columna j = k+1..n-1.
// Como A es simetrica esto es A*v y sirve para los dos lados (H_k * A * H_k).
__global__ void get_Av_kernel(double* A, double* v, double* p, const int start, const int n){
    extern __shared__ double smem[];
    int j = start + blockIdx.x;

    double temp = 0.0;
    for(int t = start + threadIdx.x; t < n; t += blockDim.x){
        temp += A[j*n+t] * v[t];
    }
    smem[threadIdx.x] = temp;
    __syncthreads();

    for(int i = blockDim.x / 2; i > 0; i >>= 1){
        if(threadIdx.x < i){
            smem[threadIdx.x] += smem[threadIdx.x + i];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0){
        p[j] = smem[0];
    }
}

// w = p - (v.p) * v, un solo bloque
__global__ void get_w_kernel(double* v, double* p, double* w, const int start, const int n){
    extern __shared__ double smem[];

    double temp = 0.0;
    for(int t = start + threadIdx.x; t < n; t += blockDim.x){
        temp += v[t] * p[t];
    }
    smem[threadIdx.x] = temp;
    __syncthreads();

    for(int i = blockDim.x / 2; i > 0; i >>= 1){
        if(threadIdx.x < i){
            smem[threadIdx.x] += smem[threadIdx.x + i];
        }
        __syncthreads();
    }

    const double vp = smem[0];
    for(int t = start + threadIdx.x; t < n; t += blockDim.x){
        w[t] = p[t] - vp * v[t];
    }
}

// A -= 2*(v*w^T + w*v^T) sobre el bloque k+1..n-1, equivale a H_k * A * H_k
__global__ void matrix_substract_kernel(double* A, double* v, double* w, const int start, const int n){
    int t = start + blockIdx.y * blockDim.x + threadIdx.x;
    int j = start + blockIdx.x;
    if(t < n){
        A[j*n+t] -= 2 * (v[t] * w[j] + w[t] * v[j]);
    }
}

void hessenberg(double* A, int n){
    double* aD, *vD, *pD, *wD;
    size_t sizem, sizev;
    int k;
    sizem = n * n * sizeof(double);
    sizev = n * sizeof(double);

    cudaMalloc(&aD, sizem);
    cudaMalloc(&vD, sizev);
    cudaMalloc(&pD, sizev);
    cudaMalloc(&wD, sizev);
    cudaMemcpy(aD, A, sizem, cudaMemcpyHostToDevice);

    for(k=0; k<n-2; k++){
        int start = k + 1;
        int m = n - start;
        int nb = (m + NT - 1) / NT;

        householder_kernel<<<1, NT, NT*sizeof(double)>>>(aD, vD, k, n);

        // A <- H_k * A * H_k, solo en el bloque k+1..n-1
        get_Av_kernel<<<m, NT, NT*sizeof(double)>>>(aD, vD, pD, start, n);
        get_w_kernel<<<1, NT, NT*sizeof(double)>>>(vD, pD, wD, start, n);
        dim3 grid(m, nb);
        matrix_substract_kernel<<<grid, NT>>>(aD, vD, wD, start, n);
    }

    cudaMemcpy(A, aD, sizem, cudaMemcpyDeviceToHost);

    cudaFree(aD);
    cudaFree(vD);
    cudaFree(pD);
    cudaFree(wD);
}

void print_matrix(double* A, int n){
    int max_size = 10;
    int size = n > max_size ? max_size - 2 : n;
    int total_cols = (n > max_size) ? size + 2 : size;
    for(int row=0; row<size; row++){
        for(int col=0; col<size; col++){
            printf("%9.3f ", A[col*n+row]);
        }
        if(n > max_size){
            printf("%9s %9.3f", "...", A[(n-1)*n+row]);
        }
        printf("\n");
    }
    if(n > max_size){
        for(int i=0; i<total_cols; i++){
            printf("%9s ", "...");
        }
        printf("\n");
        for(int col=0; col<size; col++){
            printf("%9.3f ", A[col*n+(n-1)]);
        }
        printf("%9s %9.3f\n\n", "...", A[(n-1)*n+(n-1)]);
    }
}

double* rand_sym_matrix(const int n){
    double* A = (double*)malloc(n * n * sizeof(double));
    for(int i=0; i<n; i++){
        A[i*n+i] = (double)rand() / RAND_MAX * 200.0 - 100.0;
        for(int j=i+1; j<n; j++){
            double x = (double)rand() / RAND_MAX * 200.0 - 100.0;
            A[i*n+j] = x;
            A[j*n+i] = x;
        }
    }
    return A;
}

void save_matrix(const char* filename, double* A, int n){
      FILE* f = fopen(filename, "wb");
      fwrite(A, sizeof(double), n * n, f);
      fclose(f);
}

int main(int argc, char** argv){
    srand(500);
    int n = 2500;

    if(argc > 1){
        n = atoi(argv[1]);
    }

    // Segundo argumento opcional: 0 para no guardar A.bin/T.bin (benchmarks)
    int guardar = 1;
    if(argc > 2){
        guardar = atoi(argv[2]);
    }

    double* A = rand_sym_matrix(n);
    if(guardar) save_matrix("matlab/A.bin", A, n);
    print_matrix(A, n);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    hessenberg(A, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    print_matrix(A, n);

    if(guardar) save_matrix("matlab/T.bin", A, n);
    printf("%d,%d,%.6f\n", n, NT, elapsed_ms / 1000.0);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(A);
    return 0;
}
