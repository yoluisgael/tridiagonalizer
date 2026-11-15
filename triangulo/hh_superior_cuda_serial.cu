#include <iostream>
#include <cstdio>
#include <cstdlib>

int NB_V;
int NT = 512;

__global__ void col2vector_kernel(double* A, double* v, const int k, const int n){
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if(t < n){
        if(t < k) v[t] = 0;
        else v[t] = A[k * n + t];
    }
}

__global__ void dotp_partial_kernel(double* a, double* b, double* partial, const int n){
    extern __shared__ double smem[];
    int t = blockDim.x * blockIdx.x + threadIdx.x;

    //multiplicación
    double temp = 0.0;
    if(t < n){
        temp = a[t] * b[t];
    }
    smem[threadIdx.x] = temp;
    __syncthreads();

    //reducción dentro del bloque
    for(int i = blockDim.x / 2; i > 0; i >>= 1){
        if(threadIdx.x < i){
            smem[threadIdx.x] += smem[threadIdx.x + i];
        }
        __syncthreads();
    }

    //cada bloque escribe su propia parcial
    if(threadIdx.x == 0){
        partial[blockIdx.x] = smem[0];
    }
}

__global__ void reduce_partial_kernel(double* partial, double* result, const int nb){
    extern __shared__ double smem[];

    double sum = 0.0;
    for(int t = threadIdx.x; t < nb; t += blockDim.x){
        sum += partial[t];
    }
    smem[threadIdx.x] = sum;
    __syncthreads();

    for(int i = blockDim.x / 2; i > 0; i >>= 1){
        if(threadIdx.x < i){
            smem[threadIdx.x] += smem[threadIdx.x + i];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0){
        *result = smem[0];
    }
}

void dotp_dev(double* a, double* b, double* partial, double* result, const int n){
    dotp_partial_kernel<<<NB_V, NT, NT*sizeof(double)>>>(a, b, partial, n);
    reduce_partial_kernel<<<1, NT, NT*sizeof(double)>>>(partial, result, NB_V);
}

__global__ void set_alpha_kernel(double* v, double* norma_sqr, const int k){
    double norma = sqrt(*norma_sqr);
    v[k] -= v[k] < 0 ? norma : -norma;
}

__global__ void normalize_kernel(double* v, double* norma_sqr, const int n){
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    double norma = sqrt(*norma_sqr);

    if(t < n){
        v[t] /= norma;
    }
}

void get_v_dev(double* A, double* v, double* partial, double* norma_sqr, const int k, const int n){
    //Extraemos la columna
    col2vector_kernel<<<NB_V, NT>>>(A, v, k, n);

    //Ponemos alfa
    dotp_dev(v, v, partial, norma_sqr, n);
    set_alpha_kernel<<<1, 1>>>(v, norma_sqr, k);

    //Normalizamos
    dotp_dev(v, v, partial, norma_sqr, n);
    normalize_kernel<<<NB_V, NT>>>(v, norma_sqr, n);
}

void get_delta_dev(double* A, double* v, double* partial, double* delta, const int j, const int n){
    dotp_dev(A+j*n, v, partial, delta, n);
}

__global__ void column_substract_kernel(double* A, double* v, double* delta, const int j, const int n){
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if(t < n){
        A[j*n+t] -= 2 * (*delta) * v[t];
    }
}

__global__ void get_all_deltas_kernel(double* A, double* v, double *deltas, const int k, const int n){
    extern __shared__ double smem[];
    int t = threadIdx.x;
    double temp = 0.0;
    int j = k + blockIdx.x;

    for(int t = threadIdx.x; t < n; t += blockDim.x){
        temp += A[j*n+t] * v[t];
    }

    smem[t] = temp;
    __syncthreads();

    for(int i = blockDim.x/2; i > 0; i >>= 1){
        if(threadIdx.x < i){
            smem[threadIdx.x] += smem[threadIdx.x + i];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0){
        deltas[j] = smem[0];
    }
}

__global__ void matrix_substract_kernel(double* A, double* v, double* deltas, const int k, const int n){
    int t = blockIdx.y * blockDim.x + threadIdx.x;
    int j = k + blockIdx.x;
    if(t < n){
        A[j*n+t] -= 2 * deltas[j] * v[t];
    }
}

void get_v(double* A, double* aD, double* v, double* vD, const int k, const int n){
    cudaMemcpy(A, aD+(k*n), n*sizeof(double), cudaMemcpyDeviceToHost);

    double norma = 0.0;
    for(int i=0; i<k; i++) v[i] = 0;
    for(int i=k; i<n; i++){
        v[i] = A[i];
        norma += v[i] * v[i];
    }

    norma = sqrt(norma);
    
    v[k] -= v[k] < 0 ? norma : -norma;
    norma = 0.0;

    for(int i=k; i<n; i++) norma += v[i] * v[i];
    norma = sqrt(norma);

    for(int i=k; i<n; i++) v[i] /= norma;

    cudaMemcpy(vD, v, n*sizeof(double), cudaMemcpyHostToDevice);
}

void hh_triangular(double* A, int n){
    double* aD, *vD, *partialD, *normaD;
    double* deltasD;
    size_t sizem, sizev;
    int k;
    sizem = n * n * sizeof(double);
    sizev = n * sizeof(double);
    NB_V = (n + NT - 1) / NT;
    
    double* aH = (double*)malloc(sizev);
    double* vH = (double*)malloc(sizev);
    cudaMalloc(&aD, sizem);
    cudaMalloc(&vD, sizev);
    cudaMalloc(&partialD, NB_V * sizeof(double));
    cudaMalloc(&normaD, sizeof(double));
    cudaMemcpy(aD, A, sizem, cudaMemcpyHostToDevice);
    cudaMalloc(&deltasD, sizev);


    for(k=0; k<n-1; k++){
        get_v(aH, aD, vH, vD, k, n);
        get_all_deltas_kernel<<<n-k, NT, NT*sizeof(double)>>>(aD, vD, deltasD, k, n);
        dim3 grid(n-k, NB_V);
        matrix_substract_kernel<<<grid, NT>>>(aD, vD, deltasD, k, n);
    }

    cudaMemcpy(A, aD, sizem, cudaMemcpyDeviceToHost);

    cudaFree(aD);
    cudaFree(vD);
    cudaFree(partialD);
    cudaFree(normaD);
    cudaFree(deltasD);
    free(aH);
    free(vH);
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

int main(int argc, char** argv){
    srand(500);
    int n = 2500;

    if(argc > 1){
        n = atoi(argv[1]);
    }

    double* A = rand_sym_matrix(n);
    //print_matrix(A, n);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    hh_triangular(A, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    print_matrix(A, n);

    printf("%d,%d,%.6f\n", n, NT, elapsed_ms / 1000.0);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(A);
    return 0;
}