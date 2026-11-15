#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int NT = 128;
// Columnas por bloque en get_Av_rows_kernel
int CH = 64;

// Solo se guarda el triangulo inferior, empaquetado por columnas:
// el elemento (t, j) con t >= j esta en P[off(j, n) + (t - j)].
__host__ __device__ inline size_t off(const int j, const int n){
    return (size_t)j * (2 * (size_t)n - j + 1) / 2;
}

__global__ void householder_kernel(double* P, double* v, const int k, const int n){
    extern __shared__ double smem[];
    const int start = k + 1;
    // x[t] = elemento (t, k)
    double* x = P + off(k, n) - k;
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
    }

    // Solo se guarda la subdiagonal; los ceros los pone el CPU al reconstruir
    if(threadIdx.x == 0){
        x[start] = alpha;
    }
}

// Parte de A*v que cae en el triangulo superior, leida por filas del inferior:
// S[by*n + r] = sum_{c en el bloque de columnas by, c < r} L[r][c] * v[c]
// Un hilo por fila r; hilos vecinos leen posiciones contiguas de la misma columna.
__global__ void get_Av_rows_kernel(const double* P, const double* v, double* S, const int start, const int n, const int ch){
    const int r = start + blockIdx.x * blockDim.x + threadIdx.x;
    const int c0 = start + blockIdx.y * ch;
    if(r >= n || c0 >= r) return;
    const int c1 = c0 + ch < r ? c0 + ch : r;

    size_t idx = off(c0, n) + (r - c0);
    double temp = 0.0;
    for(int c = c0; c < c1; c++){
        temp += P[idx] * v[c];
        idx += n - c - 1;
    }
    S[(size_t)blockIdx.y * n + r] = temp;
}

// p[j] = sum_{t=j}^{n-1} L[t][j] * v[t] + parciales de S, un bloque por columna j = k+1..n-1.
// Junto con get_Av_rows_kernel da p = A*v usando solo el triangulo inferior.
__global__ void get_Av_kernel(const double* P, const double* v, const double* S, double* p, const int start, const int n, const int ch){
    extern __shared__ double smem[];
    const int j = start + blockIdx.x;
    const double* x = P + off(j, n) - j;

    double temp = 0.0;
    for(int t = j + threadIdx.x; t < n; t += blockDim.x){
        temp += x[t] * v[t];
    }
    const int nc = (j - start + ch - 1) / ch;
    for(int i = threadIdx.x; i < nc; i += blockDim.x){
        temp += S[(size_t)i * n + j];
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

// A -= 2*(v*w^T + w*v^T) sobre el bloque k+1..n-1, solo el triangulo inferior (t >= j).
// Un bloque por columna j.
__global__ void matrix_substract_kernel(double* P, const double* v, const double* w, const int start, const int n){
    const int j = start + blockIdx.x;
    double* x = P + off(j, n) - j;
    const double vj = v[j];
    const double wj = w[j];
    for(int t = j + threadIdx.x; t < n; t += blockDim.x){
        x[t] -= 2 * (v[t] * wj + w[t] * vj);
    }
}

// out[i] = diagonal, out[n+i] = subdiagonal
__global__ void extract_kernel(const double* P, double* out, const int n){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        const size_t o = off(i, n);
        out[i] = P[o];
        if(i < n - 1) out[n + i] = P[o + 1];
    }
}

void hessenberg(double* A, int n){
    double* pD, *vD, *avD, *wD, *sD, *outD;
    size_t sizep, sizev;
    int k;
    sizep = off(n, n) * sizeof(double);
    sizev = n * sizeof(double);

    // Empaquetar el triangulo inferior
    double* P = (double*)malloc(sizep);
    for(int j=0; j<n; j++){
        memcpy(P + off(j, n), A + (size_t)j*n + j, (n - j) * sizeof(double));
    }

    cudaMalloc(&pD, sizep);
    cudaMalloc(&vD, sizev);
    cudaMalloc(&avD, sizev);
    cudaMalloc(&wD, sizev);
    cudaMalloc(&sD, (size_t)((n + CH - 1) / CH) * sizev);
    cudaMalloc(&outD, 2 * sizev);
    cudaMemcpy(pD, P, sizep, cudaMemcpyHostToDevice);

    for(k=0; k<n-2; k++){
        int start = k + 1;
        int m = n - start;
        int nb = (m + NT - 1) / NT;
        int nc = (m - 1 + CH - 1) / CH;

        householder_kernel<<<1, NT, NT*sizeof(double)>>>(pD, vD, k, n);

        // A <- H_k * A * H_k, solo en el bloque k+1..n-1
        dim3 grid(nb, nc);
        get_Av_rows_kernel<<<grid, NT>>>(pD, vD, sD, start, n, CH);
        get_Av_kernel<<<m, NT, NT*sizeof(double)>>>(pD, vD, sD, avD, start, n, CH);
        get_w_kernel<<<1, NT, NT*sizeof(double)>>>(vD, avD, wD, start, n);
        matrix_substract_kernel<<<m, NT>>>(pD, vD, wD, start, n);
    }

    extract_kernel<<<(n + NT - 1) / NT, NT>>>(pD, outD, n);

    // Solo bajan la diagonal y la subdiagonal; la tridiagonal completa se arma aqui
    double* out = (double*)malloc(2 * sizev);
    cudaMemcpy(out, outD, 2 * sizev, cudaMemcpyDeviceToHost);

    memset(A, 0, (size_t)n * n * sizeof(double));
    for(int i=0; i<n; i++){
        A[(size_t)i*n+i] = out[i];
        if(i < n - 1){
            A[(size_t)i*n+i+1] = out[n+i];
            A[(size_t)(i+1)*n+i] = out[n+i];
        }
    }

    free(P);
    free(out);
    cudaFree(pD);
    cudaFree(vD);
    cudaFree(avD);
    cudaFree(wD);
    cudaFree(sD);
    cudaFree(outD);
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
    //print_matrix(A, n);

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
