#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cusolverDn.h>

// Tridiagonalizacion con cuSOLVER (cusolverDnDsytrd), referencia para 3d_opt.cu.
// Se cronometra lo mismo que en 3d_opt.cu: reservas, copias host<->device y calculo.
// dsytrd solo devuelve la diagonal d y la subdiagonal e; con ellas se arma T.

#define CHECK_CUSOLVER(x) do{ \
    cusolverStatus_t s = (x); \
    if(s != CUSOLVER_STATUS_SUCCESS){ \
        fprintf(stderr, "cuSOLVER error %d en linea %d\n", (int)s, __LINE__); \
        exit(1); \
    } \
}while(0)

// Deja en d la diagonal y en e la subdiagonal de T (A se copia al device y no se modifica)
void hessenberg(cusolverDnHandle_t handle, double* A, double* d, double* e, int n){
    double* aD, *dD, *eD, *tauD, *workD;
    int* infoD;
    int lwork, info;
    size_t sizem = (size_t)n * n * sizeof(double);
    size_t sizev = n * sizeof(double);
    const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    cudaMalloc(&aD, sizem);
    cudaMalloc(&dD, sizev);
    cudaMalloc(&eD, sizev);
    cudaMalloc(&tauD, sizev);
    cudaMalloc(&infoD, sizeof(int));
    cudaMemcpy(aD, A, sizem, cudaMemcpyHostToDevice);

    CHECK_CUSOLVER(cusolverDnDsytrd_bufferSize(handle, uplo, n, aD, n, dD, eD, tauD, &lwork));
    cudaMalloc(&workD, (size_t)lwork * sizeof(double));

    CHECK_CUSOLVER(cusolverDnDsytrd(handle, uplo, n, aD, n, dD, eD, tauD, workD, lwork, infoD));

    cudaMemcpy(d, dD, sizev, cudaMemcpyDeviceToHost);
    cudaMemcpy(e, eD, (n - 1) * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&info, infoD, sizeof(int), cudaMemcpyDeviceToHost);
    if(info != 0){
        fprintf(stderr, "dsytrd devolvio info = %d\n", info);
        exit(1);
    }

    cudaFree(aD);
    cudaFree(dD);
    cudaFree(eD);
    cudaFree(tauD);
    cudaFree(workD);
    cudaFree(infoD);
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
    double* A = (double*)malloc((size_t)n * n * sizeof(double));
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
      fwrite(A, sizeof(double), (size_t)n * n, f);
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

    // El handle se crea fuera del cronometro (inicializa el contexto de CUDA)
    cusolverDnHandle_t handle;
    CHECK_CUSOLVER(cusolverDnCreate(&handle));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double* d = (double*)malloc(n * sizeof(double));
    double* e = (double*)malloc(n * sizeof(double));

    cudaEventRecord(start);
    hessenberg(handle, A, d, e, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    // T por columnas, igual que el resultado de 3d_opt.cu (fuera del cronometro)
    for(size_t i=0; i<(size_t)n*n; i++) A[i] = 0.0;
    for(int i=0; i<n; i++){
        A[i*n+i] = d[i];
        if(i < n-1){
            A[i*n+i+1] = e[i];
            A[(i+1)*n+i] = e[i];
        }
    }
    free(d);
    free(e);

    print_matrix(A, n);

    if(guardar) save_matrix("matlab/T.bin", A, n);
    printf("%d,cusolver,%.6f\n", n, elapsed_ms / 1000.0);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cusolverDnDestroy(handle);

    free(A);
    return 0;
}
