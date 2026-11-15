#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

void hessenberg(double* A, const int n){
    double* v = (double*)calloc(n, sizeof(double));
    double* p = (double*)calloc(n, sizeof(double));
    double* w = (double*)calloc(n, sizeof(double));

    #pragma omp parallel
    for(int k=0; k<n-2; k++){
        const int start = k + 1;

        // Vector householder de la columna k (O(n), lo hace un solo hilo)
        #pragma omp single
        {
            double* x = A + k * n;
            const double x0 = x[start];

            //norma al cuadrado de la columna
            double temp = 0.0;
            for(int t=start; t<n; t++) temp += x[t] * x[t];

            const double norma = sqrt(temp);
            const double alpha = x0 < 0 ? norma : -norma;

            //||v||^2 = 2*norma*(norma + |x0|), no hace falta un segundo producto punto
            const double norma_v = sqrt(2.0 * norma * (norma + fabs(x0)));
            const double inv = norma_v > 0 ? 1.0 / norma_v : 0.0;

            for(int t=start; t<n; t++){
                double vt = x[t];
                if(t == start) vt -= alpha;
                v[t] = vt * inv;

                double a = t == start ? alpha : 0.0;
                x[t] = a;
                A[t*n+k] = a;
            }
        }

        // p[j] = sum_{t=k+1}^{n-1} A[j*n+t] * v[t]
        #pragma omp for schedule(static)
        for(int j=start; j<n; j++){
            double temp = 0.0;
            for(int t=start; t<n; t++) temp += A[j*n+t] * v[t];
            p[j] = temp;
        }

        // w = p - (v.p) * v
        #pragma omp single
        {
            double vp = 0.0;
            for(int t=start; t<n; t++) vp += v[t] * p[t];
            for(int t=start; t<n; t++) w[t] = p[t] - vp * v[t];
        }

        // A -= 2*(v*w^T + w*v^T) sobre el bloque k+1..n-1, equivale a H_k * A * H_k
        #pragma omp for schedule(static)
        for(int j=start; j<n; j++){
            const double vj = v[j];
            const double wj = w[j];
            double* col = A + j * n;
            for(int t=start; t<n; t++){
                col[t] -= 2 * (v[t] * wj + w[t] * vj);
            }
        }
    }

    free(v);
    free(p);
    free(w);
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
    double t0, t1;
    int n_threads = omp_get_max_threads();

    if(argc > 1){
        n = atoi(argv[1]);
    }
    if(argc > 2){
        n_threads = atoi(argv[2]);
    }

    // Tercer argumento opcional: 0 para no guardar A.bin/T.bin (benchmarks)
    int guardar = 1;
    if(argc > 3){
        guardar = atoi(argv[3]);
    }

    omp_set_num_threads(n_threads);

    double* A = rand_sym_matrix(n);
    if(guardar) save_matrix("matlab/A.bin", A, n);
    //print_matrix(A, n);

    t0 = omp_get_wtime();
    hessenberg(A, n);
    t1 = omp_get_wtime();

    print_matrix(A, n);

    if(guardar) save_matrix("matlab/T.bin", A, n);
    printf("%d,%d,%.6f\n", n, n_threads, t1-t0);

    free(A);
    return 0;
}
