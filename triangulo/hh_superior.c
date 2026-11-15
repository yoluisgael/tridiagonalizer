#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <time.h>

void print_vector(double* v, int size){
    int big_vector = size > 9 ? 1 : 0;
    if(big_vector) size = 9;
    printf("[");
    for(int i=0; i<size-1; i++) printf("%f, ", v[i]);
    if(big_vector) printf("%f, ...]\n", v[size-1]);
    else printf("%f]\n", v[size-1]);
}

void print_matrix(double* A, int n){
    int max_size = 10;
    int size = n > max_size ? max_size - 2 : n;
    int total_cols = (n > max_size) ? size + 2 : size;
    for(int i=0; i<size; i++){
        for(int j=0; j<size; j++){
            printf("%9.3f ", A[i*n+j]);
        }
        if(n > max_size){
            printf("%9s %9.3f", "...", A[i*n+n-1]);
        }
        printf("\n");
    }
    if(n > max_size){
        for(int i=0; i<total_cols; i++){
            printf("%9s ", "...");
        }
        printf("\n");
        int last_row = n * (n-1);
        for(int j=0; j<size; j++){
            printf("%9.3f ", A[last_row+j]);
        }
        printf("%9s %9.3f\n\n", "...", A[last_row+n-1]);
    }
}

double dot_product(double* a, double*b, const int size){
    double result = 0;
    for(int i=0; i<size; i++) result += a[i] * b[i];
    return result;
}

double matrix_vector_dot_product(double* A, double* v, const int j, const int n){
    double result = 0;
    for(int i=0; i<n; i++) result += v[i] * A[i*n+j];
    return result;
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

double* hh_vector(double* x, const int size){
    // Norma euclidiana
    double norma = 0;
    norma = sqrt(dot_product(x, x, size));

    // Vector householder
    double alpha = x[0] < 0 ? norma : -norma;
    double* v = (double*)malloc(size * sizeof(double));
    for(int i=0; i<size; i++){
        v[i] = x[i];
    }
    v[0] = v[0] - alpha;

    // Calcular beta
    double beta = 2.0 / dot_product(v, v, size);

    // Calcular Hx
    double vx = dot_product(v, x, size);
    double* Hx = (double*)malloc(size * sizeof(double));
    for(int i=0; i<size; i++){
        Hx[i] = x[i] - beta * vx * v[i];
    }

    free(v);
    return Hx;
}

void get_v(double* A, double* v, const int k, const int n){
    if(k > 0) v[k-1] = 0;
    for(int i=k; i<n; i++) v[i] = A[i*n+k];

    //Sacamos norma para obtener alfa
    double norma = sqrt(dot_product(v,v,n));
    
    v[k] -= v[k] < 0 ? norma : -norma;

    //Normalizamos el vector
    norma = sqrt(dot_product(v,v,n));
    for(int i=k; i<n; i++) v[i] /= norma;
}

void hh_matrix_superior(double* A, const int n){
    double* v = (double*)calloc(n, sizeof(double));
    double delta;
    int i, j;
    for(int k=0; k<n-1; k++){
        get_v(A, v, k, n);
        #pragma omp parallel for private(delta, i)
        for(j=k; j<n; j++){
            delta = matrix_vector_dot_product(A,v,j,n);
            //Resta de columnas
            for(i=k; i<n; i++) A[i*n+j] -= 2.0 * delta * v[i];
        }
    }

    free(v);
}

int main(int argc, char** argv){
    srand(500);
    int n = 2500;
    double t0, t1;
    int n_threads = omp_get_num_threads();

    if(argc > 1){
        n = atoi(argv[1]);
    }
    if(argc > 2){
        n_threads = atoi(argv[2]);
    }

    omp_set_num_threads(n_threads);

    double* A = rand_sym_matrix(n);
    //print_matrix(A, n);
    
    t0 = omp_get_wtime();
    hh_matrix_superior(A, n);
    t1 = omp_get_wtime();

    print_matrix(A, n);

    printf("%d,%d,%.6f\n", n, n_threads, t1-t0);

    free(A);
    return 0;
}