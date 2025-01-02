#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>

void print_matrix(double** a, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.32f ", a[i][j]);
        }
        printf("\n");
    }
}

double** malloc_2d(int n) {
    double** p = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        p[i] = (double*)malloc(n * sizeof(double));
    }
    return p;
}

void free_2d(double** a, int n) {
    for (int i = 0; i < n; i++) {
        free(a[i]);
    }
    free(a);
}

double sum(double* a, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += a[i];
    }
    return sum;
}

void add(double* a, double* b, int n) {
    for (int i = 0; i < n; i++) {
        a[i] += b[i];
    }
}

void copy(double* a, double* b, int n) {
    for (int i = 0; i < n; i++) {  
        a[i] = b[i];
    }
}

void scalar_multiply(double* a, double* b, double scalar, int n) {
    for(int i = 0; i < n; i++){
        a[i] -= b[i] * scalar;
    }
}

double inner_product(double* a, double* b, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

void swap(double* a, double* b, int n) {
    for (int i = 0; i < n; i++) {
        double temp = a[i];
        a[i] = b[i];
        b[i] = temp;
    }
}

void update_mu(double** mu, double** b, double** b_star, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            mu[i][j] = inner_product(b[i], b_star[j], n) / inner_product(b_star[j], b_star[j], n);
        }
    }
}

void schmidt(double** b, double** b_star, int n){
    for(int i = 0 ; i < n ; i ++){
        memset(b_star[i], 0, n * sizeof(double));
    }
    for(int i = 0; i < n; i++){
        copy(b_star[i], b[i], n);
        for(int j = 0; j < i; j++){
            double scalar = inner_product(b[i], b_star[j], n) / inner_product(b_star[j], b_star[j], n);
            scalar_multiply(b_star[i], b_star[j], scalar, n);
        }
    }
}

void lll_algorithm(double** b, int n, double delta) {
    double** b_star = malloc_2d(n);
    double** mu = malloc_2d(n);
    schmidt(b, b_star, n);
    update_mu(mu, b, b_star, n);
    for(int k = 1; k < n;){
        for(int j = k - 1; j >= 0; j --){
            if(fabs(mu[k][j]) > 0.5){
                for(int i = 0; i < n; i++){
                    b[k][i] -= round(mu[k][j]) * b[j][i];
                }
                schmidt(b, b_star, n);
                update_mu(mu, b, b_star, n);
            }
        }
        if(inner_product(b_star[k], b_star[k], n) >= (delta - mu[k][k - 1] * mu[k][k - 1]) * inner_product(b_star[k - 1], b_star[k - 1], n)){
            k ++;
        }
        else{
            swap(b[k], b[k - 1], n);
            schmidt(b, b_star, n);
            update_mu(mu, b, b_star, n);
            k = fmax(k - 1, 1);
        }
    }
    
}

int compare_matrix(double** a, double** b, int n){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            if(a[i][j] != b[i][j]){
                return 0;
            }
        }
    }
    return 1;
}

int main(){
    double** problem1 = malloc_2d(2);
    problem1[0][0] = 3, problem1[0][1] = 1;
    problem1[1][0] = 2, problem1[1][1] = 2;
    double** solution1 = malloc_2d(2);
    solution1[0][0] = 3, solution1[0][1] = 1;
    solution1[1][0] = -0.4, solution1[1][1] = 1.2;
    double** b = malloc_2d(2);
    schmidt(problem1, b, 2);
    print_matrix(b, 2);
    print_matrix(solution1, 2);

    double** problem2 = malloc_2d(3);
    problem2[0][0] = 1, problem2[0][1] = 1, problem2[0][2] = 1;
    problem2[1][0] = -1, problem2[1][1] = 0, problem2[1][2] = 2;
    problem2[2][0] = 3, problem2[2][1] = 5, problem2[2][2] = 6;
    double** solution2 = malloc_2d(3);
    solution2[0][0] = 0, solution2[0][1] = 1, solution2[0][2] = 0;
    solution2[1][0] = 1, solution2[1][1] = 0, solution2[1][2] = 1;
    solution2[2][0] = -1, solution2[2][1] = 0, solution2[2][2] = 2;
    lll_algorithm(problem2, 3, 0.75);
    print_matrix(problem2, 3);
    print_matrix(solution2, 3);
}
