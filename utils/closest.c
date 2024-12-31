#include <malloc.h>
#include <math.h>
#include <stdio.h>

/**
 * @param n: number of dimensions 
 * @param G: generator matrix of the lattice (lower-triangular, positive diagonal elements)
 * @param r: point in n-dimensional space
 * @return: 
 */
void get_closest_point(int n, double G[n][n], double r[n], int u_hat[n]) {
    double c = 1e18;
    double eps = 1e-7;
    int i = n;

    int* d = malloc(n * sizeof(int)); 
    int* u = malloc(n * sizeof(int));
    int* delta = malloc(n * sizeof(int));
    double* lambda = malloc((n + 1) * sizeof(double));
    double* p = malloc(n * sizeof(double));
    double** F = malloc(n * sizeof(double*));

    for (int i = n - 1; i >= 0; i--) {
        F[i] = malloc(n * sizeof(double));
        d[i] = n - 1;
        F[n - 1][i] = r[i];
    }
    lambda[n] = 0;

    while (1) {
        do {
            if (i != 0) {
                i--;
                for (int j = d[i]; j > i; j--) 
                    F[j - 1][i] = F[j][i] - u[j] * G[j][i];
                p[i] = F[i][i] / G[i][i];
                u[i] = round(p[i]);
                double y = (p[i] - u[i]) * G[i][i];
                delta[i] = y > eps ? 1 : -1;
                lambda[i] = lambda[i + 1] + y * y;
            } else {
                for (int j = 0; j < n; j++) 
                    u_hat[j] = u[j];
                c = lambda[0];
            }
        } while (lambda[i] < c);

        int m = i;

        do {
            if (i == n - 1) {
                free(d), free(lambda), free(delta), free(u), free(p);
                for (int i = 0; i < n; i++)
                    free(F[i]);
                free(F);
                return; 
            } else {
                i++;
                u[i] += delta[i];
                delta[i] = -delta[i] - (delta[i] > eps ? 1 : -1);
                double y = (p[i] - u[i]) * G[i][i];
                lambda[i] = lambda[i + 1] + y * y;
            }
        } while (lambda[i] >= c);

        for (int j = m; j < i; j++) d[j] = i;
        for (int j = m - 1; j >= 0; j--) {
            if (d[j] < i) d[j] = i;
            else break;
        }
    }
} 

void test() {
    double G[2][2] = {{1, 0}, {0.5, 1}};
    double r[2] = {1.5, 1.5};
    int u[2];
    get_closest_point(2, G, r, u);
    printf("Closest point: (%d, %d)\n", u[0], u[1]);
}

int main() {
    test();
    return 0;
}