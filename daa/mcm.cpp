#include <stdio.h>
#define MAX 10
#define INF 1000000

int min(int a, int b) { return (a < b) ? a : b; }

int matrixChain(int p[], int n) {
    int m[MAX][MAX] = {0};
    for(int L=2; L<n; L++) {
        for(int i=1; i<=n-L; i++) {
            int j=i+L-1;
            m[i][j] = INF;
            for(int k=i; k<j; k++)
                m[i][j] = min(m[i][j], m[i][k] + m[k+1][j] + p[i-1]*p[k]*p[j]);
        }
    }
    return m[1][n-1];
}

int main() {
    int dim[] = {40, 20, 30, 10, 30};
    int n = sizeof(dim)/sizeof(dim[0]);
    printf("Minimum multiplication cost: %d\n", matrixChain(dim, n));
    return 0;
}
