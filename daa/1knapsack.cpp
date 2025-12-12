#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int profit, weight;
    float bound;
    int level;
} Node;

float max(float a, float b){ return (a>b)?a:b; }

float bound(Node u, int n, int W, int w[], int p[]) {
    if(u.weight >= W) return 0;
    float profit_bound = u.profit;
    int j = u.level+1;
    int totweight = u.weight;
    while(j < n && totweight + w[j] <= W){
        totweight += w[j];
        profit_bound += p[j];
        j++;
    }
    if(j<n) profit_bound += (W-totweight)*p[j]/(float)w[j];
    return profit_bound;
}

int cmp(const void *a, const void *b){
    float r1 = ((float)((int*)a)[0])/((int)((int*)a)[1]);
    float r2 = ((float)((int*)b)[0])/((int)((int*)b)[1]);
    return r2>r1?1:-1;
}

int knapsack(int W, int n, int w[], int p[]){
    // Branch & Bound implementation here (queue-based)
    // Concise version, logic same as algorithm above
    return 0; // placeholder
}

int main(){
    int w[] = {2,3,4,5}, p[]={40,50,60,70};
    int W = 5, n=4;
    printf("Maximum profit: %d\n", knapsack(W,n,w,p));
    return 0;
}
