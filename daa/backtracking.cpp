#include <stdio.h>
#define V 5

int graph[V][V] = {
    {0,1,0,1,1},
    {1,0,1,1,0},
    {0,1,0,1,1},
    {1,1,1,0,1},
    {1,0,1,1,0}
};

int colors[V];

int isSafe(int v, int c) {
    for(int i=0;i<V;i++)
        if(graph[v][i] && colors[i]==c) return 0;
    return 1;
}

void graphColoring(int v, int m, int *solutions, int depth) {
    if(v==V) {
        (*solutions)++;
        for(int i=0;i<V;i++) printf("%d ", colors[i]);
        printf("\n");
        return;
    }
    for(int c=1;c<=m;c++) {
        if(isSafe(v,c)) {
            colors[v]=c;
            graphColoring(v+1,m,solutions,depth+1);
            colors[v]=0; // backtrack
        }
    }
}

int main() {
    int m=3, solutions=0;
    graphColoring(0,m,&solutions,0);
    printf("Chromatic number: %d\n", m);
    printf("Total solutions: %d\n", solutions);
    return 0;
}
