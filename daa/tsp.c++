#include <bits/stdc++.h>
using namespace std;

int n;
int dist[20][20];
int dp[1<<20][20];
int parentPath[1<<20][20];

int tsp(int mask, int pos) {
    if (mask == (1<<n) - 1)
        return dist[pos][0];

    if (dp[mask][pos] != -1)
        return dp[mask][pos];

    int ans = INT_MAX, nextCity = -1;

    for (int city = 0; city < n; city++) {
        if ((mask & (1 << city)) == 0) {
            int cost = dist[pos][city] + tsp(mask | (1 << city), city);
            if (cost < ans) {
                ans = cost;
                nextCity = city;
            }
        }
    }

    parentPath[mask][pos] = nextCity;
    return dp[mask][pos] = ans;
}

int main() {
    n = 4;

    int matrix[4][4] = {
        {0, 10, 15, 20},
        {10, 0, 35, 25},
        {15, 35, 0, 30},
        {20, 25, 30, 0}
    };

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            dist[i][j] = matrix[i][j];

    memset(dp, -1, sizeof(dp));
    memset(parentPath, -1, sizeof(parentPath));

    int minCost = tsp(1, 0);

    cout << "Minimum TSP Cost: " << minCost << "\n";

    cout << "Optimal Path: ";
    int mask = 1, pos = 0;
    cout << pos << " ";

    while (true) {
        int nxt = parentPath[mask][pos];
        if (nxt == -1) break;
        cout << nxt << " ";
        mask |= (1 << nxt);
        pos = nxt;
    }

    cout << "0";

    return 0;
}
