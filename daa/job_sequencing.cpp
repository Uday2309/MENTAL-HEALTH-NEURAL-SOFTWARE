#include <iostream>
#include <algorithm>
using namespace std;

struct Job {
    char id;
    int deadline;
    int profit;
};

bool cmp(Job a, Job b) {
    return a.profit > b.profit;
}

int main() {
    Job jobs[] = {
        {'A', 2, 100},
        {'B', 1, 19},
        {'C', 2, 27},
        {'D', 1, 25},
        {'E', 3, 15}
    };

    int n = sizeof(jobs) / sizeof(jobs[0]);
    sort(jobs, jobs + n, cmp);

    int maxDeadline = 0;
    for (int i = 0; i < n; i++)
        if (jobs[i].deadline > maxDeadline)
            maxDeadline = jobs[i].deadline;

    char result[maxDeadline];
    bool slot[maxDeadline];

    for (int i = 0; i < maxDeadline; i++)
        slot[i] = false;

    for (int i = 0; i < n; i++) {
        for (int j = jobs[i].deadline - 1; j >= 0; j--) {
            if (!slot[j]) {
                slot[j] = true;
                result[j] = jobs[i].id;
                break;
            }
        }
    }

    for (int i = 0; i < maxDeadline; i++)
        if (slot[i])
            cout << result[i] << " ";

    return 0;
}
