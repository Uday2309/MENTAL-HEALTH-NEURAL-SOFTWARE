#include <stdio.h>
#include <string.h>
#define MAX 20

int isLL1(char production[MAX][MAX], int n) {
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
           
            if (production[i][0] == production[j][0] && production[i][0] != '#') {
                return 0; 
            }
        }
    }
    return 1; 
}

int main() {
    int n;
    char production[MAX][MAX];

    printf("Enter number of productions: ");
    scanf("%d", &n);

    printf("Enter productions (use # for epsilon):\n");
    for (int i = 0; i < n; i++) {
        scanf("%s", production[i]);
    }

    printf("Given grammar:\n");
    for (int i = 0; i < n; i++) {
        printf("%s\n", production[i]);
    }

    int result = isLL1(production, n);
    if (result) {
        printf("The grammar is LL(1)\n");
    } else {
        printf("The grammar is NOT LL(1)\n");
    }

    return 0;
}

