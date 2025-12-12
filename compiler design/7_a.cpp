#include <stdio.h>
#include <string.h>
#include <ctype.h>

int pos = 0;
char input[200];

// Derivation strings
char leftmost[500] = "E";
char rightmost[500] = "E";

// Utility functions
void print_left(char *replacement) {
    char *p = strstr(leftmost, "E");
    if(p != NULL) {
        char temp[500];
        strncpy(temp, leftmost, p-leftmost);
        temp[p-leftmost] = '\0';
        strcat(temp, replacement);
        strcat(temp, p+1);
        strcpy(leftmost, temp);
        printf("Leftmost: %s\n", leftmost);
    }
}

void print_right(char *replacement) {
    char *p = strrchr(rightmost, 'E');
    if(p != NULL) {
        char temp[500];
        strncpy(temp, rightmost, p-rightmost);
        temp[p-rightmost] = '\0';
        strcat(temp, replacement);
        strcat(temp, p+1);
        strcpy(rightmost, temp);
        printf("Rightmost: %s\n", rightmost);
    }
}

// Forward declarations
void E();
void EPrime();
void T();
void TPrime();
void F();

// F -> (E) | id | a | b | c | d
void F() {
    if(input[pos] == '(') {
        pos++;
        print_left("(E)");
        print_right("(E)");
        E();
        if(input[pos] == ')') pos++;
    } else if(isalpha(input[pos])) {
        // accept a, b, c, d, id
        if(input[pos] == 'i' && input[pos+1]=='d') {
            pos += 2;
            print_left("id");
            print_right("id");
        } else {
            pos++;
            char temp[2] = {input[pos-1], '\0'};
            print_left(temp);
            print_right(temp);
        }
    } else {
        printf("Error: invalid symbol '%c'\n", input[pos]);
        pos++;
    }
}

// T' -> * F T' | / F T' | ε
void TPrime() {
    while(input[pos] == '*' || input[pos] == '/') {
        char op[3];
        op[0] = input[pos];
        op[1] = '\0';
        pos++;
        print_left(op); print_left("F");
        print_right(op); print_right("F");
        F();
    }
}

// T -> F T'
void T() {
    F();
    TPrime();
}

// E' -> + T E' | - T E' | ε
void EPrime() {
    while(input[pos] == '+' || input[pos] == '-') {
        char op[3];
        op[0] = input[pos];
        op[1] = '\0';
        pos++;
        print_left(op); print_left("T");
        print_right(op); print_right("T");
        T();
    }
}

// E -> T E'
void E() {
    T();
    EPrime();
}

// Main
int main() {
    printf("Enter arithmetic expression (a,b,c,d,id and + - * / with parentheses):\n");
    scanf("%s", input);

    printf("\nInitial: %s\n", leftmost);
    E();

    if(pos == strlen(input)) {
        printf("\nParsing completed successfully!\n");
    } else {
        printf("\nError in parsing at position %d!\n", pos);
    }

    printf("\nNote: Parse tree can be visualized based on leftmost/rightmost derivations.\n");
    return 0;
}
