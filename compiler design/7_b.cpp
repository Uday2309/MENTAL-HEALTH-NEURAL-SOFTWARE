#include <stdio.h>

void parse1() {
    printf("Derivation 1 (id + id) * id :\n");
    printf("E => E * E\n");
    printf("E => (E + E) * E\n");
    printf("E => (id + id) * id\n\n");
}

void parse2() {
    printf("Derivation 2 id + (id * id) :\n");
    printf("E => E + E\n");
    printf("E => id + (E * E)\n");
    printf("E => id + (id * id)\n\n");
}

int main() {
    printf("Given Grammar:\nE → E + E | E * E | (E) | id\n\n");
    printf("Input string: id + id * id\n\n");
    
    parse1();
    parse2();
    
    printf("Conclusion: Grammar is AMBIGUOUS (multiple parse trees possible).\n");
    return 0;
}
