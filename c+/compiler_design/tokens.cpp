#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

char keywords[32][10] = {
    "auto","break","case","char","const","continue","default","do","double","else",
    "enum","extern","float","for","goto","if","int","long","register","return",
    "short","signed","sizeof","static","struct","switch","typedef","union",
    "unsigned","void","volatile","while"
};

int totalTokens = 0; // Global token counter

int isKeyword(char *word) {
    for (int i = 0; i < 32; i++) {
        if (strcmp(word, keywords[i]) == 0)
            return 1;
    }
    return 0;
}

int isDelimiter(char ch) {
    return ch == ' ' || ch == '+' || ch == '-' || ch == '*' || ch == '/' || 
           ch == ',' || ch == ';' || ch == '>' || ch == '<' || ch == '=' || 
           ch == '(' || ch == ')' || ch == '[' || ch == ']' || ch == '{' || ch == '}';
}

int isOperator(char ch) {
    return ch == '+' || ch == '-' || ch == '*' || ch == '/' || 
           ch == '=' || ch == '>' || ch == '<';
}

int isNumber(char *str) {
    for (int i = 0; str[i]; i++) {
        if (!isdigit(str[i]))
            return 0;
    }
    return 1;
}

void extractTokens(char *line) {
    int i = 0;
    int len = strlen(line);
    char token[100];
    int tokenIndex = 0;

    while (i <= len) {
        char ch = line[i];

        if (isDelimiter(ch) || ch == '\0' || ch == '\n') {
            if (tokenIndex > 0) {
                token[tokenIndex] = '\0';

                if (isKeyword(token)) {
                    printf("[Keyword]        %s\n", token);
                } else if (isNumber(token)) {
                    printf("[Number]         %s\n", token);
                } else {
                    printf("[Identifier]     %s\n", token);
                }

                totalTokens++;
                tokenIndex = 0;
            }

            if (isOperator(ch)) {
                printf("[Operator]       %c\n", ch);
                totalTokens++;
            } else if (ch == ';' || ch == ',' || ch == '(' || ch == ')' ||
                       ch == '{' || ch == '}' || ch == '[' || ch == ']') {
                printf("[Special Symbol] %c\n", ch);
                totalTokens++;
            }
        } else {
            token[tokenIndex++] = ch;
        }

        i++;
    }
}

int main() {
    char line[256];
    FILE *file;

    file = fopen("input.cpp", "r");

    if (file == NULL) {
        printf("Error: Cannot open file input.cpp\n");
        return 1;
    }

    printf("Tokens identified in input.cpp:\n\n");

    while (fgets(line, sizeof(line), file)) {
        extractTokens(line);
    }

    fclose(file);

    printf("\nTotal number of tokens: %d\n", totalTokens);

    return 0;
}
