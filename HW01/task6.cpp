#include <iostream>
#include <cstdio>
#include <cstdlib>


int main(int argc, char* argv[]) {          // integer retern type of main function, argc is argument count, argv is argument vector
    int N = atoi(argv[1]);                  // convert argument from string to integer

    for (int i = 0; i <= N; i++) {           // for loop from 0 to N
        printf("%d ", i);                   // print the value of i followed by a space
    }
    printf("\n");                           // print a newline character after the loop

    for (int i = N; i>= 0; i--) {            // for loop from N to 0
        std::cout << i << " ";              // print the value of i followed by a space
    }
    std::cout << std::endl;                 // prints new line and flushes output

    return 0;                               // ends program
}
